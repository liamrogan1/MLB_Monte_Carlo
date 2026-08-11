import math
import os
import pickle
from scipy.stats import gaussian_kde, poisson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from import_odds import latest_snapshot

team_abbr = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}

BASES = np.array([0, 1, 2, 3, 4])
LEAGUE_HIT_SHARES = np.array(
    [0.63, 0.20, 0.02, 0.15]
)  # 1B,2B,3B,HR among hits (fallback)


# Odds-ratio (log5). Use ONLY for bounded probabilities: K%, BB%, BIP%.
# If the hitter is exactly league average, this returns the pitcher's rate.
def combine_prob(h, p, league):
    if pd.isnull(league):
        raise ValueError("combine_prob requires a league baseline")
    if pd.isnull(h) and pd.isnull(p):
        return league
    if pd.isnull(h):
        return p
    if pd.isnull(p):
        return h
    num = (h * p) / league
    den = num + ((1 - h) * (1 - p)) / (1 - league)
    return num / den


# Multiplicative. Use for rates NOT bounded by 1: xSLG, pitches/PA.
# Same fixed point as above (h == league  ->  returns p), but stays sane when
# est_slg > 1.0, which happens on ~200 rows of the arsenal files.
def combine_rate_mult(h, p, league, lo=0.0, hi=2.5):
    if pd.isnull(league) or league == 0:
        raise ValueError("combine_rate_mult requires a non-zero league baseline")
    if pd.isnull(h) and pd.isnull(p):
        return league
    if pd.isnull(h):
        return p
    if pd.isnull(p):
        return h
    return float(np.clip((h * p) / league, lo, hi))


def get_league_avg_by_pitch(date: str, min_pa: int = 250) -> dict:
    """League xSLG by pitch type, PA-weighted. Returns {pitch_type: xSLG} with an 'ALL' key."""
    base = "./data/daily_data"
    df = pd.concat(
        [
            pd.read_csv(f"{base}/statcast_pitcher_arsenal_{date}.csv"),
            pd.read_csv(f"{base}/statcast_hitter_arsenal_{date}.csv"),
        ],
        ignore_index=True,
    ).dropna(subset=["est_slg", "pa"])
    df = df[df["pa"] > 0]

    g = df.groupby("pitch_type").apply(
        lambda x: pd.Series(
            {
                "xSLG": np.average(x["est_slg"], weights=x["pa"]),
                "pa": x["pa"].sum(),
            }
        ),
        include_groups=False,
    )

    overall = np.average(df["est_slg"], weights=df["pa"])
    w = g["pa"] / (g["pa"] + min_pa)  # shrink thin pitch types toward overall
    out = (w * g["xSLG"] + (1 - w) * overall).to_dict()
    out["ALL"] = overall
    return out


def get_league_context(date: str) -> dict:
    """All league-wide baselines. Build once in monte_carlo, pass down."""
    return {
        "k": get_hitter_k_prob("LEAGUE", date),
        "bb": get_hitter_bb_prob("LEAGUE", date),
        "bip": get_hitter_bip_prob("LEAGUE", date),
        "pab": get_hitter_pitches_per_ab("LEAGUE", date),
        "xslg": get_hitter_xSLG("LEAGUE", date),
        "hbp": get_hitter_hbp_prob("LEAGUE", date),
        "by_pitch": get_league_avg_by_pitch(date),
    }


def get_player_id(player_name: str, team: str):
    pdb = pd.read_csv("./data/player_database.csv")
    row = pdb[(pdb["rotowire_name"] == player_name) & (pdb["team"] == team)]

    if row.empty:
        return -1

    return row["key_mlbam"].iloc[0]


# ---------- new getters ----------
def get_hitter_hit_type_profile(hitter_id, date: str = None) -> np.ndarray:
    """Normalized [1B,2B,3B,HR] shares among a hitter's hits, from bref counts."""
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    bh = pd.read_csv(f"./data/daily_data/bref_hitting_{date}.csv")
    row = bh[bh["mlbID"] == hitter_id]
    if row.empty or row["H"].values[0] <= 0:
        return LEAGUE_HIT_SHARES.copy()
    H, d, t, hr = (row[c].values[0] for c in ["H", "2B", "3B", "HR"])
    singles = max(H - d - t - hr, 0)
    s = np.array([singles, d, t, hr], float)
    return s / s.sum() if s.sum() > 0 else LEAGUE_HIT_SHARES.copy()


# NOTE k will be a variable we can extract from backtesting, however could be noisy with no weather data
def get_component_park_factors(team: str, k: float = 0.2):
    """Returns (scalar, [1B,2B,3B,HR] multipliers). Index 100 = neutral, damped by k."""
    pf = pd.read_csv("./data/park_factors.csv")
    r = pf[pf["Team"] == team]
    scalar = 1 + k * ((r["Park Factor"].values[0] - 100) / 100)
    comp = np.array(
        [1 + k * ((r[c].values[0] - 100) / 100) for c in ["1B", "2B", "3B", "HR"]]
    )
    return scalar, comp


# ---------- the heart: discrete BIP resolution, mean-preserved to matchup xSLG ----------
# NOTE h_max will be a variable that can be refined through backtesting against actual total bases in games
# It is 1:1 with BABIP
def draw_bip_outcome(
    hit_shares, target_xslg, park_comp, park_scalar, rng, h_max=0.7, h_min=0.1
):
    s = np.asarray(hit_shares, float) * np.asarray(
        park_comp, float
    )  # park_comp adjusts the batter's hit distribution according to how a ballpark has historically played out
    s = s / s.sum()
    mean_bph = (
        s * np.array([1, 2, 3, 4])
    ).sum()  # aquire mean bases per hit by multiplying the new distribution by the corresponding bases
    T = target_xslg * park_scalar  # park level effect
    h = max(T / mean_bph, h_min)  # P(hit|contact)
    if h > h_max:
        print(f"H eclipsed {h_max}")
    probs = np.concatenate([[1 - h], h * s])  # [out,1B,2B,3B,HR]
    idx = rng.choice(5, p=probs)
    return int(BASES[idx]), idx == 0  # bases, is_out


# ---------- atomic PA ----------
def simulate_pa(r, in_pen, rng):
    xK, xBB, xHBP, ppab, T = (
        (r["bp_xK"], r["bp_xBB"], r["bp_xHBP"], r["bp_ppab"], r["bp_T"])
        if in_pen
        else (r["xK"], r["xBB"], r["xHBP"], r["ppab"], r["T"])
    )
    pitches = int(rng.geometric(1.0 / ppab))
    u = rng.random()
    if u < xBB:
        return dict(out=False, K=False, BB=True, bases=0, pitches=pitches)
    if u < xBB + xHBP:
        return dict(out=False, K=False, BB=False, bases=0, pitches=pitches)  # HBP≈reach
    if u < xBB + xHBP + xK:
        return dict(out=True, K=True, BB=False, bases=0, pitches=pitches)
    b, is_out = draw_bip_outcome(r["hit_shares"], T, r["pf_comp"], r["pf_scalar"], rng)
    return dict(out=is_out, K=False, BB=False, bases=b, pitches=pitches)


# ---------- one game against the lineup (27 outs), boundary hook + bullpen ----------
def simulate_start_and_game(batters, outing_target, rng):
    outs = 0
    b = 0
    pc = 0
    in_pen = False
    st = dict(K=0, BB=0, BF=0, outs=0, H=0, pc=0)
    tb = np.zeros(9)
    kb = np.zeros(9)
    bbb = np.zeros(9)
    hrh = np.zeros(9)
    while outs < 27:
        inning_outs = 0
        while inning_outs < 3 and outs < 27:
            r = batters[b % 9]
            res = simulate_pa(r, in_pen, rng)
            if not in_pen:  # starter bookkeeping
                st["BF"] += 1
                st["pc"] += res["pitches"]
                pc += res["pitches"]
                st["K"] += res["K"]
                st["BB"] += res["BB"]
                st["H"] += not res["out"] and not res["BB"] and res["bases"] > 0
                st["outs"] += res["out"]
            tb[b % 9] += res["bases"]
            kb[b % 9] += res["K"]
            bbb[b % 9] += res["BB"]
            hrh[b % 9] += 1 if res["bases"] == 4 else 0
            if res["out"]:
                inning_outs += 1
                outs += 1
            b += 1
        if (
            not in_pen and pc >= outing_target and outs < 27
        ):  # v1 hook: inning boundary only
            in_pen = True
    return st, tb, kb, bbb, hrh


def build_batter_params(
    row,
    hitter_id,
    pitcher_id,
    pitching_team,
    pf_scalar,
    pf_comp,
    league,
    date: str = None,
):
    xhbp = combine_prob(
        get_hitter_hbp_prob(hitter_id, date),
        get_pitcher_hbp_prob(pitcher_id, date),
        league["hbp"],
    )
    shares = get_hitter_hit_type_profile(hitter_id, date)

    # NOTE fixed xSLG calculations by computing xK_AB which = xK / (1-xBB-xHBP)
    # xK is per PA, xSLG is per AB
    xbb, xk = row["xBB%"], row["xK%"]
    xk_ab = min(xk / max(1e-6, 1.0 - xbb - xhbp), 0.60)
    T = row["avg_xSLG"] / max(1e-6, 1.0 - xk_ab)

    xbb_bp, xk_bp = row["bp_xBB%"], row["bp_xK%"]
    xk_ab_bp = min(
        xk_bp / max(1e-6, 1.0 - xbb_bp - 0.01), 0.60
    )  # TODO Using 0.01 for bullpen HBP since we have not calculated that
    T_bp = row["avg_xSLG"] / max(1e-6, 1.0 - xk_ab_bp)

    return dict(
        xK=row["xK%"],
        xBB=row["xBB%"],
        xHBP=xhbp or 0.01,
        ppab=row["xPpAB"],
        T=T,
        bp_xK=row["bp_xK%"],
        bp_xBB=row["bp_xBB%"],
        bp_xHBP=0.01,
        bp_ppab=row["bp_xPpAB"],
        bp_T=T_bp,
        hit_shares=shares,
        pf_scalar=pf_scalar,
        pf_comp=pf_comp,
    )


# Given the lineup 1 through 9 and the opposing pitcher, come up with expected K%, BB%, P/AB, xSLG vs each pitch...
def get_lineup_dataframe(lineup: pd.DataFrame, date: str, league: dict) -> pd.DataFrame:
    results = []
    for _, row in lineup.iterrows():
        hitter_name = row["Player"]
        pitcher_name = row["Opposing Pitcher"]
        pitching_team = row["Opponent"]
        hitting_team = row["Team"]

        hitter_id = get_player_id(hitter_name, hitting_team)
        pitcher_id = get_player_id(pitcher_name, pitching_team)

        # Hitter base stats
        hitter_k = get_hitter_k_prob(hitter_id, date)
        hitter_bb = get_hitter_bb_prob(hitter_id, date)
        hitter_bip = get_hitter_bip_prob(hitter_id, date)
        hitter_pab = get_hitter_pitches_per_ab(hitter_id, date)
        hitter_xslg = get_hitter_xSLG(hitter_id, date)

        # Pitcher base stats
        pitcher_k = get_pitcher_k_prob(pitcher_id, date)
        pitcher_bb = get_pitcher_bb_prob(pitcher_id, date)
        pitcher_bip = get_pitcher_bip_prob(pitcher_id, date)
        pitcher_pab = get_pitcher_pitches_per_ab(pitcher_id, date)

        # Bullpen base stats
        bp_k = get_bullpen_k_prob(pitching_team, date)
        bp_bb = get_bullpen_bb_prob(pitching_team, date)
        bp_pab = get_bullpen_pitches_per_ab(pitching_team, date)
        bp_xslg = get_bullpen_xSLG(pitching_team, date)

        # Arsenal xSLG stats
        # Dict Pitch_type: [usage, xSLG]
        hitter_arsenal = get_hitter_arsenal(hitter_id, date)
        pitcher_arsenal = get_pitcher_arsenal(pitcher_id, date)

        # Dict Pitch_type: [usage, xSLG]
        pitch_xslg = get_matchup_xslg_by_pitch(
            hitter_arsenal, pitcher_arsenal, league["by_pitch"]
        )

        # Compute each expected rate once, then reuse for the diff column.
        # Probabilities -> odds ratio. Unbounded rates -> multiplicative.
        x_k = combine_prob(hitter_k, pitcher_k, league["k"])
        x_bb = combine_prob(hitter_bb, pitcher_bb, league["bb"])
        x_bip = combine_prob(hitter_bip, pitcher_bip, league["bip"])
        x_pab = combine_rate_mult(
            hitter_pab, pitcher_pab, league["pab"], lo=2.0, hi=5.0
        )

        results.append(
            {
                "Date": date,
                "Player": hitter_name,
                "Team": row["Team"],
                "Lineup Position": row["Lineup Position"],
                "xK%": x_k,
                "K%": hitter_k,
                "xK - K": (x_k - hitter_k if pd.notnull(hitter_k) else None),
                "xBB%": x_bb,
                "BB%": hitter_bb,
                "xBB - BB": (x_bb - hitter_bb if pd.notnull(hitter_bb) else None),
                "xBIP%": x_bip,
                "BIP%": hitter_bip,
                "xBIP - BIP": (x_bip - hitter_bip if pd.notnull(hitter_bip) else None),
                "xPpAB": x_pab,
                **{f"xSLG_{typ}": xSLG for typ, (_, xSLG) in pitch_xslg.items()},
                "avg_xSLG": (
                    np.sum([u * xSLG for u, xSLG in pitch_xslg.values()])
                    if pitch_xslg
                    else None
                ),
                "szn_xSLG": hitter_xslg,
                "bp_xK%": combine_prob(hitter_k, bp_k, league["k"]),
                "bp_xBB%": combine_prob(hitter_bb, bp_bb, league["bb"]),
                "bp_xPpAB": combine_rate_mult(
                    hitter_pab, bp_pab, league["pab"], lo=2.0, hi=5.0
                ),
                "bp_xSLG": combine_rate_mult(hitter_xslg, bp_xslg, league["xslg"]),
            }
        )

    new_df = pd.DataFrame(results)

    return new_df


# Dict pitch type: [usage, xSLG]
def get_matchup_xslg_by_pitch(
    hitter_arsenal: dict, pitcher_arsenal: dict, league_avg: dict
) -> dict:

    # --- Handle missing arsenals ---
    if pitcher_arsenal is None and hitter_arsenal is None:
        return {}

    if pitcher_arsenal is None:
        total = sum(v[0] for v in hitter_arsenal.values()) or 1
        return {
            pitch: [vals[0] / total, vals[1]]
            for pitch, vals in hitter_arsenal.items()
            if pd.notnull(vals[1])
        }

    matchup = {}

    for pitch_type, (p_usage, p_xslg) in pitcher_arsenal.items():
        lg = league_avg.get(pitch_type, league_avg["ALL"])

        p_val = p_xslg if pd.notnull(p_xslg) else None

        h_val = None
        if hitter_arsenal and pitch_type in hitter_arsenal:
            hv = hitter_arsenal[pitch_type][1]
            h_val = hv if pd.notnull(hv) else None

        # No fallback assignment needed: combine_rate_mult returns the league
        # value when both sides are missing, and the other side when one is.
        matchup[pitch_type] = [p_usage, combine_rate_mult(h_val, p_val, lg)]

    return matchup


def get_american_odds(probability):
    if probability == 0:
        return 0
    american_odds = (
        (100 * (1 / probability)) - 100
        if probability < 0.5
        else -(probability / (1 - probability)) * 100
    )
    return round(american_odds, 0)


def get_kelly(odds, probability, p=0.5):
    b = odds - 1

    kelly_fraction = (b * probability - (1 - probability)) / b
    return min(max(kelly_fraction * p, 0), 0.1)  # No negative bets


def compare_odds_pitcher(
    date, team, is_home, pitcher_name, pitcher_ks_history, pitcher_bb_history
):
    pitcher_ks_history = np.asarray(pitcher_ks_history)
    pitcher_bb_history = np.asarray(pitcher_bb_history)
    odds_df = pd.read_csv(f"./data/odds/{date}_props.csv")
    for col in ["pred_prob", "kelly", "bet_size"]:
        if col not in odds_df.columns:
            odds_df[col] = np.nan

    # Keep only the most recent snapshot of each unique line (original indices preserved)
    key_cols = ["date", "home_team", "away_team", "type", "player", "side", "point"]
    if "fetched_at" in odds_df.columns:
        latest = odds_df.sort_values("fetched_at").drop_duplicates(
            subset=key_cols, keep="last"
        )
    else:
        latest = odds_df  # old files without timestamps

    team_odds = (
        latest[latest["home_team"] == team_abbr.get(team)]
        if is_home
        else latest[latest["away_team"] == team_abbr.get(team)]
    )
    pitcher_odds = team_odds[
        team_odds["type"].isin(["pitcher strikeouts", "pitcher walks"])
    ].copy()

    # Use fuzzy matching to find the correct pitcher
    pitcher_odds["name_match"] = pitcher_odds["player"].apply(
        lambda x: fuzz.ratio(x, pitcher_name)
    )
    pitcher_odds = pitcher_odds[pitcher_odds["name_match"] > 70].copy()

    # Using pitcher_ks_history, calculate the probability of the pitcher going over and under each line and compare to the odds
    # "side" is either over or under

    for idx, row in pitcher_odds.iterrows():
        point = row["point"]

        if row["type"] == "pitcher strikeouts":
            if row["side"] == "over":
                prob_ks = np.mean(pitcher_ks_history > point)
                if prob_ks <= 0 or prob_ks >= 1:
                    continue
                odds_df.loc[idx, "pred_prob"] = round(1 / (prob_ks), 2)
                odds_df.loc[idx, "kelly"] = get_kelly(row["price"], prob_ks)
                odds_df.loc[idx, "bet_size"] = 1000 * get_kelly(row["price"], prob_ks)
                pitcher_odds.loc[idx, "pred_prob"] = round(1 / (prob_ks), 2)
                pitcher_odds.loc[idx, "bet_size"] = 1000 * get_kelly(
                    row["price"], prob_ks
                )
            elif row["side"] == "under":
                prob_ks = np.mean(pitcher_ks_history < point)
                if prob_ks <= 0 or prob_ks >= 1:
                    continue
                odds_df.loc[idx, "pred_prob"] = round(1 / prob_ks, 2)
                odds_df.loc[idx, "kelly"] = get_kelly(row["price"], prob_ks)
                odds_df.loc[idx, "bet_size"] = 1000 * get_kelly(row["price"], prob_ks)
                pitcher_odds.loc[idx, "pred_prob"] = round(1 / prob_ks, 2)
                pitcher_odds.loc[idx, "bet_size"] = 1000 * get_kelly(
                    row["price"], prob_ks
                )
        if row["type"] == "pitcher walks":
            if row["side"] == "over":
                prob_bbs = np.mean(pitcher_bb_history > point)
                if prob_bbs <= 0 or prob_bbs >= 1:
                    continue
                odds_df.loc[idx, "pred_prob"] = round(1 / (prob_bbs), 2)
                odds_df.loc[idx, "kelly"] = get_kelly(row["price"], prob_bbs)
                odds_df.loc[idx, "bet_size"] = 1000 * get_kelly(row["price"], prob_bbs)
                pitcher_odds.loc[idx, "pred_prob"] = round(1 / (prob_bbs), 2)
                pitcher_odds.loc[idx, "bet_size"] = 1000 * get_kelly(
                    row["price"], prob_bbs
                )
            elif row["side"] == "under":
                prob_bbs = np.mean(pitcher_bb_history < point)
                if prob_bbs <= 0 or prob_bbs >= 1:
                    continue
                odds_df.loc[idx, "pred_prob"] = round(1 / prob_bbs, 2)
                odds_df.loc[idx, "kelly"] = get_kelly(row["price"], prob_bbs)
                odds_df.loc[idx, "bet_size"] = 1000 * get_kelly(row["price"], prob_bbs)
                pitcher_odds.loc[idx, "pred_prob"] = round(1 / prob_bbs, 2)
                pitcher_odds.loc[idx, "bet_size"] = 1000 * get_kelly(
                    row["price"], prob_bbs
                )
    # Convert price and pred_prob to american odds in print statement need to divide by 1 to get probability
    pitcher_odds["price"] = pitcher_odds["price"].apply(
        lambda x: get_american_odds(1 / x)
    )
    pitcher_odds["pred_prob"] = pitcher_odds["pred_prob"].apply(
        lambda x: get_american_odds(1 / x)
    )
    print(
        pitcher_odds[pitcher_odds["bet_size"] > 0][
            ["player", "type", "side", "point", "price", "pred_prob", "bet_size"]
        ]
    )

    odds_df.to_csv(f"./data/odds/{date}_props.csv", index=False)


# TODO
# 1. Cannot determine how many pitches a bullpen will throw, NEED TO TRACK OUTS
# 2. Add in park factors
# 3. Add in platoon splits
# 4. Add in weather
# NOTE Not accounting for HBP, IBB,...
def monte_carlo(date: str, plot_p=False, plot_b=False, odds=False):
    lineups_data = pd.read_csv(f"./data/lineups/{date}_lineup.csv")
    league = get_league_context(date)

    matchup_file = pd.DataFrame()
    ml_prob = {}
    hfa = 0.0125

    # Iterate through each lineup assuming each team has 9 players
    for i in range(0, len(lineups_data), 9):
        lineup = lineups_data.iloc[i : i + 9]
        lineup_stats_df = get_lineup_dataframe(lineup, date, league)
        matchup_file = pd.concat([matchup_file, lineup_stats_df], ignore_index=True)

        batting_team = lineup["Team"].values[0]
        pitching_team = lineup["Opponent"].values[0]
        is_home = lineup["Is Home"].values[0]
        park_factor = get_park_factor(batting_team if is_home else pitching_team)

        pitcher_name = lineup["Opposing Pitcher"].values[0]
        pitcher_arsenal = get_pitcher_arsenal(
            get_player_id(pitcher_name, pitching_team), date
        )
        if pitcher_arsenal is None or len(pitcher_arsenal) <= 1:
            continue

        ml_prob.setdefault(batting_team, 0)
        ml_prob.setdefault(pitching_team, 0)

        avg_pitches_per_game, pc_std = get_pitcher_pc_distribution(
            get_player_id(pitcher_name, pitching_team), date
        )
        print(f"{pitcher_name} vs {batting_team} lineup")
        arsenal_print = ""
        for pitch, vals in pitcher_arsenal.items():
            arsenal_print += f"{pitch}: usage {vals[0]:.2f}, xSLG {vals[1]:.3f} / "

        print(arsenal_print)
        pitch_usages = [v[0] for v in pitcher_arsenal.values()]

        # Create a dictionary with batter name and expected total bases, strikeouts, and walks
        hitter_expected_bases = {}
        hitter_expected_strikeouts = {}
        hitter_expected_walks = {}

        pitcher_ks_history = []
        pitcher_pc_history = []
        pitcher_bbs_history = []

        # Run Monte Carlo simulation through entire lineup n games
        for n in range(15000):
            # Batter tracking
            game_total_bases = {}
            game_total_strikeouts = {}
            game_total_walks = {}

            # TODO Understand this logic and make more flexible / efficient
            # Draw K% and BB% adjustments from a covariance matrix
            sigma_k = 0.15
            sigma_bb = 0.20  # BB% is slightly more volatile than K% across starts

            # Negative correlation: good K day -> fewer walks
            correlation = -0.2

            cov_matrix = [
                [sigma_k**2, correlation * sigma_k * sigma_bb],
                [correlation * sigma_k * sigma_bb, sigma_bb**2],
            ]

            # Single draw produces both modifiers for the game
            k_noise, bb_noise = np.random.multivariate_normal([0, 0], cov_matrix)

            starter_k_modifier = np.exp(k_noise)  # lognormal keeps values positive
            starter_bb_modifier = np.exp(bb_noise)

            # Starter tracking
            starter_strikeouts = 0
            starter_walks = 0
            starter_batters_faced = 0
            starter_bips = 0
            starter_times_through_order = 0

            # Pitch Distribution
            game_total_pitches = np.random.normal(160, round(pc_std * 1.5, 0))
            pitch_count_st = int(
                np.random.normal(
                    max(avg_pitches_per_game, 65), max(min(pc_std, 30), 10)
                ),
            )

            # Run through entire lineup until pitcher has run out of pitches
            total_starter_pitches = 0
            batter_number = 0
            while pitch_count_st > 0:
                batter_row = lineup_stats_df.iloc[batter_number]
                batter_name = batter_row["Player"]
                starter_batters_faced += 1

                # Get the average amount of pitches per at-bat for hitter vs pitcher using previously calculated adjustments
                ab_end_prob = 1 / (batter_row["xPpAB"])
                raw_K = batter_row["xK%"] * starter_k_modifier
                raw_BB = batter_row["xBB%"] * starter_bb_modifier
                raw_BIP = max(1 - raw_K - raw_BB, 0.05)

                total = raw_K + raw_BB + raw_BIP
                K_prob = raw_K / total
                BB_prob = raw_BB / total
                BIP_prob = raw_BIP / total

                # Determine AB outcome type upfront
                rand_outcome = np.random.uniform(0, 1)
                if rand_outcome <= BB_prob:
                    ab_outcome = "BB"
                    min_pitches = 4
                elif rand_outcome <= BB_prob + K_prob:
                    ab_outcome = "K"
                    min_pitches = 3
                else:
                    ab_outcome = "BIP"
                    min_pitches = 1

                # Now simulate pitches until the outcome resolves
                pitch_count_AB = 0
                batter_strikeout = 0
                batter_walk = 0
                expected_bases = 0
                at_bat = True
                while at_bat:
                    pitch_type = np.random.choice(
                        list(pitcher_arsenal.keys()), p=pitch_usages
                    )
                    pitch_count_AB += 1
                    total_starter_pitches += 1

                    rand_ab = np.random.uniform(0, 1)
                    if pitch_count_AB >= min_pitches and rand_ab <= ab_end_prob:
                        at_bat = False
                        if ab_outcome == "BB":
                            starter_walks += 1
                            batter_walk += 1
                        elif ab_outcome == "K":
                            starter_strikeouts += 1
                            batter_strikeout += 1
                        else:
                            expected_bases = (
                                batter_row[f"xSLG_{pitch_type}"] * park_factor
                            )
                            starter_bips += 1

                # Instead of just keeping a total, append to a list to get distribution later
                if batter_name not in game_total_bases:
                    game_total_bases[batter_name] = 0
                if batter_name not in game_total_strikeouts:
                    game_total_strikeouts[batter_name] = 0
                if batter_name not in game_total_walks:
                    game_total_walks[batter_name] = 0

                game_total_bases[batter_name] += expected_bases
                game_total_strikeouts[batter_name] += batter_strikeout
                game_total_walks[batter_name] += batter_walk

                # Remove pitches used in this at-bat from the total pitch count
                pitch_count_st -= pitch_count_AB

                # Go to next batter
                if batter_number < 8:
                    batter_number = batter_number + 1
                else:
                    batter_number = 0
                    starter_times_through_order += 1

            # Now move onto the bullpen
            pitches_left = game_total_pitches - total_starter_pitches
            while pitches_left > 0:
                batter_row = lineup_stats_df.iloc[batter_number]
                batter_name = batter_row["Player"]

                ab_end_prob = 1 / batter_row["bp_xPpAB"]
                K_prob = batter_row["bp_xK%"]
                BB_prob = batter_row["bp_xBB%"]

                # Simulate the at-bat
                # Determine AB outcome type upfront
                rand_outcome = np.random.uniform(0, 1)
                if rand_outcome <= BB_prob:
                    ab_outcome = "BB"
                    min_pitches = 4
                elif rand_outcome <= BB_prob + K_prob:
                    ab_outcome = "K"
                    min_pitches = 3
                else:
                    ab_outcome = "BIP"
                    min_pitches = 1

                # Now simulate pitches until the outcome resolves
                pitch_count_AB = 0
                batter_strikeout = 0
                batter_walk = 0
                expected_bases = 0
                at_bat = True
                while at_bat:
                    pitch_type = np.random.choice(
                        list(pitcher_arsenal.keys()), p=pitch_usages
                    )
                    pitch_count_AB += 1

                    rand_ab = np.random.uniform(0, 1)
                    if pitch_count_AB >= min_pitches and rand_ab <= ab_end_prob:
                        at_bat = False
                        if ab_outcome == "BB":
                            batter_walk += 1
                        elif ab_outcome == "K":
                            batter_strikeout += 1
                        else:
                            expected_bases = (
                                batter_row[f"xSLG_{pitch_type}"] * park_factor
                            )

                pitches_left -= pitch_count_AB

                if batter_name not in game_total_bases:
                    game_total_bases[batter_name] = 0
                if batter_name not in game_total_strikeouts:
                    game_total_strikeouts[batter_name] = 0
                if batter_name not in game_total_walks:
                    game_total_walks[batter_name] = 0

                game_total_bases[batter_name] += expected_bases
                game_total_strikeouts[batter_name] += batter_strikeout
                game_total_walks[batter_name] += batter_walk

                # Go to next batter
                batter_number = batter_number + 1 if batter_number < 8 else 0

            # Add game totals to hitter_expected_bases
            for hitter in game_total_bases:
                if hitter not in hitter_expected_bases:
                    hitter_expected_bases[hitter] = []
                if hitter not in hitter_expected_strikeouts:
                    hitter_expected_strikeouts[hitter] = []
                if hitter not in hitter_expected_walks:
                    hitter_expected_walks[hitter] = []
                hitter_expected_bases[hitter].append(game_total_bases[hitter])
                hitter_expected_strikeouts[hitter].append(game_total_strikeouts[hitter])
                hitter_expected_walks[hitter].append(game_total_walks[hitter])

            # Add pitcher history
            pitcher_ks_history.append(starter_strikeouts)
            pitcher_bbs_history.append(starter_walks)
            pitcher_pc_history.append(total_starter_pitches)

        # Print pitchers stats
        mean_ks = np.mean(pitcher_ks_history)
        std_ks = np.std(pitcher_ks_history)
        mean_pc = np.mean(pitcher_pc_history)
        std_pc = np.std(pitcher_pc_history)
        mean_bbs = np.mean(pitcher_bbs_history)
        std_bbs = np.std(pitcher_bbs_history)

        # Compare pitcher strikeouts and walks to current odds
        if odds:
            compare_odds_pitcher(
                date,
                lineups_data.iloc[i]["Opponent"],
                1 if not is_home else 0,
                pitcher_name,
                np.array(pitcher_ks_history),
                np.array(pitcher_bbs_history),
            )

        # Print mean and median expected bases for each hitter
        cum_mean = 0
        cum_median = 0
        print("------------------------------------------------------")
        for hitter in hitter_expected_bases:
            expected_bases_list = np.array(hitter_expected_bases[hitter])
            expected_strikeouts_list = np.array(hitter_expected_strikeouts[hitter])
            expected_walks_list = np.array(hitter_expected_walks[hitter])
            mean_expected_bases = np.mean(expected_bases_list)
            median_expected_bases = np.median(expected_bases_list)
            std_expected_bases = np.std(expected_bases_list)

            # Calculate the percent chance of going over total bases with poisson distribution
            prob_over_1_5 = np.mean(expected_bases_list > 1.5)
            prob_hitter_k = np.mean(expected_strikeouts_list > 0.5)
            prob_hitter_k2 = np.mean(expected_strikeouts_list > 1.5)
            prob_hitter_walk = np.mean(expected_walks_list > 0.5)
            print(
                f"{hitter}: {mean_expected_bases:.3f} mean | [o1.5 TB {round(get_american_odds(prob_over_1_5),0)}] o0.5BBs {round(get_american_odds(prob_hitter_walk),0)} |  o0.5Ks {round(get_american_odds(prob_hitter_k),0)} | o1.5Ks {round(get_american_odds(prob_hitter_k2),0)}"
            )
            cum_mean += mean_expected_bases
            cum_median += median_expected_bases
            if plot_b:
                plt.rcParams.update(
                    {
                        "figure.facecolor": "black",
                        "axes.facecolor": "black",
                        "axes.labelcolor": "white",
                        "xtick.color": "white",
                        "ytick.color": "white",
                        "text.color": "white",
                        "axes.edgecolor": "white",
                        "font.family": "monospace",
                        "font.monospace": [
                            "Cascadia Code",
                            "DejaVu Sans Mono",
                            "Consolas",
                        ],
                    }
                )

                data = np.array(expected_bases_list)

                kde = gaussian_kde(data)
                x_vals = np.linspace(data.min(), data.max(), 500)
                y_vals = kde(x_vals)

                plt.plot(x_vals, y_vals, linewidth=2, color="springgreen")

                plt.xlabel("Total Bases")
                plt.ylabel("Density")
                plt.title(f"{hitter} Total Bases Distribution vs {pitcher_name} {date}")

                # Mean + std lines
                plt.axvline(
                    mean_expected_bases, linestyle="solid", linewidth=3, color="tomato"
                )
                plt.axvline(
                    mean_expected_bases + std_expected_bases,
                    linestyle=(0, (1, 4)),
                    linewidth=3,
                    color="tomato",
                )
                plt.axvline(
                    mean_expected_bases - std_expected_bases,
                    linestyle=(0, (1, 4)),
                    linewidth=3,
                    color="tomato",
                )

                plt.show()

        print(f"Lineup Total: {cum_mean:.3f} mean, {cum_median:.3f} median")

        # Moneyline tracking
        ml_prob[batting_team] += cum_mean
        if ml_prob[batting_team] > 0 and ml_prob[pitching_team] > 0:
            print(
                f"{batting_team} {get_american_odds((ml_prob[batting_team] / (ml_prob[batting_team] + ml_prob[pitching_team])) + hfa)} vs {pitching_team} {get_american_odds((ml_prob[pitching_team] / (ml_prob[batting_team] + ml_prob[pitching_team])) - hfa)}"
            )

        print(f"Pitcher Stats: Mean Ks = {mean_ks:.2f}, STD = {std_ks:.2f}")

        # Calculate pitcher strikeout and bb odds with poisson distribution using mean
        pitcher_whole_mid = math.floor(mean_ks) + 0.5
        pitcher_whole_upper = pitcher_whole_mid + 1
        pitcher_whole_lower = pitcher_whole_mid - 1
        ks_array = np.array(pitcher_ks_history)
        bbs_array = np.array(pitcher_bbs_history)
        pitcher_prob_ks = np.mean(ks_array > pitcher_whole_mid)
        pitcher_prob_ks_upper = np.mean(ks_array > pitcher_whole_upper)
        pitcher_prob_ks_lower = np.mean(ks_array > pitcher_whole_lower)
        pitcher_prob_bbs_lower = np.mean(bbs_array > 1.5)
        pitcher_prob_bbs_upper = np.mean(bbs_array > 2.5)

        print(
            f"Pitcher odds to go o{pitcher_whole_lower}: {get_american_odds(pitcher_prob_ks_lower)} [o{pitcher_whole_mid}: {get_american_odds(pitcher_prob_ks)}] o{pitcher_whole_upper}: {get_american_odds(pitcher_prob_ks_upper)}"
        )
        print(f"Pitcher Stats: Mean BBs = {mean_bbs:.2f}, STD = {std_bbs:.2f}")
        print(
            f"Pitcher odds to go o1.5: {get_american_odds(pitcher_prob_bbs_lower)} o2.5: {get_american_odds(pitcher_prob_bbs_upper)}"
        )
        print(f"Pitcher Stats: Mean PC = {mean_pc:.2f}, STD = {std_pc:.2f}")
        print()

        # Plot pitcher strikeouts histogram
        if plot_p:
            plt.rcParams.update(
                {
                    "figure.facecolor": "black",
                    "axes.facecolor": "black",
                    "axes.labelcolor": "white",
                    "xtick.color": "white",
                    "ytick.color": "white",
                    "text.color": "white",
                    "axes.edgecolor": "white",
                    "font.family": "monospace",
                    "font.monospace": ["Cascadia Code", "DejaVu Sans Mono", "Consolas"],
                }
            )

            bins = range(min(pitcher_ks_history), max(pitcher_ks_history) + 2)
            plt.hist(
                pitcher_ks_history,
                bins=bins,
                edgecolor="springgreen",
                facecolor="none",
                linewidth=1.5,
            )
            centers = np.array(list(bins[:-1])) + 0.5
            plt.xticks(centers, bins[:-1])
            plt.xlabel("Strikeouts")
            plt.ylabel("Frequency")
            plt.title(
                f"{pitcher_name} Strikeouts Distribution vs {batting_team} {date}"
            )

            plt.axvline(mean_ks, color="tomato", linestyle="solid", linewidth=3)
            plt.axvline(
                mean_ks + std_ks,
                color="tomato",
                linestyle=(0, (1, 4)),
                linewidth=3,
            )
            plt.axvline(
                mean_ks - std_ks,
                color="tomato",
                linestyle=(0, (1, 4)),
                linewidth=3,
            )
            plt.show()

    matchup_file.to_csv(f"./data/todays_matchups.csv", index=False)


def monte_carlo_outs(
    date,
    outing_ratios,
    league,
    n_sims=15000,
    seed=0,
    plot_p=False,
    plot_b=False,
    odds=False,
):
    rng = np.random.default_rng(seed)
    lineups_data = pd.read_csv(f"./data/lineups/{date}_lineup.csv")
    matchup_file = pd.DataFrame()
    ml_prob = {}
    hfa = 0.0125

    # Iterate through each lineup assuming each team has 9 players
    for i in range(0, len(lineups_data), 9):
        lineup = lineups_data.iloc[i : i + 9]
        df = get_lineup_dataframe(lineup, date, league)
        pitching_team = lineup["Opponent"].values[0]
        pitcher_name = lineup["Opposing Pitcher"].values[0]
        pitcher_id = get_player_id(pitcher_name, pitching_team)
        batting_team = lineup["Team"].values[0]
        is_home = lineup["Is Home"].values[0]
        pf_scalar, pf_comp = get_component_park_factors(
            batting_team if is_home else pitching_team
        )

        batters = [
            build_batter_params(
                df.iloc[j],
                get_player_id(df.iloc[j]["Player"], batting_team),
                pitcher_id,
                pitching_team,
                pf_scalar,
                pf_comp,
                league,
                date,
            )
            for j in range(9)
        ]
        base_pc, std = get_pitcher_pc_distribution(pitcher_id, date)

        matchup_file = pd.concat([matchup_file, df], ignore_index=True)

        pitcher_arsenal = get_pitcher_arsenal(pitcher_id, date)
        if pitcher_arsenal is None:
            continue

        ml_prob.setdefault(batting_team, 0)
        ml_prob.setdefault(pitching_team, 0)
        print(f"{pitcher_name} vs {batting_team} lineup")
        arsenal_print = ""
        for pitch, vals in pitcher_arsenal.items():
            arsenal_print += f"{pitch}: usage {vals[0]:.2f}, xSLG {vals[1]:.3f} / "

        print(arsenal_print)

        # Create a dictionary with batter name and expected total bases, strikeouts, and walks
        hitter_expected_bases = {}
        hitter_expected_strikeouts = {}
        hitter_expected_walks = {}
        hitter_expected_hrs = {}

        pitcher_ks_history = []
        pitcher_pc_history = []
        pitcher_bbs_history = []
        pitcher_outs_history = []

        # ---- v1 out-based simulation (replaces the old pitch-budget starter + bullpen loops) ----
        for n in range(n_sims):
            target = base_pc * rng.choice(outing_ratios)  # calibrated left-skew hook
            st, tb, kb, bbb, hrh = simulate_start_and_game(batters, target, rng)

            pitcher_ks_history.append(st["K"])
            pitcher_bbs_history.append(st["BB"])
            pitcher_pc_history.append(st["pc"])
            pitcher_outs_history.append(st["outs"])

            for j, name in enumerate(df["Player"]):
                hitter_expected_bases.setdefault(name, []).append(tb[j])
                hitter_expected_strikeouts.setdefault(name, []).append(kb[j])
                hitter_expected_walks.setdefault(name, []).append(bbb[j])
                hitter_expected_hrs.setdefault(name, []).append(hrh[j])

        # Print pitchers stats
        mean_ks = np.mean(pitcher_ks_history)
        std_ks = np.std(pitcher_ks_history)
        mean_bbs = np.mean(pitcher_bbs_history)
        std_bbs = np.std(pitcher_bbs_history)
        mean_pc = np.mean(pitcher_pc_history)
        std_pc = np.std(pitcher_pc_history)
        mean_outs = np.mean(pitcher_outs_history)
        std_outs = np.std(pitcher_outs_history)

        # Compare pitcher strikeouts and walks to current odds
        if odds:
            compare_odds_pitcher(
                date,
                lineups_data.iloc[i]["Opponent"],
                1 if not is_home else 0,
                pitcher_name,
                np.array(pitcher_ks_history),
                np.array(pitcher_bbs_history),
            )

        # Print mean and median expected bases for each hitter
        cum_mean = 0
        cum_median = 0
        print("------------------------------------------------------")
        for hitter in hitter_expected_bases:
            expected_bases_list = np.array(hitter_expected_bases[hitter])
            expected_hrs_list = np.array(hitter_expected_hrs[hitter])
            expected_strikeouts_list = np.array(hitter_expected_strikeouts[hitter])
            expected_walks_list = np.array(hitter_expected_walks[hitter])
            mean_expected_bases = np.mean(expected_bases_list)
            median_expected_bases = np.median(expected_bases_list)
            std_expected_bases = np.std(expected_bases_list)

            # Calculate the percent chance of going over total bases with poisson distribution
            prob_over_1_5 = np.mean(expected_bases_list > 1.5)
            prob_hr = np.mean(expected_hrs_list > 0)
            prob_hitter_k = np.mean(expected_strikeouts_list > 0.5)
            prob_hitter_k2 = np.mean(expected_strikeouts_list > 1.5)
            prob_hitter_walk = np.mean(expected_walks_list > 0.5)
            print(
                f"{hitter}: {mean_expected_bases:.3f} mean | [o1.5 TB {round(get_american_odds(prob_over_1_5),0)}] HR {round(get_american_odds(prob_hr),0)} || o0.5BBs {round(get_american_odds(prob_hitter_walk),0)} |  o0.5Ks {round(get_american_odds(prob_hitter_k),0)} | o1.5Ks {round(get_american_odds(prob_hitter_k2),0)}"
            )
            cum_mean += mean_expected_bases
            cum_median += median_expected_bases
            if plot_b:
                plt.rcParams.update(
                    {
                        "figure.facecolor": "black",
                        "axes.facecolor": "black",
                        "axes.labelcolor": "white",
                        "xtick.color": "white",
                        "ytick.color": "white",
                        "text.color": "white",
                        "axes.edgecolor": "white",
                        "font.family": "monospace",
                        "font.monospace": [
                            "Cascadia Code",
                            "DejaVu Sans Mono",
                            "Consolas",
                        ],
                    }
                )

                data = np.array(expected_bases_list)

                kde = gaussian_kde(data)
                x_vals = np.linspace(data.min(), data.max(), 500)
                y_vals = kde(x_vals)

                plt.plot(x_vals, y_vals, linewidth=2, color="springgreen")

                plt.xlabel("Total Bases")
                plt.ylabel("Density")
                plt.title(f"{hitter} Total Bases Distribution vs {pitcher_name} {date}")

                # Mean + std lines
                plt.axvline(
                    mean_expected_bases, linestyle="solid", linewidth=3, color="tomato"
                )
                plt.axvline(
                    mean_expected_bases + std_expected_bases,
                    linestyle=(0, (1, 4)),
                    linewidth=3,
                    color="tomato",
                )
                plt.axvline(
                    mean_expected_bases - std_expected_bases,
                    linestyle=(0, (1, 4)),
                    linewidth=3,
                    color="tomato",
                )

                plt.show()

        print(f"Lineup Total: {cum_mean:.3f} mean, {cum_median:.3f} median")

        # Moneyline tracking
        ml_prob[batting_team] += cum_mean
        if ml_prob[batting_team] > 0 and ml_prob[pitching_team] > 0:
            print(
                f"{batting_team} {get_american_odds((ml_prob[batting_team] / (ml_prob[batting_team] + ml_prob[pitching_team])) + hfa)} vs {pitching_team} {get_american_odds((ml_prob[pitching_team] / (ml_prob[batting_team] + ml_prob[pitching_team])) - hfa)}"
            )

        print(f"Pitcher Stats: Mean Ks = {mean_ks:.2f}, STD = {std_ks:.2f}")

        pitcher_whole_mid = math.floor(mean_ks) + 0.5
        pitcher_whole_upper = pitcher_whole_mid + 1
        pitcher_whole_lower = pitcher_whole_mid - 1
        ks_array = np.array(pitcher_ks_history)
        bbs_array = np.array(pitcher_bbs_history)
        pitcher_prob_ks = np.mean(ks_array > pitcher_whole_mid)
        pitcher_prob_ks_upper = np.mean(ks_array > pitcher_whole_upper)
        pitcher_prob_ks_lower = np.mean(ks_array > pitcher_whole_lower)
        pitcher_prob_bbs_lower = np.mean(bbs_array > 1.5)
        pitcher_prob_bbs_upper = np.mean(bbs_array > 2.5)

        print(
            f"Pitcher odds to go o{pitcher_whole_lower}: {get_american_odds(pitcher_prob_ks_lower)} [o{pitcher_whole_mid}: {get_american_odds(pitcher_prob_ks)}] o{pitcher_whole_upper}: {get_american_odds(pitcher_prob_ks_upper)}"
        )
        print(f"Pitcher Stats: Mean BBs = {mean_bbs:.2f}, STD = {std_bbs:.2f}")
        print(
            f"Pitcher odds to go o1.5: {get_american_odds(pitcher_prob_bbs_lower)} o2.5: {get_american_odds(pitcher_prob_bbs_upper)}"
        )
        print(f"Pitcher Stats: Mean PC = {mean_pc:.2f}, STD = {std_pc:.2f}")
        print(f"Pitcher Stats: Mean Outs = {mean_outs:.2f}, STD = {std_outs:.2f}")
        print()

        # Plot pitcher strikeouts histogram
        if plot_p:
            plt.rcParams.update(
                {
                    "figure.facecolor": "black",
                    "axes.facecolor": "black",
                    "axes.labelcolor": "white",
                    "xtick.color": "white",
                    "ytick.color": "white",
                    "text.color": "white",
                    "axes.edgecolor": "white",
                    "font.family": "monospace",
                    "font.monospace": ["Cascadia Code", "DejaVu Sans Mono", "Consolas"],
                }
            )

            bins = range(min(pitcher_ks_history), max(pitcher_ks_history) + 2)
            plt.hist(
                pitcher_ks_history,
                bins=bins,
                edgecolor="springgreen",
                facecolor="none",
                linewidth=1.5,
            )
            centers = np.array(list(bins[:-1])) + 0.5
            plt.xticks(centers, bins[:-1])
            plt.xlabel("Strikeouts")
            plt.ylabel("Frequency")
            plt.title(
                f"{pitcher_name} Strikeouts Distribution vs {batting_team} {date}"
            )

            plt.axvline(mean_ks, color="tomato", linestyle="solid", linewidth=3)
            plt.axvline(
                mean_ks + std_ks,
                color="tomato",
                linestyle=(0, (1, 4)),
                linewidth=3,
            )
            plt.axvline(
                mean_ks - std_ks,
                color="tomato",
                linestyle=(0, (1, 4)),
                linewidth=3,
            )
            plt.show()

    matchup_file.to_csv(f"./data/todays_matchups.csv", index=False)


def get_park_factor(team: str) -> float:
    park_factors = pd.read_csv("./data/park_factors.csv")
    team_park_factor = park_factors[park_factors["Team"] == team]
    factor = team_park_factor["Park Factor"].values[0]
    k = 0.2  # Adjust this value to control how much the park factor influences the expected bases

    # # Mexico City Game
    # if team == "SDP" or team == "ARI":
    #     factor = 150
    #     k = 0.75
    return 1 + k * ((factor - 100) / 100)


# X NOTE Need to account for exact same rotowire names with team check
def get_pitcher_arsenal(pitcher_id: str, date: str = None) -> dict:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    pitcher_arsenal = {}

    statcast_pitching = pd.read_csv(
        f"./data/daily_data/statcast_pitcher_arsenal_{date}.csv"
    )

    pitcher_data = statcast_pitching[statcast_pitching["player_id"] == pitcher_id]
    if pitcher_data.empty:
        return None

    for _, row in pitcher_data.iterrows():
        pitch_type = row["pitch_type"]
        usage = row["normalized_blended_pitch_usage"] / 100
        xSLG = row["blended_est_slg"]
        pitcher_arsenal[pitch_type] = [usage, xSLG]

    return pitcher_arsenal


# X
def get_hitter_arsenal(hitter_id: str, date: str = None) -> dict:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    hitter_arsenal = {}

    statcast_hitting = pd.read_csv(
        f"./data/daily_data/statcast_hitter_arsenal_{date}.csv"
    )

    hitter_data = statcast_hitting[statcast_hitting["player_id"] == hitter_id]
    if hitter_data.empty:
        return None

    for _, row in hitter_data.iterrows():
        pitch_type = row["pitch_type"]
        usage = row["pitch_usage"]
        xSLG = row["blended_est_slg"]
        hitter_arsenal[pitch_type] = [usage, xSLG]

    total_usage = sum(v[0] for v in hitter_arsenal.values())
    if total_usage > 0:
        for pitch_type in hitter_arsenal:
            hitter_arsenal[pitch_type][0] /= total_usage

    return hitter_arsenal


def get_bullpen_k_prob(team: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    bullpen_pitchers = pd.read_csv(f"./data/daily_data/bullpen_activity_{date}.csv")
    pitching_basic = pd.read_csv(f"./data/daily_data/bref_pitching_{date}.csv")

    team_bullpen = bullpen_pitchers[bullpen_pitchers["team"] == team]
    if team_bullpen.empty:
        return None
    bullpen_k_probs = []
    for _, row in team_bullpen.iterrows():
        weight = row["weight"]
        pitcher_name = row["player"]
        pitcher_id = get_player_id(pitcher_name, team)
        pitcher_stats = pitching_basic[pitching_basic["mlbID"] == pitcher_id]
        if pitcher_stats.empty or pitcher_stats["BF"].values[0] == 0:
            continue
        k_prob = pitcher_stats["blended_Kp"].values[0]
        bullpen_k_probs.append(k_prob * weight)

    if len(bullpen_k_probs) == 0:
        return None
    return np.sum(bullpen_k_probs)


def get_bullpen_bb_prob(team: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    bullpen_pitchers = pd.read_csv(f"./data/daily_data/bullpen_activity_{date}.csv")
    pitching_basic = pd.read_csv(f"./data/daily_data/bref_pitching_{date}.csv")

    team_bullpen = bullpen_pitchers[bullpen_pitchers["team"] == team]
    if team_bullpen.empty:
        return None
    bullpen_bb_probs = []
    for _, row in team_bullpen.iterrows():
        weight = row["weight"]
        pitcher_name = row["player"]
        pitcher_id = get_player_id(pitcher_name, team)
        pitcher_stats = pitching_basic[pitching_basic["mlbID"] == pitcher_id]
        if pitcher_stats.empty or pitcher_stats["BF"].values[0] == 0:
            continue
        bb_prob = pitcher_stats["blended_BBp"].values[0]
        bullpen_bb_probs.append(bb_prob * weight)

    if len(bullpen_bb_probs) == 0:
        return None
    return np.sum(bullpen_bb_probs)


def get_bullpen_xSLG(team: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    bullpen_pitchers = pd.read_csv(f"./data/daily_data/bullpen_activity_{date}.csv")
    pitching_basic = pd.read_csv(f"./data/daily_data/statcast_pitching_{date}.csv")

    team_bullpen = bullpen_pitchers[bullpen_pitchers["team"] == team]
    if team_bullpen.empty:
        return None
    bullpen_xSLG = []
    for _, row in team_bullpen.iterrows():
        weight = row["weight"]
        pitcher_name = row["player"]
        pitcher_id = get_player_id(pitcher_name, team)
        pitcher_stats = pitching_basic[pitching_basic["player_id"] == pitcher_id]
        if pitcher_stats.empty:
            continue
        xSLG = pitcher_stats["blended_est_slg"].values[0]
        bullpen_xSLG.append(xSLG * weight)

    if len(bullpen_xSLG) == 0:
        return None
    return np.sum(bullpen_xSLG)


def get_bullpen_pitches_per_ab(team: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    bullpen_pitchers = pd.read_csv(f"./data/daily_data/bullpen_activity_{date}.csv")
    pitching_basic = pd.read_csv(f"./data/daily_data/bref_pitching_{date}.csv")

    team_bullpen = bullpen_pitchers[bullpen_pitchers["team"] == team]
    if team_bullpen.empty:
        return None
    bullpen_pitches_per_ab = []
    for _, row in team_bullpen.iterrows():
        weight = row["weight"]
        pitcher_name = row["player"]
        pitcher_id = get_player_id(pitcher_name, team)
        pitcher_stats = pitching_basic[pitching_basic["mlbID"] == pitcher_id]
        if pitcher_stats.empty or pitcher_stats["BF"].values[0] == 0:
            continue
        pitches_per_ab = pitcher_stats["Pit"].values[0] / pitcher_stats["BF"].values[0]
        bullpen_pitches_per_ab.append(pitches_per_ab * weight)

    if len(bullpen_pitches_per_ab) == 0:
        return None
    return np.sum(bullpen_pitches_per_ab)


def get_pitcher_pc_distribution(
    pitcher_id: int, date: str = None
) -> tuple[float, float]:
    """
    Returns (median_pc, std_pc) using TTO-anchored floor/ceiling as guardrails
    rather than raw Q1/Q3 which can be skewed by small samples.
    """
    pitching_log = pd.read_csv("./data/pitcher_outing_logs.csv")

    pitcher_log = pitching_log[pitching_log["mlbID"] == pitcher_id].sort_values("date")
    starts = pitcher_log[pitcher_log["is_start"] == 1]

    # TTO-based guardrails using pitcher's own P/BF if available
    p_per_bf = get_pitcher_pitches_per_ab(pitcher_id, date)

    floor_pc = (
        1.5 * 9 * p_per_bf
    )  # 1.5 times through the order (9 batters * 2 times * avg pitchcount)
    ceiling_pc = (
        3.5 * 9 * p_per_bf
    )  # 3.5 times through the order (9 batters * 3 times * avg pitchcount)

    if len(starts) >= 10:
        sample = starts.tail(10)["pitch_count"]
    elif len(starts) > 0:
        # Pad with relief outings but clip at floor so they don't drag median down
        relief = pitcher_log[pitcher_log["is_start"] == 0].tail(10 - len(starts))
        relief_clipped = relief["pitch_count"].clip(lower=floor_pc * 0.5)
        sample = pd.concat([starts["pitch_count"], relief_clipped])
    else:
        # No starts at all — debut projection
        # Or relief pitcher
        # TODO Came across Easton McGee, strictly a relief pitcher so returned the floor and 10
        sample = pd.Series([floor_pc])

    median_pc = np.clip(sample.median(), floor_pc, ceiling_pc)
    raw_std = sample.std()  # ddof=1 → NaN on a single observation
    std_pc = 8.0 if pd.isna(raw_std) else max(raw_std, 8.0)
    return round(median_pc), round(std_pc)


def get_pitcher_pitches_per_ab(pitcher_id: str, date: str = None) -> float:
    """
    Returns a pitcher's average pitches per batter faced (bounded by [2,6]) unless
    the pitcher cannot be found within the data files
    """
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    pitching_basic = pd.read_csv(f"./data/daily_data/bref_pitching_{date}.csv")
    pitcher_data = pitching_basic[pitching_basic["mlbID"] == pitcher_id]

    if (
        pitcher_id == "LEAGUE"
        or pitcher_data.empty
        or pitcher_data["BF"].values[0] == 0
    ):
        return pitching_basic["Pit"].sum() / pitching_basic["BF"].sum()

    return min(max(pitcher_data["Pit"].values[0] / pitcher_data["BF"].values[0], 2), 6)


# K / BF X
def get_pitcher_k_prob(pitcher_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")

    pitching_basic = pd.read_csv(f"./data/daily_data/bref_pitching_{date}.csv")
    pitcher_data = pitching_basic[pitching_basic["mlbID"] == pitcher_id]
    if pitcher_data.empty or pitcher_data["BF"].values[0] == 0:
        return None

    # TODO Change/remove when Mexico City games are over
    return pitcher_data["blended_Kp"].values[0]


# BB / BF X
def get_pitcher_bb_prob(pitcher_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")

    pitching_basic = pd.read_csv(f"./data/daily_data/bref_pitching_{date}.csv")
    pitcher_data = pitching_basic[pitching_basic["mlbID"] == pitcher_id]
    if pitcher_data.empty or pitcher_data["BF"].values[0] == 0:
        return None

    return pitcher_data["blended_BBp"].values[0]


def get_pitcher_hbp_prob(pitcher_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")

    pitching_basic = pd.read_csv(f"./data/daily_data/bref_pitching_{date}.csv")
    pitcher_data = pitching_basic[pitching_basic["mlbID"] == pitcher_id]
    if pitcher_data.empty or pitcher_data["BF"].values[0] == 0:
        return None

    return pitcher_data["HBP"].values[0] / pitcher_data["BF"].values[0]


def get_hitter_hbp_prob(hitter_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    hitting_basic = pd.read_csv(f"./data/daily_data/bref_hitting_{date}.csv")
    if hitter_id == "LEAGUE":
        m = hitting_basic["PA"] > 0
        return hitting_basic.loc[m, "HBP"].sum() / hitting_basic.loc[m, "PA"].sum()
    hitter_data = hitting_basic[hitting_basic["mlbID"] == hitter_id]
    if hitter_data.empty or hitter_data["PA"].values[0] == 0:
        return None
    return hitter_data["HBP"].values[0] / hitter_data["PA"].values[0]


# Total Pitches / PA X
def get_hitter_pitches_per_ab(hitter_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    bref_df = pd.read_csv(f"./data/daily_data/bref_hitting_{date}.csv")
    sc_df = pd.read_csv(f"./data/daily_data/statcast_hitter_arsenal_{date}.csv")

    if hitter_id == "LEAGUE":
        return sc_df["pitches"].sum() / bref_df["PA"].sum()

    hitter_bref = bref_df[bref_df["mlbID"] == hitter_id]
    hitter_sc = sc_df[sc_df["player_id"] == hitter_id]
    if hitter_bref.empty or hitter_sc.empty or hitter_bref["PA"].values[0] == 0:
        return None

    total_pitches = hitter_sc["pitches"].sum()
    return total_pitches / hitter_bref["PA"].values[0]


# SO / PA X
def get_hitter_k_prob(hitter_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")

    stats_df = pd.read_csv(f"./data/daily_data/bref_hitting_{date}.csv")
    if hitter_id == "LEAGUE":
        m = np.isfinite(stats_df["blended_Kp"])
        return np.average(stats_df.loc[m, "blended_Kp"], weights=stats_df.loc[m, "PA"])
    hitter_data = stats_df[stats_df["mlbID"] == hitter_id]
    if hitter_data.empty or hitter_data["PA"].values[0] == 0:
        return None

    return hitter_data["blended_Kp"].values[0]


# BB / PA X
def get_hitter_bb_prob(hitter_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")

    stats_df = pd.read_csv(f"./data/daily_data/bref_hitting_{date}.csv")
    if hitter_id == "LEAGUE":
        m = np.isfinite(stats_df["blended_BBp"])
        return np.average(stats_df.loc[m, "blended_BBp"], weights=stats_df.loc[m, "PA"])

    hitter_data = stats_df[stats_df["mlbID"] == hitter_id]
    if hitter_data.empty or hitter_data["PA"].values[0] == 0:
        return None

    return hitter_data["blended_BBp"].values[0]


# est SLG overall X
def get_hitter_xSLG(hitter_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")

    hitting_basic = pd.read_csv(f"./data/daily_data/statcast_hitting_{date}.csv")
    if hitter_id == "LEAGUE":
        m = np.isfinite(hitting_basic["blended_est_slg"])
        return np.average(
            hitting_basic.loc[m, "blended_est_slg"], weights=hitting_basic.loc[m, "bip"]
        )
    hitter_data = hitting_basic[hitting_basic["player_id"] == hitter_id]
    if hitter_data.empty:
        return None

    return hitter_data["blended_est_slg"].values[0]


# BIP / PA X
def get_hitter_bip_prob(hitter_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")
    hitting_basic = pd.read_csv(f"./data/daily_data/statcast_hitting_{date}.csv")
    if hitter_id == "LEAGUE":
        return hitting_basic["bip"].sum() / hitting_basic["pa"].sum()
    hitter_data = hitting_basic[hitting_basic["player_id"] == hitter_id]
    if hitter_data.empty or hitter_data["pa"].values[0] == 0:
        return None

    return hitter_data["bip"].values[0] / hitter_data["pa"].values[0]


# BIP / PA X
def get_pitcher_bip_prob(pitcher_id: str, date: str = None) -> float:
    date = date or pd.to_datetime("today").strftime("%Y%m%d")

    pitcher_basic = pd.read_csv(f"./data/daily_data/statcast_pitching_{date}.csv")
    pitcher_data = pitcher_basic[pitcher_basic["player_id"] == pitcher_id]
    if pitcher_data.empty or pitcher_data["pa"].values[0] == 0:
        return None

    return pitcher_data["bip"].values[0] / pitcher_data["pa"].values[0]


def against_market_results(
    date=(pd.to_datetime("today") - pd.Timedelta(days=1)).strftime("%Y%m%d"),
    bankroll=1000,
):
    results_file = f"./data/against_market_results.csv"
    results_df = pd.read_csv(results_file)

    if os.path.exists(f"./data/odds/{date}_props.csv"):
        odds_df = pd.read_csv(f"./data/odds/{date}_props.csv")
        odds_df = odds_df.dropna(subset=["kelly", "result"])
        # Get only the rows with the most recent fetched_at
        odds_df = latest_snapshot(odds_df)

        total_risked = odds_df[odds_df["kelly"] > 0]["kelly"].sum() * bankroll
        bets_made = odds_df[odds_df["kelly"] > 0]
        win = 0
        loss = 0
        rolling_bankroll = bankroll
        for idx, row in bets_made.iterrows():
            bet_amount = row["kelly"] * bankroll
            if row["side"] == "over":
                if row["result"] > row["point"]:
                    rolling_bankroll += bet_amount * (row["price"] - 1)
                    win += 1
                    # print(
                    #     f"Win: {row['player']} {row['side']}{row['point']}{"K" if row["type"] == "pitcher strikeouts" else "BB"}s at {row['price']} | Bet Amount: ${bet_amount:.2f} | Profit: ${rolling_bankroll - 1000:.2f}"
                    # )
                else:
                    rolling_bankroll -= bet_amount
                    loss += 1
                    # print(
                    #     f"Loss: {row['player']} {row['side']}{row['point']}{"K" if row["type"] == "pitcher strikeouts" else "BB"}s at {row['price']} | Bet Amount: ${bet_amount:.2f} | Profit: ${rolling_bankroll - 1000:.2f}"
                    # )
            elif row["side"] == "under":
                if row["result"] < row["point"]:
                    rolling_bankroll += bet_amount * (row["price"] - 1)
                    win += 1
                    # print(
                    #     f"Win: {row['player']} {row['side']}{row['point']}{"K" if row["type"] == "pitcher strikeouts" else "BB"}s at {row['price']} | Bet Amount: ${bet_amount:.2f} | Profit: ${rolling_bankroll - 1000:.2f}"
                    # )
                else:
                    rolling_bankroll -= bet_amount
                    loss += 1
                    # print(
                    #     f"Loss: {row['player']} {row['side']}{row['point']}{"K" if row["type"] == "pitcher strikeouts" else "BB"}s at {row['price']} | Bet Amount: ${bet_amount:.2f} | Profit: ${rolling_bankroll - 1000:.2f}"
                    # )
        roi = (rolling_bankroll - 1000) / total_risked if total_risked > 0 else 0
        print(
            f"Total Risked: ${total_risked:.2f}, Record: {win}-{loss}, Final Bankroll: ${rolling_bankroll:.2f}, ROI: {roi:.2%}"
        )

        # Save to master file
        if int(date) not in results_df["date"].values:
            header = not os.path.exists(results_file)
            df_to_add = pd.DataFrame(
                [
                    {
                        "date": date,
                        "risked": total_risked,
                        "win": win,
                        "loss": loss,
                        "profit": rolling_bankroll - bankroll,
                        "roi": roi,
                    }
                ],
                columns=["date", "risked", "win", "loss", "profit", "roi"],
            )
            df_to_add.to_csv(
                results_file,
                mode="a",
                header=header,
                index=False,
            )
            results_df = pd.concat([results_df, df_to_add], ignore_index=True)
    print(
        f"All Time {len(results_df)}days :: Record: {results_df['win'].sum()}-{results_df['loss'].sum()}, Profit: ${results_df['profit'].sum():.2f}, ROI: {(results_df['profit'].sum() / results_df['risked'].sum()):.2%}"
    )
    print(
        f"Rolling L5 Record: {results_df.tail(5)['win'].sum()}-{results_df.tail(5)['loss'].sum()}, Profit: ${results_df.tail(5)['profit'].sum():.2f}, ROI: {(results_df.tail(5)['profit'].sum() / results_df.tail(5)['risked'].sum()):.2%}"
    )
    print(
        f"Rolling L10 Record: {results_df.tail(10)['win'].sum()}-{results_df.tail(10)['loss'].sum()}, Profit: ${results_df.tail(10)['profit'].sum():.2f}, ROI: {(results_df.tail(10)['profit'].sum() / results_df.tail(10)['risked'].sum()):.2%}\n"
    )


if __name__ == "__main__":
    curr_date = pd.to_datetime("today").strftime("%Y%m%d")

    # 1) Outing-length shape for the hook (left-skewed, ~CV 0.14), built once
    logs = pd.read_csv("./data/pitcher_outing_logs.csv")
    valid = logs[logs["date"] < int(curr_date)]
    st = valid[valid["is_start"] == 1]
    g = st.groupby("mlbID")["pitch_count"]
    keep = g.count()[g.count() >= 8].index
    outing_ratios = np.concatenate(
        [
            st.loc[st["mlbID"] == p, "pitch_count"].values
            / st.loc[st["mlbID"] == p, "pitch_count"].mean()
            for p in keep
        ]
    )

    # 2) League baselines for combine_prob / combine_rate_mult
    pa = pd.read_csv(f"./data/daily_data/statcast_pitcher_arsenal_{curr_date}.csv")
    w = pa.dropna(subset=["est_slg"])
    by_pitch = (
        w.groupby("pitch_type")
        .apply(lambda d: np.average(d["est_slg"], weights=d["pitches"]))
        .to_dict()
    )
    by_pitch["ALL"] = np.average(
        w["est_slg"], weights=w["pitches"]
    )  # fallback for unseen pitch types

    league = {
        "k": get_hitter_k_prob("LEAGUE", curr_date),
        "bb": get_hitter_bb_prob("LEAGUE", curr_date),
        "bip": get_hitter_bip_prob("LEAGUE", curr_date),
        "pab": get_hitter_pitches_per_ab("LEAGUE", curr_date),
        "xslg": get_hitter_xSLG("LEAGUE", curr_date),
        "hbp": get_hitter_hbp_prob("LEAGUE", curr_date),
        "by_pitch": by_pitch,
    }

    against_market_results()
    monte_carlo_outs(
        date=curr_date,
        outing_ratios=outing_ratios,
        league=league,
        n_sims=15000,
        plot_p=False,
        plot_b=False,
        odds=False,
    )
