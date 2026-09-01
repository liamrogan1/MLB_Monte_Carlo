from pathlib import Path
import re
import numpy as np
import pybaseball
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import time
import os
from fuzzywuzzy import fuzz
import unicodedata
from datetime import datetime

team_abbr = {
    "Diamondbacks": "ARI",
    "Braves": "ATL",
    "Orioles": "BAL",
    "Red Sox": "BOS",
    "Cubs": "CHC",
    "White Sox": "CWS",
    "Reds": "CIN",
    "Guardians": "CLE",
    "Rockies": "COL",
    "Tigers": "DET",
    "Astros": "HOU",
    "Royals": "KC",
    "Angels": "LAA",
    "Dodgers": "LAD",
    "Marlins": "MIA",
    "Brewers": "MIL",
    "Twins": "MIN",
    "Mets": "NYM",
    "Yankees": "NYY",
    "Athletics": "ATH",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Padres": "SDP",
    "Giants": "SFG",
    "Mariners": "SEA",
    "Cardinals": "STL",
    "Rays": "TB",
    "Rangers": "TEX",
    "Blue Jays": "TOR",
    "Nationals": "WSN",
}

# Matches literal escape text like "\xc3\xa9" that pybaseball's bref
# scraper leaves in names (bytes repr'd into the string).
_ESCAPE_RE = re.compile(r"\\x[0-9a-fA-F]{2}")


def clean_name(name: str) -> str:
    """Normalize a player name for matching.

    Handles three layers of mess:
      1. Literal '\\xc3\\xa9'-style escape text -> decoded to real chars
      2. Accented chars -> ASCII (Pérez -> Perez, Muñoz -> Munoz)
      3. Case, punctuation, Jr./Sr./II suffixes stripped
    """
    if not isinstance(name, str):
        return ""

    # 1. Decode literal escape sequences if present
    if _ESCAPE_RE.search(name):
        try:
            name = (
                name.encode("latin-1")
                .decode("unicode_escape")
                .encode("latin-1")
                .decode("utf-8")
            )
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Fall back to just deleting the escape text
            name = _ESCAPE_RE.sub("", name)

    # 2. Strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # 3. Lowercase, drop punctuation, drop generational suffixes
    name = name.lower()
    name = re.sub(r"[.\-']", " ", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# TODO Add in cancellation check (name not in lineup csv)
def match_player(
    player_name: str,
    candidates: pd.DataFrame,
    name_col: str = "Name",
    threshold: int = 85,
) -> tuple[int | None, str | None, int]:
    """Fuzzy-match an odds-feed name to a row in `candidates`.

    Returns (mlbID, matched_display_name, score) or (None, best_name, score)
    if nothing clears the threshold. Matching is done on cleaned names, and
    the return key is mlbID -- never the name string -- so downstream
    lookups are collision-proof.
    """
    target = clean_name(player_name)

    best_score = -1
    best_idx = None
    for idx, raw in candidates[name_col].items():
        score = fuzz.ratio(target, clean_name(raw))
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None, None, 0

    best_name = candidates.loc[best_idx, name_col]
    if best_score >= threshold:
        return int(candidates.loc[best_idx, "mlbID"]), best_name, best_score
    return None, best_name, best_score


def add_prop_results():
    yesterday = (pd.to_datetime("today") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yday = (pd.to_datetime("today") - pd.Timedelta(days=1)).strftime("%Y%m%d")

    if not Path(f"./data/odds/{yday}_props.csv").is_file():
        print("No props gathered yesterday")
        return

    odds_df = pd.read_csv(f"./data/odds/{yday}_props.csv")

    if "result" not in odds_df.columns:
        odds_df["result"] = np.nan
    elif odds_df["result"].notna().any():
        print(f"Odds have already been graded for {yesterday}")
        return

    yesterday_results = pybaseball.pitching_stats_range(yesterday, yesterday)
    yesterday_results["mlbID"] = pd.to_numeric(
        yesterday_results["mlbID"], errors="coerce"
    )

    day_totals = (
        yesterday_results.dropna(subset=["mlbID"])
        .groupby(["mlbID", "Name"], as_index=False)[["SO", "BB"]]
        .sum()
    )
    day_totals["mlbID"] = day_totals["mlbID"].astype(int)

    print(f"Grading props for {yday}")

    for prop_type in ["pitcher strikeouts", "pitcher walks"]:
        subset = odds_df[odds_df["type"] == prop_type]
        if subset.empty:
            continue
        stat_col = "SO" if prop_type == "pitcher strikeouts" else "BB"

        print(f"\nGrading {prop_type}:")
        for player_name in subset["player"].unique():
            mlb_id, matched_name, score = match_player(player_name, day_totals)
            row_idx = subset[subset["player"] == player_name].index

            if mlb_id is None:
                odds_df.loc[row_idx, "result"] = np.nan
                closest = (
                    f"closest: {matched_name} ({score})"
                    if matched_name
                    else "no candidates"
                )
                print(f"  {player_name:<25} -> NO MATCH ({closest}) -- left as NaN")
                continue

            result_rows = day_totals.loc[day_totals["mlbID"] == mlb_id, stat_col]
            if result_rows.empty:
                odds_df.loc[row_idx, "result"] = np.nan
                print(
                    f"  {player_name:<25} -> id {mlb_id} not in results -- left as NaN"
                )
                continue

            result = result_rows.iloc[0]
            odds_df.loc[row_idx, "result"] = result
            flag = "" if score == 100 else f"  [fuzzy {score}]"
            print(f"  {player_name:<25} -> {matched_name:<25} {result:>3.0f} {flag}")

    odds_df.to_csv(f"./data/odds/{yday}_props.csv", index=False)
    graded = odds_df["result"].notna().sum()
    print(f"\nSaved: {graded}/{len(odds_df)} lines graded.")


def save_relevant_data(year, date=pd.to_datetime("today").strftime("%Y%m%d")):
    if Path(f"./data/daily_data/statcast_hitter_arsenal_{date}.csv").is_file():
        print("Stats for today already gathered")
        return
    print("Saving bullpen data...")
    get_bullpen_data(date)

    print("Saving bref hitting data...")
    hitting_data = pybaseball.batting_stats_bref(year)
    hitting_data.to_csv(f"./data/daily_data/bref_hitting_{date}.csv")
    time.sleep(10)

    print("Saving statcast hitting data...")
    hitting_sc_data = pybaseball.statcast_batter_expected_stats(year, minPA=1)
    hitting_sc_data.to_csv(f"./data/daily_data/statcast_hitting_{date}.csv")
    time.sleep(10)

    print("Saving bref pitching data...")
    pitching_data = pybaseball.pitching_stats_bref(year)
    pitching_data.to_csv(f"./data/daily_data/bref_pitching_{date}.csv")
    add_pc_data()
    time.sleep(10)

    print("Saving statcast pitching data...")
    pitching_sc_data = pybaseball.statcast_pitcher_expected_stats(year, minPA=1)
    pitching_sc_data.to_csv(f"./data/daily_data/statcast_pitching_{date}.csv")
    time.sleep(10)

    print("Saving statcast pitcher arsenal data...")
    pitcher_arsenal_data = pybaseball.statcast_pitcher_arsenal_stats(2026, minPA=1)
    pitcher_arsenal_data.to_csv(
        f"./data/daily_data/statcast_pitcher_arsenal_{date}.csv"
    )
    time.sleep(10)

    print("Saving statcast hitter arsenal data...")
    hitter_arsenal_data = pybaseball.statcast_batter_pitch_arsenal(2026, minPA=1)
    hitter_arsenal_data.to_csv(f"./data/daily_data/statcast_hitter_arsenal_{date}.csv")
    time.sleep(10)

    print("Blending hitter K%...")
    blend_hitter_k(date)
    print("Blending hitter BB%...")
    blend_hitter_bb(date)
    print("Blending pitcher K%...")
    blend_pitcher_k(date)
    print("Blending pitcher BB%...")
    blend_pitcher_bb(date)
    print("Blending pitcher arsenal usage...")
    blend_pitcher_arsenal_usage(date)
    print("Blending pitcher arsenal xSLG...")
    blend_pitcher_arsenal_xSLG(date)
    print("Blending hitter arsenal xSLG...")
    blend_hitter_arsenal_xSLG(date)
    print("Blending hitter overall xSLG...")
    blend_hitter_xslg(date)
    print("Blending pitcher overall xSLG...")
    blend_pitcher_xslg(date)


# TODO Add in last available data instead of just using yesterday
def add_pc_data(
    date=(pd.to_datetime("today") - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
):
    log_path = "./data/pitcher_outing_logs.csv"

    pitching = pybaseball.pitching_stats_range(date, date)

    if pitching.empty:
        print(f"No data for {date}")
        return

    new_rows = pitching[["mlbID", "Name", "G", "GS", "IP", "BF", "Pit"]].copy()
    new_rows["date"] = date.replace("-", "")
    new_rows["pitch_count"] = new_rows["Pit"]
    ip10 = (new_rows["IP"] * 10).round().astype(int)
    new_rows["Outs"] = (ip10 // 10) * 3 + (ip10 % 10)
    new_rows = new_rows[
        ["mlbID", "Name", "date", "G", "GS", "IP", "Outs", "BF", "pitch_count"]
    ]

    existing_log = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
    _flush_outing_rows([new_rows], existing_log, log_path)
    print(f"Added {len(new_rows)} outings for {date}")


# gets the current date's lineups and the next days if available
# TODO Add in functionality to skip days that don't have games
def get_lineups():
    ROTOWIRE_URLS = [
        "https://www.rotowire.com/baseball/daily-lineups.php",
        "https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow",
    ]
    lineup_dfs = []
    headers = requests.utils.default_headers()
    headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0",
        }
    )
    for url in ROTOWIRE_URLS:
        rotowire_link = requests.get(url, headers=headers)
        rotowire_content = bs(rotowire_link.text, "lxml")

        master_df = pd.DataFrame()

        # Retrieve lineups
        game_pattern = re.compile(r"^lineup is-mlb")
        games = rotowire_content.find_all("div", class_=game_pattern)[:-1]
        if len(games) == 0:
            print(f"No games found for URL: {url}")
            continue

        for game in games:
            # Extract and clean visiting team name
            visit_team_raw = game.find(
                "div", class_="lineup__mteam is-visit"
            ).text.strip()
            visit_team_name = re.sub(r"\s*\(.*\)", "", visit_team_raw).strip()
            visit_team = team_abbr.get(visit_team_name)

            # Extract and clean home team name
            home_team_raw = game.find(
                "div", class_="lineup__mteam is-home"
            ).text.strip()
            home_team_name = re.sub(r"\s*\(.*\)", "", home_team_raw).strip()
            home_team = team_abbr.get(home_team_name)

            # Get lineup for visiting team
            visit_lineup = game.find("ul", class_="lineup__list is-visit")
            if visit_lineup is None:
                print(
                    f"No visiting lineup found for {visit_team_name} vs {home_team_name}"
                )
                continue
            visit_players = visit_lineup.find_all("a")
            visit_names = [
                [i, a.get_text(strip=True)] for i, a in enumerate(visit_players)
            ]
            visit_df = pd.DataFrame(visit_names, columns=["Lineup Position", "Player"])

            # Get lineup for home team
            home_lineup = game.find("ul", class_="lineup__list is-home")
            home_players = home_lineup.find_all("a")
            home_names = [
                [i, a.get_text(strip=True)] for i, a in enumerate(home_players)
            ]
            home_df = pd.DataFrame(home_names, columns=["Lineup Position", "Player"])

            # Get the opposing pitcher (first player in the other team's lineup)
            visit_opp_pitcher = home_names[0][1] if home_names else None
            home_opp_pitcher = visit_names[0][1] if visit_names else None

            # Add the opposing pitcher as a new column
            visit_df["Opposing Pitcher"] = visit_opp_pitcher
            home_df["Opposing Pitcher"] = home_opp_pitcher

            master_df = pd.concat(
                [
                    master_df,
                    pd.DataFrame(
                        {
                            "Team": [visit_team] * len(visit_df),
                            "Opponent": [home_team] * len(visit_df),
                            "Is Home": [0] * len(visit_df),
                            "Lineup Position": visit_df["Lineup Position"],
                            "Player": visit_df["Player"],
                            "Opposing Pitcher": visit_df["Opposing Pitcher"],
                        }
                    ),
                    pd.DataFrame(
                        {
                            "Team": [home_team] * len(home_df),
                            "Opponent": [visit_team] * len(home_df),
                            "Is Home": [1] * len(home_df),
                            "Lineup Position": home_df["Lineup Position"],
                            "Player": home_df["Player"],
                            "Opposing Pitcher": home_df["Opposing Pitcher"],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Remove rows where Position is 0 (pitchers)
        master_df = master_df[master_df["Lineup Position"] != 0]
        lineup_dfs.append(master_df)

    return lineup_dfs


# Used to calculate probability a bullpen arm will be used based on fatigue level
def pitch_prob(fatigue, a=0.12, b=18):
    return 1 / (1 + np.exp(a * (fatigue - b)))


# Fangraphs has Hard Hit%, K%, BB%, TBF, Pitches (Pitches/TBF), no xSLG
# Could maybe take average xSLG and scale by Hard-Hit??
def get_bullpen_data(date):
    url = "https://www.rotowire.com/baseball/tables/bullpen-usage.php?team="

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.rotowire.com/baseball/bullpen-usage.php",
        "X-Requested-With": "XMLHttpRequest",
    }

    all_rows = []

    for team in team_abbr.values():
        if team == "WSN":
            team = "WSH"
        if team == "SDP":
            team = "SD"
        if team == "SFG":
            team = "SF"
        res = requests.get(url + team, headers=headers)
        data = res.json()

        for player in data.get(team, []):
            #  Change back the team abbreviation to match the rest of the data
            if team == "WSH":
                player["team"] = "WSN"
            elif team == "SD":
                player["team"] = "SDP"
            elif team == "SF":
                player["team"] = "SFG"
            else:
                player["team"] = team_abbr.get(player["team"], player["team"])
            all_rows.append(player)

        time.sleep(1)

    df = pd.DataFrame(all_rows)
    df["last2"] = df["day1"] + df["day2"]
    # fatigue
    df["fatigue"] = (
        1.0 * df["day1"]
        + 0.7 * df["day2"]
        + 0.5 * df["day3"]
        + 0.3 * df["day4"]
        + 0.2 * df["day5"]
    )

    # usage probability
    df["usage_prob"] = df["fatigue"].apply(pitch_prob)

    # expected pitches if used
    days = df[["day1", "day2", "day3", "day4", "day5"]]
    non_zero_counts = (days > 0).sum(axis=1)
    pitch_sums = days.sum(axis=1)

    df["exp_pitch_if_used"] = np.where(
        non_zero_counts > 0, pitch_sums / non_zero_counts, 12
    ).clip(min=12, max=35)

    # expected contribution
    df["expected_pitches_total"] = df["usage_prob"] * df["exp_pitch_if_used"]

    # If "inj" is not NaN, set weight to 0 since injured players won't be contributing
    mask = df["inj"].notna() & (df["inj"].str.strip() != "")
    df.loc[mask, "expected_pitches_total"] = 0

    # weights within team
    df["weight"] = df.groupby("team")["expected_pitches_total"].transform(
        lambda x: x / x.sum()
    )

    df.to_csv(f"./data/daily_data/bullpen_activity_{date}.csv", index=False)


# Sample size at which a stat gets 50% weight against its prior.
# Roughly the published stabilization points; tune these against your own
# backtests rather than treating them as fixed.
STAB = {
    "hitter_k": 60,
    "hitter_bb": 120,
    "hitter_xslg": 200,
    "pitcher_k": 70,
    "pitcher_bb": 170,
    "pitcher_xslg": 250,
    "arsenal_usage": 150,
    "arsenal_xslg": 300,
}


def shrink(units, k_stab):
    """Weight on the observed sample. Asymptotes to 1, never reaches it.

    Replaces the logistic curve, which hit w=1.0 exactly at max_units (prior
    weight -> 0 at a finite sample) and exceeded 1.0 beyond it (extrapolating
    past the observed value).
    """
    units = np.asarray(units, dtype=float)
    units = np.where(np.isnan(units), 0.0, units)
    return units / (units + k_stab)


def blend3(cur_units, cur, prev_units, prev, league, k_stab):
    """Three-level shrinkage: current -> prior -> league.

    The prior is itself shrunk toward league, so a player with 40 PA last year
    does not anchor this year's projection on noise. Works elementwise on
    Series or scalars.
    """
    cur = pd.Series(cur, dtype=float)
    prev = pd.Series(prev, dtype=float)
    league = pd.Series(league, dtype=float) if hasattr(league, "__len__") else league

    w_prev = pd.Series(shrink(prev_units, k_stab), index=prev.index)
    w_prev = w_prev.where(prev.notna(), 0.0)  # no prior sample -> prior IS league
    prior = w_prev * prev.fillna(0.0) + (1 - w_prev) * league

    w_cur = pd.Series(shrink(cur_units, k_stab), index=cur.index)
    w_cur = w_cur.where(cur.notna(), 0.0)  # no current sample -> fall back to prior
    return w_cur * cur.fillna(0.0) + (1 - w_cur) * prior


def age_curve(prev_df, age_col, num_col, den_col, window=1):
    """League rate by age (+/- window), computed once instead of per player."""
    ages = prev_df[age_col].dropna().unique()
    out = {}
    for a in ages:
        m = prev_df[(prev_df[age_col] >= a - window) & (prev_df[age_col] <= a + window)]
        den = m[den_col].sum()
        out[a] = m[num_col].sum() / den if den > 0 else np.nan
    return out


# =============================================================================
# REPLACES: blend_hitter_k / blend_hitter_bb / blend_pitcher_k / blend_pitcher_bb
# One vectorized function instead of four row loops.
# =============================================================================


def blend_rate(cur_path, prev_path, id_col, num_col, den_col, out_col, k_stab):
    cur = pd.read_csv(cur_path)
    prev = pd.read_csv(prev_path)

    league = cur[num_col].sum() / cur[den_col].sum()

    # Age-based prior, per the methodology doc, but built once and mapped
    by_age = age_curve(prev, "Age", num_col, den_col)
    age_rate = cur["Age"].map(by_age).fillna(league)

    prev_small = prev[[id_col, num_col, den_col]].rename(
        columns={num_col: "prev_num", den_col: "prev_den"}
    )
    m = cur[[id_col, "Age", num_col, den_col]].merge(prev_small, on=id_col, how="left")

    cur_rate = (m[num_col] / m[den_col]).replace([np.inf, -np.inf], np.nan)
    prev_rate = (m["prev_num"] / m["prev_den"]).replace([np.inf, -np.inf], np.nan)

    cur[out_col] = blend3(
        cur_units=m[den_col],
        cur=cur_rate,
        prev_units=m["prev_den"],
        prev=prev_rate,
        league=age_rate.values,  # age-adjusted league level, not a flat mean
        k_stab=k_stab,
    ).values

    cur.to_csv(cur_path, index=False)
    return cur


def blend_hitter_k(date):
    return blend_rate(
        f"./data/daily_data/bref_hitting_{date}.csv",
        "./data/bref_hitting_2025.csv",
        "mlbID",
        "SO",
        "PA",
        "blended_Kp",
        STAB["hitter_k"],
    )


def blend_hitter_bb(date):
    return blend_rate(
        f"./data/daily_data/bref_hitting_{date}.csv",
        "./data/bref_hitting_2025.csv",
        "mlbID",
        "BB",
        "PA",
        "blended_BBp",
        STAB["hitter_bb"],
    )


def blend_pitcher_k(date):
    return blend_rate(
        f"./data/daily_data/bref_pitching_{date}.csv",
        "./data/bref_pitching_2025.csv",
        "mlbID",
        "SO",
        "BF",
        "blended_Kp",
        STAB["pitcher_k"],
    )


def blend_pitcher_bb(date):
    return blend_rate(
        f"./data/daily_data/bref_pitching_{date}.csv",
        "./data/bref_pitching_2025.csv",
        "mlbID",
        "BB",
        "BF",
        "blended_BBp",
        STAB["pitcher_bb"],
    )


# =============================================================================
# REPLACES: blend_pitcher_arsenal_xSLG / blend_hitter_arsenal_xSLG
# =============================================================================


def league_xslg_by_pitch(cur, prev, min_pa=250):
    """PA-weighted league xSLG per pitch type, CURRENT season first.

    The old code took prev_year_data[pitch_type].mean(), which (a) is unweighted
    and (b) returns NaN for pitch types that did not exist last year.
    """
    both = pd.concat([cur, prev], ignore_index=True).dropna(subset=["est_slg", "pa"])
    both = both[both["pa"] > 0]
    overall = np.average(both["est_slg"], weights=both["pa"])

    g = both.groupby("pitch_type").apply(
        lambda x: pd.Series(
            {"xSLG": np.average(x["est_slg"], weights=x["pa"]), "pa": x["pa"].sum()}
        ),
        include_groups=False,
    )
    w = g["pa"] / (g["pa"] + min_pa)
    out = (w * g["xSLG"] + (1 - w) * overall).to_dict()
    out["ALL"] = overall
    return out


def blend_arsenal_xslg(cur_path, prev_path, out_col="blended_est_slg"):
    cur = pd.read_csv(cur_path)
    prev = pd.read_csv(prev_path)

    lg = league_xslg_by_pitch(cur, prev)
    lg_series = cur["pitch_type"].map(lg).fillna(lg["ALL"])

    prev_small = prev[["player_id", "pitch_type", "est_slg", "pitches"]].rename(
        columns={"est_slg": "prev_slg", "pitches": "prev_pitches"}
    )
    m = cur[["player_id", "pitch_type", "est_slg", "pitches"]].merge(
        prev_small, on=["player_id", "pitch_type"], how="left"
    )

    cur[out_col] = blend3(
        cur_units=m["pitches"],
        cur=m["est_slg"],
        prev_units=m["prev_pitches"],
        prev=m["prev_slg"],
        league=lg_series.values,
        k_stab=STAB["arsenal_xslg"],
    ).values

    # Nothing should escape as NaN. If it does, the league value is the answer.
    cur[out_col] = cur[out_col].fillna(lg_series)
    cur.to_csv(cur_path, index=False)
    return cur


def blend_pitcher_arsenal_xSLG(date):
    return blend_arsenal_xslg(
        f"./data/daily_data/statcast_pitcher_arsenal_{date}.csv",
        "./data/statcast_pitcher_arsenal_2025.csv",
    )


def blend_hitter_arsenal_xSLG(date):
    return blend_arsenal_xslg(
        f"./data/daily_data/statcast_hitter_arsenal_{date}.csv",
        "./data/statcast_hitter_arsenal_2025.csv",
    )


# =============================================================================
# REPLACES: blend_pitcher_arsenal_usage
# =============================================================================


def blend_pitcher_arsenal_usage(date):
    path = f"./data/daily_data/statcast_pitcher_arsenal_{date}.csv"
    cur = pd.read_csv(path)
    prev = pd.read_csv("./data/statcast_pitcher_arsenal_2025.csv")

    prev_small = prev[["player_id", "pitch_type", "pitch_usage", "pitches"]].rename(
        columns={"pitch_usage": "prev_usage", "pitches": "prev_pitches"}
    )
    m = cur[["player_id", "pitch_type", "pitch_usage", "pitches"]].merge(
        prev_small, on=["player_id", "pitch_type"], how="left"
    )

    # A pitch type absent from last year blends against 0 usage, not league mean:
    # "he did not throw it" is real information, unlike a missing xSLG.
    w_prev = pd.Series(shrink(m["prev_pitches"], STAB["arsenal_usage"]))
    prior = w_prev * m["prev_usage"].fillna(0.0)

    w_cur = pd.Series(shrink(m["pitches"], STAB["arsenal_usage"]))
    blended = w_cur * m["pitch_usage"].fillna(0.0) + (1 - w_cur) * prior

    cur["_blended"] = blended.values
    total = cur.groupby("player_id")["_blended"].transform("sum")
    cur["normalized_blended_pitch_usage"] = np.where(
        total > 0, cur["_blended"] / total * 100, 0.0
    )
    cur = cur.drop(columns="_blended")
    cur.to_csv(path, index=False)
    return cur


# =============================================================================
# REPLACES: blend_hitter_xslg   (season-wide xSLG -> szn_xSLG / bp_xSLG)
# NEW:      blend_pitcher_xslg  (statcast_pitching was never blended at all)
# =============================================================================


def blend_season_xslg(cur_path, prev_path, k_stab, out_col="blended_est_slg"):
    cur = pd.read_csv(cur_path)
    prev = pd.read_csv(prev_path)

    # One BIP-weighted league level. NOT conditioned on sample size -- see notes.
    both = pd.concat([cur, prev], ignore_index=True).dropna(subset=["est_slg", "bip"])
    both = both[both["bip"] > 0]
    league = float(np.average(both["est_slg"], weights=both["bip"]))

    prev_small = prev[["player_id", "est_slg", "bip"]].rename(
        columns={"est_slg": "prev_slg", "bip": "prev_bip"}
    )
    m = cur[["player_id", "est_slg", "bip"]].merge(
        prev_small, on="player_id", how="left"
    )

    cur[out_col] = blend3(
        cur_units=m["bip"],
        cur=m["est_slg"],
        prev_units=m["prev_bip"],
        prev=m["prev_slg"],
        league=league,
        k_stab=k_stab,
    ).values

    cur[out_col] = cur[out_col].fillna(league)
    cur.to_csv(cur_path, index=False)
    return cur


def blend_hitter_xslg(date):
    return blend_season_xslg(
        f"./data/daily_data/statcast_hitting_{date}.csv",
        "./data/statcast_hitting_2025.csv",
        STAB["hitter_xslg"],
    )


def blend_pitcher_xslg(date):
    return blend_season_xslg(
        f"./data/daily_data/statcast_pitching_{date}.csv",
        "./data/statcast_pitching_2025.csv",
        STAB["pitcher_xslg"],
    )


def save_lineups():
    print("Saving lineups...")
    today = pd.to_datetime("today").strftime("%Y%m%d")
    tomorrow = (pd.to_datetime("today") + pd.Timedelta(days=1)).strftime("%Y%m%d")
    lineups = get_lineups()
    if len(lineups) != 0:
        lineups[0].to_csv(f"./data/lineups/{today}_lineup.csv", index=False)
        if len(lineups) != 1:
            lineups[1].to_csv(f"./data/lineups/{tomorrow}_lineup.csv", index=False)


def rebuild_pitcher_outing_logs(
    season_start="2026-03-27",
    sleep_base=12,
    sleep_jitter=4,
    max_retries=3,
):
    import time, random
    from pybaseball.datasources.bref import BRefSession

    session = BRefSession()
    session.session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        }
    )
    pybaseball.cache.enable()

    log_path = "./data/pitcher_outing_logs.csv"

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        done_dates = set(existing_log["date"].astype(str).unique())
    else:
        existing_log = pd.DataFrame()
        done_dates = set()

    all_dates = pd.date_range(
        pd.to_datetime(season_start), pd.to_datetime("today") - pd.Timedelta(days=1)
    )
    pending = [
        d.strftime("%Y-%m-%d")
        for d in all_dates
        if d.strftime("%Y%m%d") not in done_dates
    ]

    print(f"Dates already in log : {len(done_dates)}")
    print(f"Dates to fetch       : {len(pending)}")
    print(
        f"Estimated time       : ~{len(pending) * (sleep_base + sleep_jitter / 2) / 60:.0f} minutes"
    )
    print()

    all_new_rows = []

    for i, date in enumerate(pending):
        retries = 0
        while retries <= max_retries:
            try:
                pitching = pybaseball.pitching_stats_range(date, date)

                if pitching.empty:
                    print(f"[{i+1}/{len(pending)}] {date} — no data (off day)")
                    break

                new_rows = pitching[
                    ["mlbID", "Name", "G", "GS", "IP", "BF", "Pit"]
                ].copy()
                new_rows["date"] = date.replace("-", "")
                new_rows["pitch_count"] = new_rows["Pit"]
                new_rows = new_rows[
                    ["mlbID", "Name", "date", "G", "GS", "IP", "BF", "pitch_count"]
                ]

                all_new_rows.append(new_rows)
                print(f"[{i+1}/{len(pending)}] {date} — {len(new_rows)} outings")
                break

            except IndexError:
                print(
                    f"[{i+1}/{len(pending)}] {date} — no table (off day or BRef not updated)"
                )
                break

            except Exception as e:
                retries += 1
                if "403" in str(e) or "Forbidden" in str(e):
                    wait = 60 * retries
                    print(
                        f"[{i+1}/{len(pending)}] {date} — 403, waiting {wait}s (retry {retries}/{max_retries})"
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"[{i+1}/{len(pending)}] {date} — {e} (retry {retries}/{max_retries})"
                    )
                    time.sleep(10)

                if retries > max_retries:
                    print(f"  Giving up on {date}")
                    break

        if all_new_rows and (i + 1) % 10 == 0:
            existing_log = _flush_outing_rows(all_new_rows, existing_log, log_path)
            all_new_rows = []
            print(f"  >> Checkpoint saved at {date}")

        time.sleep(sleep_base + random.uniform(0, sleep_jitter))

    if all_new_rows:
        _flush_outing_rows(all_new_rows, existing_log, log_path)

    print("\nRebuild complete.")


def _flush_outing_rows(new_rows_list, existing_log, log_path):
    combined = pd.concat(new_rows_list, ignore_index=True)

    if not existing_log.empty:
        new_dates = combined["date"].astype(str).unique()
        existing_clean = existing_log[~existing_log["date"].astype(str).isin(new_dates)]
        updated = pd.concat([existing_clean, combined], ignore_index=True)
    else:
        updated = combined

    updated = updated.sort_values(["mlbID", "date"]).reset_index(drop=True)

    # G and GS from pitching_stats_range are per-outing (1 or 0)
    # so cumsum per player gives correct running totals
    updated["total_G"] = updated.groupby("mlbID")["G"].cumsum()
    updated["total_GS"] = updated.groupby("mlbID")["GS"].cumsum()
    updated["total_Pit"] = updated.groupby("mlbID")["pitch_count"].cumsum()
    updated["total_Outs"] = updated.groupby("mlbID")["Outs"].cumsum()

    updated.to_csv(log_path, index=False)
    return updated


if __name__ == "__main__":
    save_lineups()
    save_relevant_data(2026)
    add_prop_results()
    rebuild_pitcher_outing_logs()
