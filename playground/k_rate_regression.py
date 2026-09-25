"""
Derive model-ready targets + features from raw pybaseball.statcast() output.

Three stages (filter to the relevant subset when training each):
  1. swing        : all rows                     -> swing vs take
  2. swing_result : rows where swing == 1        -> whiff / foul / in_play
  3. total_bases  : rows where description == 'hit_into_play'  -> 0..4

Wire into the pull loop:
    import pybaseball
    from build_targets import build_targets
    pybaseball.cache.enable()
    for year in range(2020, 2027):
        raw = pybaseball.statcast(f"{year}-03-01", f"{year}-11-15")
        build_targets(raw).to_parquet(f"./data/statcast_{year}.parquet")
"""

import pandas as pd
import numpy as np
import os
import time
import pybaseball
import warnings
import duckdb
from functools import lru_cache


@lru_cache()
def _read_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)


warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

# description -> swing / contact buckets
SWING_DESCS = {
    "hit_into_play",
    "foul",
    "foul_tip",
    "foul_bunt",
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCS = {"foul", "foul_tip", "foul_bunt"}  # foul_tip = contact (judgment call)
INPLAY_DESCS = {"hit_into_play"}

# events -> total bases (only meaningful on balls in play; reached-on-error = 0)
BASES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
    "field_out": 0,
    "force_out": 0,
    "grounded_into_double_play": 0,
    "double_play": 0,
    "sac_fly": 0,
    "sac_bunt": 0,
    "fielders_choice": 0,
    "fielders_choice_out": 0,
    "field_error": 0,
    "sac_fly_double_play": 0,
    "triple_play": 0,
}

FASTBALL = {"FF", "FT", "SI", "FC", "FS", "FO", "FA"}

OFFSPEED = {"CH", "EP"}

BREAKING_BALL = {
    "SL",
    "ST",
    "CU",
    "KC",
    "SV",
    "SC",
    "KN",
    "CS",
    "GY",
}

# pitch_type -> coarse pitch group. Everything not in the three sets above
# (PO, IN, AB, AS, NP, UN, NaN, or any future code) falls through to "other".
PITCH_GROUP = {
    **{p: "fastball" for p in FASTBALL},
    **{p: "offspeed" for p in OFFSPEED},
    **{p: "breaking" for p in BREAKING_BALL},
}
GROUPS = ["fastball", "breaking", "offspeed", "other"]

# two-strike pitches that deliver strike three (swinging or looking).
# foul_tip is deliberately excluded — only a K if cleanly caught, and it's
# coded inconsistently; the miss is negligible.
STRIKE_THREE_DESCS = WHIFF_DESCS | {"called_strike"}

# window set for count-derived rates / entropy (small denominators)
RATE_TAGS = ("l5", "l10", "szn", "career")

PA_END_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "strikeout",
    "strikeout_double_play",
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "catcher_interf",
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "fielders_choice",
    "fielders_choice_out",
    "field_error",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "triple_play",
}


# TODO Will need to drop position players pitching, easiest would be through primary position data
def drop_position_player_pitching(df, velo_ceiling=84.0, hard_floor=72.0):
    """Remove outings thrown by position players.

    An outing (game_pk, pitcher) is dropped if the pitcher also batted in
    that same game AND his average release_speed is below velo_ceiling,
    OR if the outing velo is below hard_floor (catches a position player
    who pitched but didn't bat). The velo_ceiling guard keeps real NL
    pitchers who batted in 2021 and two-way players like Ohtani.
    """
    velo = df.groupby(["game_pk", "pitcher"])["release_speed"].transform("mean")

    batted_pairs = set(zip(df["game_pk"], df["batter"]))
    pitcher_batted = pd.Series(
        list(zip(df["game_pk"], df["pitcher"])), index=df.index
    ).isin(batted_pairs)

    is_ppp = (pitcher_batted & (velo < velo_ceiling)) | (velo < hard_floor)
    return df.loc[~is_ppp].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Statcast returns newest-first; sort to true order so prev-pitch is correct.
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(
        drop=True
    )
    d = df["description"]
    df["season"] = df["game_year"]

    # Coarse pitch group (fastball / breaking / offspeed / other). One label per
    # pitch; both the pitcher and batter aggregators reuse this same column.
    df["pitch_group"] = df["pitch_type"].map(PITCH_GROUP).fillna("other")

    # Targets
    df["swing"] = d.isin(SWING_DESCS).astype(int)
    df["swing_result"] = np.select(
        [d.isin(WHIFF_DESCS), d.isin(FOUL_DESCS), d.isin(INPLAY_DESCS)],
        ["whiff", "foul", "in_play"],
        default="take",
    )
    df["total_bases"] = np.where(d.eq("hit_into_play"), df["events"].map(BASES), np.nan)
    df["whiff"] = d.isin(WHIFF_DESCS).astype(int)

    # Count / Situational features
    df["count_state"] = (
        df["balls"].astype("Int64").astype(str)
        + "-"
        + df["strikes"].astype("Int64").astype(str)
    )
    df["two_strike"] = (df["strikes"] == 2).astype(int)
    df["three_ball"] = (df["balls"] == 3).astype(int)
    df["ahead_in_count"] = (df["strikes"] > df["balls"]).astype(int)

    # Previous pitch metrics (within same at-bat)
    grp = df.groupby(["game_pk", "at_bat_number"], sort=False)
    for col in ["pitch_type", "release_speed", "plate_x", "plate_z", "description"]:
        df[f"prev_{col}"] = grp[col].shift(1)
    df["velo_diff_prev"] = df["release_speed"] - df["prev_release_speed"]
    df["first_pitch"] = df["prev_pitch_type"].isna().astype(int)

    # First Pitch Metrics
    df["first_pitch_whiff"] = (
        df["first_pitch"] & df["description"].isin(WHIFF_DESCS)
    ).astype(int)
    df["first_pitch_swing"] = (
        df["first_pitch"] & df["description"].isin(SWING_DESCS)
    ).astype(int)
    df["first_pitch_strike"] = (df["first_pitch"] & (df["type"] == "S")).astype(int)
    df["first_pitch_called_strike"] = (
        df["first_pitch"] & (df["description"] == "called_strike")
    ).astype(int)

    # High-leverage counter stats
    df["runners_in_scoring_position"] = df["on_2b"].notna().astype(int) + df[
        "on_3b"
    ].notna().astype(int)
    df["is_tight_game"] = (
        (df["bat_win_exp"].ge(0.4) & df["bat_win_exp"].le(0.6))
        .fillna(False)
        .astype(int)
    )

    # Zone metrics
    HALF = 0.83
    in_x = df["plate_x"].abs() <= HALF
    in_z = df["plate_z"].between(df["sz_bot"], df["sz_top"])
    in_geom = in_x & in_z  # bool
    geom_ok = df[["plate_x", "plate_z", "sz_top", "sz_bot"]].notna().all(axis=1)

    in_zone = df["zone"] <= 9  # bool
    zone_ok = df["zone"].notna()

    iz = pd.Series(pd.NA, index=df.index, dtype="Int64")  # unknown until proven
    iz[geom_ok] = in_geom[geom_ok].astype("Int64")  # geometry first
    fb = ~geom_ok & zone_ok
    iz[fb] = in_zone[fb].astype("Int64")  # zone only where geom is gone
    df["in_zone"] = iz

    df["o_swing"] = ((df["swing"] == 1) & (df["in_zone"] == 0)).astype("Int64")
    df["chase_whiff"] = (
        (df["o_swing"] == 1) & df["description"].isin(WHIFF_DESCS)
    ).astype("Int64")
    df["z_swing"] = ((df["swing"] == 1) & (df["in_zone"] == 1)).astype("Int64")
    df["zone_whiff"] = (
        (df["z_swing"] == 1) & df["description"].isin(WHIFF_DESCS)
    ).astype("Int64")

    return df


def pitch_mix(df, entity):
    """Per-outing pitch-group counts + rates for either entity.

    entity='pitcher' -> the mix he threw; entity='batter' -> the mix he saw.
    Returns one row per (game_pk, entity) with n_<group> and <group>_rate cols.
    """
    keys = ["game_pk", entity]
    counts = (
        df.groupby(keys)["pitch_group"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=GROUPS, fill_value=0)  # guarantee all 4 cols exist
    )
    total = counts.sum(axis=1)
    out = counts.add_prefix("n_")  # n_fastball, n_breaking, ...
    for g in GROUPS:
        out[f"{g}_rate"] = counts[g] / total
    return out.reset_index()


def _roll_sum_specs(entity):
    """Shifted rolling/expanding SUMS (leak-safe). Mirrors add_rolling_features'
    windows but sums counts so rates are formed from pooled num/den."""
    return {
        "l3": ([entity], lambda s: s.shift(1).rolling(3, min_periods=1).sum()),
        "l5": ([entity], lambda s: s.shift(1).rolling(5, min_periods=1).sum()),
        "l10": ([entity], lambda s: s.shift(1).rolling(10, min_periods=1).sum()),
        "szn": ([entity, "season"], lambda s: s.shift(1).expanding().sum()),
        "career": ([entity], lambda s: s.shift(1).expanding().sum()),
    }


def add_rolled_rate(df, num, den, out, entity="pitcher", tags=RATE_TAGS):
    """Leak-safe rate = rolled(num) / rolled(den). Assumes df sorted by
    [entity, season, game_date]. Pools counts before dividing (talent_k-style)."""
    specs = _roll_sum_specs(entity)
    for tag in tags:
        keys, fn = specs[tag]
        rn = df.groupby(keys, sort=False)[num].transform(fn)
        rd = df.groupby(keys, sort=False)[den].transform(fn)
        df[f"{out}_{tag}"] = rn / rd
    return df


def add_arsenal_entropy(df, count_cols, entity="pitcher", tags=RATE_TAGS, base=2):
    """Shannon entropy of the pitch-TYPE mix, from pooled counts over the window
    (not an average of per-game entropies). Leak-safe via shift(1)."""
    specs = _roll_sum_specs(entity)
    for tag in tags:
        keys, fn = specs[tag]
        rolled = pd.DataFrame(
            {c: df.groupby(keys, sort=False)[c].transform(fn) for c in count_cols},
            index=df.index,
        )
        total = rolled.sum(axis=1)
        p = rolled.div(total, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * (np.log(p) / np.log(base)), 0.0)
        ent = -terms.sum(axis=1)
        ent[total <= 0] = np.nan  # no prior pitches yet -> unknown
        df[f"arsenal_entropy_{tag}"] = ent
    return df


def ptype_count_matrix(df, entity):
    """One row per (game_pk, entity), wide n_pt_<TYPE> counts (fill 0)."""
    m = (
        df.groupby(["game_pk", entity])["pitch_type"]
        .value_counts()
        .unstack(fill_value=0)
    )
    m.columns = [f"n_pt_{c}" for c in m.columns]
    return m.reset_index()


def two_strike_outing_counts(df):
    """Per-outing two-strike denominators/numerators, overall and per pitch group.
    den = two-strike pitches (put-away attempts); num = strike-three delivered."""
    ts = df[df["strikes"] == 2].copy()
    ts["_k"] = ts["description"].isin(STRIKE_THREE_DESCS).astype(int)

    overall = ts.groupby(["game_pk", "pitcher"]).agg(
        ts_pitches=("_k", "size"), ts_k=("_k", "sum")
    )

    grp = (
        ts.groupby(["game_pk", "pitcher", "pitch_group"])
        .agg(p=("_k", "size"), k=("_k", "sum"))
        .reset_index()
        .pivot_table(
            index=["game_pk", "pitcher"],
            columns="pitch_group",
            values=["p", "k"],
            fill_value=0,
        )
    )
    grp.columns = [f"ts_{'pitches' if a == 'p' else 'k'}_{b}" for a, b in grp.columns]
    for g in GROUPS:  # guarantee every group column exists
        grp[f"ts_pitches_{g}"] = grp.get(f"ts_pitches_{g}", 0)
        grp[f"ts_k_{g}"] = grp.get(f"ts_k_{g}", 0)

    return overall.join(grp, how="left").fillna(0).reset_index()


def pitcher_pitch_shapes(df, min_pitches=3):
    """Per-outing release consistency, velo spread, movement spread.

    release_consistency: usage-weighted mean of per-pitch_type release std
        (arm_angle std if available, else RMS of release_pos_x/z std).
    velo_spread: max-min of per-type mean velo over types clearing min_pitches.
    movement_spread: usage-weighted mean distance of each type's mean (pfx_x,
        pfx_z) from the usage-weighted centroid.
    """
    have_arm = "arm_angle" in df.columns and df["arm_angle"].notna().any()

    agg = {
        "n": ("pitch_type", "size"),
        "velo": ("release_speed", "mean"),
        "mx": ("pfx_x", "mean"),
        "mz": ("pfx_z", "mean"),
        "rx": ("release_pos_x", "std"),
        "rz": ("release_pos_z", "std"),
    }
    if have_arm:
        agg["aa"] = ("arm_angle", "std")

    pt = df.groupby(["game_pk", "pitcher", "pitch_type"]).agg(**agg).reset_index()
    pt["cons"] = (
        pt["aa"]
        if have_arm
        else np.sqrt(pt["rx"].fillna(0) ** 2 + pt["rz"].fillna(0) ** 2)
    )

    def _shape(g):
        w = g["n"].to_numpy(float)
        # release consistency: weight types that have a defined std (n >= 2)
        m = g["cons"].notna().to_numpy()
        rel = (
            np.average(g["cons"].to_numpy()[m], weights=w[m])
            if m.any() and w[m].sum() > 0
            else np.nan
        )

        u = g[g["n"] >= min_pitches]
        if len(u) >= 2:
            velo_spread = u["velo"].max() - u["velo"].min()
            wu = (u["n"] / u["n"].sum()).to_numpy()
            cx = (wu * u["mx"]).sum()
            cz = (wu * u["mz"]).sum()
            dist = np.sqrt((u["mx"] - cx) ** 2 + (u["mz"] - cz) ** 2).to_numpy()
            mov = float((wu * dist).sum())
        else:
            velo_spread = np.nan
            mov = np.nan
        return pd.Series(
            {
                "release_consistency": rel,
                "velo_spread": velo_spread,
                "movement_spread": mov,
            }
        )

    return pt.groupby(["game_pk", "pitcher"]).apply(_shape).reset_index()


def add_rolling_features(
    df, columns, entity="pitcher", tags=("l3", "l5", "l10", "szn", "career")
):
    """Leak-safe prior-form features for either entity. Assumes df is sorted by
    [entity, season, game_date]. l3/l5/l10/career carry across seasons; szn resets."""
    specs = {
        "l3": ([entity], lambda s: s.shift(1).rolling(3, min_periods=1).mean()),
        "l5": ([entity], lambda s: s.shift(1).rolling(5, min_periods=1).mean()),
        "l10": ([entity], lambda s: s.shift(1).rolling(10, min_periods=1).mean()),
        "szn": ([entity, "season"], lambda s: s.shift(1).expanding().mean()),
        "career": ([entity], lambda s: s.shift(1).expanding().mean()),
    }
    for col in columns:
        for tag in tags:
            keys, fn = specs[tag]
            df[f"avg_{col}_{tag}"] = df.groupby(keys, sort=False)[col].transform(fn)
    return df


def talent_k(df, k_todate, bf_todate, prior_rate, league_k, k_reg=60):
    prior = df[prior_rate].fillna(df[league_k])  # career; league if no career yet
    n = df[bf_todate].fillna(0)
    kk = df[k_todate].fillna(0)
    return (kk + k_reg * prior) / (n + k_reg)


# Ideas:
# Consistency of release out of hand
# Similarity to opponent's last seen pitcher
# Years in the league ~= amount of tape on the pitcher
# Amount of pitch types thrown
# Similarity/differences in pitch type speeds
# Similarity/differences in pitch type movement
# Find put away pitch (pitch most often thrown with two strikes)
#   Effectiveness of put away pitch
def build_outing_level(df: pd.DataFrame) -> pd.DataFrame:
    df["game_date"] = pd.to_datetime(df["game_date"])
    outings = (
        df.groupby(["game_pk", "pitcher"])
        .agg(
            game_date=("game_date", "first"),
            season=("season", "first"),
            player_name=("player_name", "first"),
            pitches=("pitcher", "size"),
            avg_extension=("release_extension", "mean"),
            avg_spin_rate=("release_spin_rate", "mean"),
            max_spin_rate=("release_spin_rate", "max"),
            avg_velocity=("release_speed", "mean"),
            max_velocity=("release_speed", "max"),
            avg_exit_velocity=("launch_speed", "mean"),
            strikeouts=(
                "events",
                lambda x: x.isin(["strikeout", "strikeout_double_play"]).sum(),
            ),
            walks=(
                "events",
                lambda x: x.isin(["walk", "intent_walk"]).sum(),
            ),  # add "intent_walk" if you want IBB in
            swings=("swing", "sum"),
            whiffs=("whiff", "sum"),
            iz_pitches=("in_zone", "sum"),  # 1s only; <NA> skipped
            oz_pitches=("in_zone", lambda x: (x == 0).sum()),
            z_swings=("z_swing", "sum"),
            chases=("o_swing", "sum"),
            zone_whiffs=("zone_whiff", "sum"),
            chase_whiffs=("chase_whiff", "sum"),
            first_pitch_strike_rate=("first_pitch_strike", "mean"),
            first_pitch_whiff_rate=("first_pitch_whiff", "mean"),
            risp_rate=("runners_in_scoring_position", "mean"),
            tight_game_rate=("is_tight_game", "mean"),
            tto=("n_thruorder_pitcher", "max"),
            batters_faced=("events", lambda e: e.isin(PA_END_EVENTS).sum()),
            gs=("inning", lambda x: int(min(x) == 1)),
        )
        .reset_index()
    )

    # pitch mix he threw
    outings = outings.merge(
        pitch_mix(df, "pitcher"), on=["game_pk", "pitcher"], how="left", validate="1:1"
    )
    # release/velo/movement shapes
    outings = outings.merge(
        pitcher_pitch_shapes(df), on=["game_pk", "pitcher"], how="left", validate="1:1"
    )
    # two-strike counts (put-away num/den, overall + per group)
    outings = outings.merge(
        two_strike_outing_counts(df),
        on=["game_pk", "pitcher"],
        how="left",
        validate="1:1",
    )
    # per-pitch_type counts for arsenal entropy
    ptm = ptype_count_matrix(df, "pitcher")
    pt_cols = [c for c in ptm.columns if c.startswith("n_pt_")]
    outings = outings.merge(ptm, on=["game_pk", "pitcher"], how="left", validate="1:1")

    # counts must be 0 (not NaN) so rolling sums work
    count_cols = (
        ["ts_pitches", "ts_k"]
        + [f"ts_pitches_{g}" for g in GROUPS]
        + [f"ts_k_{g}" for g in GROUPS]
        + pt_cols
    )
    outings[count_cols] = outings[count_cols].fillna(0)

    outings["k_percent"] = outings["strikeouts"] / outings["batters_faced"]
    outings["bb_percent"] = outings["walks"] / outings["batters_faced"]
    outings["swstr_percent"] = outings["whiffs"] / outings["pitches"]  # per-pitch
    outings["whiff_percent"] = (
        outings["whiffs"] / outings["swings"]
    )  # per-swing (true whiff%)
    outings["chase_percent"] = outings["chases"] / outings["oz_pitches"]
    outings["z_swing_percent"] = outings["z_swings"] / outings["iz_pitches"]
    outings["z_whiff_percent"] = outings["zone_whiffs"] / outings["z_swings"]
    outings["o_whiff_percent"] = outings["chase_whiffs"] / outings["chases"]
    outings["zone_rate"] = outings["iz_pitches"] / (
        outings["iz_pitches"] + outings["oz_pitches"]
    )
    outings = outings.replace([np.inf, -np.inf], np.nan)  # tiny relief outings can /0

    build_cols = [
        # rate skills
        "k_percent",
        "bb_percent",
        "whiff_percent",
        "swstr_percent",
        "chase_percent",
        "z_swing_percent",
        "z_whiff_percent",
        "o_whiff_percent",
        "zone_rate",
        "first_pitch_strike_rate",
        "first_pitch_whiff_rate",
        # pitch mix (roll it: this-game mix isn't known pregame)
        "fastball_rate",
        "breaking_rate",
        "offspeed_rate",
        # new scalar shapes (rolled -> avg_*_tag, leak-safe & auto-carried)
        "release_consistency",
        "velo_spread",
        "movement_spread",
        # stable descriptors — averaging denoises them
        "avg_velocity",
        "max_velocity",
        "avg_extension",
        "avg_spin_rate",
        "max_spin_rate",
        "avg_exit_velocity",
        # counts (avg-per-outing; your K/BB-count baseline)
        "strikeouts",
        "walks",
        "tto",
        "batters_faced",
    ]

    outings = outings.sort_values(by=["pitcher", "season", "game_date"])

    # gather recent form leading up to the outing
    outings = add_rolling_features(outings, build_cols)

    g = outings.groupby(["pitcher", "season"], sort=False)
    outings["k_szn_todate"] = g["strikeouts"].transform(
        lambda s: s.shift(1).expanding().sum()
    )
    outings["bf_szn_todate"] = g["batters_faced"].transform(
        lambda s: s.shift(1).expanding().sum()
    )

    outings["days_rest"] = g["game_date"].diff().dt.days

    # --- count-derived, leak-safe rates (pool num/den, then divide) ---
    outings = add_rolled_rate(outings, "ts_k", "ts_pitches", "put_away_pct")
    for grp in GROUPS:
        outings = add_rolled_rate(
            outings, f"ts_k_{grp}", f"ts_pitches_{grp}", f"put_away_pct_{grp}"
        )
        outings = add_rolled_rate(
            outings, f"ts_pitches_{grp}", "ts_pitches", f"ts_usage_{grp}"
        )
    outings = add_arsenal_entropy(outings, pt_cols)

    return outings


def build_batter_game_level(df: pd.DataFrame) -> pd.DataFrame:
    df["game_date"] = pd.to_datetime(df["game_date"])
    log = (
        df.groupby(["game_pk", "batter"])
        .agg(
            game_date=("game_date", "first"),
            season=("season", "first"),
            pitches_seen=("batter", "size"),
            pa=("events", lambda e: e.isin(PA_END_EVENTS).sum()),
            strikeouts=(
                "events",
                lambda x: x.isin(["strikeout", "strikeout_double_play"]).sum(),
            ),
            walks=("events", lambda x: x.isin(["walk", "intent_walk"]).sum()),
            swings=("swing", "sum"),
            whiffs=("whiff", "sum"),
            iz_pitches=("in_zone", "sum"),
            oz_pitches=("in_zone", lambda x: (x == 0).sum()),
            z_swings=("z_swing", "sum"),
            chases=("o_swing", "sum"),
            zone_whiffs=("zone_whiff", "sum"),
            chase_whiffs=("chase_whiff", "sum"),
            first_pitch_strike_rate=("first_pitch_strike", "mean"),
            first_pitch_whiff_rate=("first_pitch_whiff", "mean"),
            # --- batted-ball quality: the batter's real signal ---
            bip=("description", lambda s: (s == "hit_into_play").sum()),
            avg_exit_velocity=("launch_speed", "mean"),
            avg_launch_angle=("launch_angle", "mean"),
            hard_hits=("launch_speed", lambda s: (s >= 95).sum()),
            barrels=("launch_speed_angle", lambda s: (s == 6).sum()),
            xwoba_con=("estimated_woba_using_speedangle", "mean"),
            xba_con=("estimated_ba_using_speedangle", "mean"),
            xslg_con=("estimated_slg_using_speedangle", "mean"),
            avg_bat_speed=("bat_speed", "mean"),  # 2024+; mean skips NaN
            avg_swing_length=("swing_length", "mean"),
            total_bases=("events", lambda e: e.map(BASES).fillna(0).sum()),
        )
        .reset_index()
    )

    # pitch mix he saw (n_fastball, fastball_rate, ...)
    log = log.merge(
        pitch_mix(df, "batter"), on=["game_pk", "batter"], how="left", validate="1:1"
    )

    log["k_percent"] = log["strikeouts"] / log["pa"]
    log["bb_percent"] = log["walks"] / log["pa"]
    log["swstr_percent"] = log["whiffs"] / log["pitches_seen"]
    log["whiff_percent"] = log["whiffs"] / log["swings"]  # 1 - contact%
    log["chase_percent"] = log["chases"] / log["oz_pitches"]
    log["z_swing_percent"] = log["z_swings"] / log["iz_pitches"]
    log["z_whiff_percent"] = log["zone_whiffs"] / log["z_swings"]
    log["o_whiff_percent"] = log["chase_whiffs"] / log["chases"]
    log["zone_rate"] = log["iz_pitches"] / (log["iz_pitches"] + log["oz_pitches"])
    log["hard_hit_rate"] = log["hard_hits"] / log["bip"]
    log["barrel_rate"] = log["barrels"] / log["bip"]
    log = log.replace([np.inf, -np.inf], np.nan)

    build_cols = [
        # plate-discipline rates
        "k_percent",
        "bb_percent",
        "whiff_percent",
        "swstr_percent",
        "chase_percent",
        "z_swing_percent",
        "z_whiff_percent",
        "o_whiff_percent",
        "zone_rate",
        "first_pitch_strike_rate",
        # pitch mix seen (roll it: this-game mix isn't known pregame)
        "fastball_rate",
        "breaking_rate",
        "offspeed_rate",
        # batted-ball quality
        "avg_exit_velocity",
        "avg_launch_angle",
        "hard_hit_rate",
        "barrel_rate",
        "xwoba_con",
        "xba_con",
        "xslg_con",
        "avg_bat_speed",
        "avg_swing_length",
        # counts
        "strikeouts",
        "walks",
        "pa",
    ]

    log = log.sort_values(["batter", "season", "game_date"])
    log = add_rolling_features(log, build_cols, entity="batter")

    g = log.groupby(["batter", "season"], sort=False)
    log["k_szn_todate"] = g["strikeouts"].transform(
        lambda s: s.shift(1).expanding().sum()
    )
    log["pa_szn_todate"] = g["pa"].transform(lambda s: s.shift(1).expanding().sum())
    log["days_rest"] = g["game_date"].diff().dt.days
    return log


def build_at_bat_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group each AB with targets Strikeout
    """
    groupings = df.groupby(["game_pk", "at_bat_number"]).agg(
        game_date=("game_date", "first"),
        season=("game_year", "first"),
        pitcher_name=("player_name", "first"),
        pitcher_id=("pitcher", "first"),
        batter_id=("batter", "first"),
        events=("events", "first"),
        des=("des", "first"),
        # --- context / features (state entering the AB) ---
        stand=("stand", "first"),
        p_throws=("p_throws", "first"),
        tto=("n_thruorder_pitcher", "first"),  # K% drops each time through
        prior_pa_vs_pitcher=("n_priorpa_thisgame_player_at_bat", "first"),
        inning=("inning", "first"),
        top_bot=("inning_topbot", "first"),
        outs=("outs_when_up", "first"),
        on_1b=(
            "on_1b",
            lambda s: s.iloc[0] == s.iloc[0],
        ),  # notna at first pitch -> bool
        on_2b=("on_2b", lambda s: s.iloc[0] == s.iloc[0]),
        on_3b=("on_3b", lambda s: s.iloc[0] == s.iloc[0]),
        bat_score=("bat_score", "first"),
        fld_score=("fld_score", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        days_rest_pit=("pitcher_days_since_prev_game", "first"),
        age_bat=("age_bat", "first"),
        age_pit=("age_pit", "first"),
        pitches=("pitcher", "size"),  # keep for filtering, NOT a feature
    )
    groupings["strikeout"] = (
        groupings["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    )
    groupings["walk"] = groupings["events"].isin(["walk", "intent_walk"]).astype(int)
    groupings["in_play"] = (groupings["events"].isin(BASES)).astype(
        int
    )  # for the TB stage
    groupings["total_bases"] = groupings["events"].map(BASES)  # 0..4, NaN off-BIP
    groupings["is_pa"] = (
        groupings["events"].isin(PA_END_EVENTS).astype(int)
    )  # exclude truncated

    return groupings


def pull_year(year, retries=4):
    path = f"./data/training_data/statcast_{year}.parquet"
    if os.path.exists(path):
        print(f"{year}: already done, skipping")
        return
    start = "2020-07-23" if year == 2020 else f"{year}-03-01"
    end = f"{year}-11-15"
    for attempt in range(1, retries + 1):
        try:
            df = pybaseball.statcast(start, end)
            df.to_parquet(path)
            print(f"{year}: {len(df):,} rows -> {path}")
            return
        except Exception as e:
            wait = 15 * attempt
            print(
                f"{year}: attempt {attempt} failed ({type(e).__name__}); retrying in {wait}s"
            )
            time.sleep(wait)
    print(f"{year}: FAILED after {retries} attempts — leaving for a later rerun")


def oof_logloss(data, feats, y, groups, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import log_loss

    X = data[feats].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])  # scaler fit on train fold only
        m = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return log_loss(y, oof), oof


def forward_select(data, candidates, y, groups, base=(), n_splits=5):
    from sklearn.metrics import log_loss

    selected = list(base)
    remaining = [f for f in candidates if f not in selected]
    best0 = (
        oof_logloss(data, selected, y, groups, n_splits)[0]
        if selected
        else log_loss(y, np.full(len(y), y.mean()))
    )
    path = [(list(selected), best0)]  # (feature set, score) after each step
    print(f"start: {selected}  logloss {best0:.5f}")

    while remaining:
        ll, f = min(
            (oof_logloss(data, selected + [c], y, groups, n_splits)[0], c)
            for c in remaining
        )
        selected.append(f)  # always consume the winner
        remaining.remove(f)
        path.append((list(selected), ll))
        print(f"  + {f:<32} logloss {ll:.5f}")

    best_set, best_ll = min(path, key=lambda p: p[1])  # pick the low point
    print(f"best: {len(best_set)} feats  logloss {best_ll:.5f}")
    return best_set


def add_league_k(df, target="strikeout", season_col="season"):
    """Per-season league K% as the log5 normalizer. Near-constant (~.22),
    so self-inclusion is negligible; lag a season if you want zero leakage."""
    league = df.groupby(season_col)[target].transform("mean")
    df["league_k"] = league
    return df


def add_log5_k(
    df,
    pit_col="pit_avg_k_percent_career",
    bat_col="bat_avg_k_percent_career",
    league_col="league_k",
    out="log5_k",
):
    """Bill James log5: expected P(K) for this batter-pitcher pair, calibrated to league."""
    P = df[pit_col].clip(1e-6, 1 - 1e-6)
    B = df[bat_col].clip(1e-6, 1 - 1e-6)
    Lg = df[league_col].clip(1e-6, 1 - 1e-6)
    num = (P * B) / Lg
    df[out] = num / (num + ((1 - P) * (1 - B)) / (1 - Lg))
    return df


def train_ks_logistic(df):
    """
    Calculates a few additional fields like pitcher abd batter strikeout talent as wll as their log5 combination.
    Then from a list of candidates, features are forward selected using log loss. Finally these features are tested and total log loss,
    K% RMSE (for game), and K% RMSE weighted by batters faced are calculated.

    df: Has every pre-game feature available for both the batter and the pitcher at the given at bat
    """

    import numpy as np
    from sklearn.metrics import mean_squared_error

    df = df[df["is_pa"] == 1].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = add_league_k(df)
    df["pit_talent_k"] = talent_k(
        df,
        "pit_k_szn_todate",
        "pit_bf_szn_todate",
        "pit_avg_k_percent_career",
        "league_k",
    )
    df["bat_talent_k"] = talent_k(
        df,
        "bat_k_szn_todate",
        "bat_pa_szn_todate",
        "bat_avg_k_percent_career",
        "league_k",
    )
    df = add_log5_k(df, pit_col="pit_talent_k", bat_col="bat_talent_k")

    candidates = [
        "log5_k",
        "bat_avg_chase_percent_career",
        "bat_avg_whiff_percent_career",
        "bat_avg_first_pitch_strike_rate_career",
        "bat_avg_avg_swing_length_career",
        "bat_avg_chase_percent_szn",
        "bat_avg_whiff_percent_szn",
        "bat_avg_first_pitch_strike_rate_szn",
        "bat_avg_avg_swing_length_szn",
        "pit_avg_chase_percent_career",
        "pit_avg_whiff_percent_career",
        "pit_avg_chase_percent_szn",
        "pit_avg_whiff_percent_szn",
        "pit_avg_avg_velocity_career",
        "pit_avg_avg_velocity_l5",
        "pit_avg_first_pitch_strike_rate_career",
        "pit_avg_first_pitch_strike_rate_l5",
        "pit_avg_max_velocity_career",
        "pit_avg_max_velocity_l5",
        "tto",
        "pit_avg_release_consistency_szn",
        "pit_avg_velo_spread_szn",
        "pit_avg_movement_spread_szn",
        "pit_avg_movement_spread_l5",
        "pit_put_away_pct_career",
        "pit_put_away_pct_l5",
        "pit_put_away_pct_breaking_career",
        "pit_ts_usage_breaking_szn",
        "pit_arsenal_entropy_szn",
    ]
    keep = [
        "game_pk",
        "pitcher_id",
        "game_date",
        "pitcher_name",
        "strikeout",
        *candidates,
    ]
    data = df[keep].dropna().sort_values("game_date").reset_index(drop=True)
    y = data["strikeout"].to_numpy()
    groups = data["game_pk"].to_numpy()

    chosen = forward_select(
        data,
        candidates,
        y,
        groups,
        base=["log5_k", "pit_avg_whiff_percent_career", "pit_avg_movement_spread_szn"],
    )
    print("selected:", chosen)

    ll, oof = oof_logloss(data, chosen, y, groups)
    data["p_k"] = oof
    print(f"final OOF logloss {ll:.5f}")
    gm = (
        data.groupby(["game_pk", "pitcher_id"])
        .agg(
            pred_k=("p_k", "mean"),
            actual_k=("strikeout", "mean"),
            pitcher_name=("pitcher_name", "first"),
            game_date=("game_date", "first"),
            bf=("strikeout", "size"),
            actual_total=("strikeout", "sum"),
        )
        .reset_index()
    )
    gm["pred_total"] = gm["pred_k"] * gm["bf"]

    gm = gm.sort_values(["pitcher_id", "game_date"])
    g = gm.groupby("pitcher_id")
    prior_k = g["actual_total"].transform(lambda s: s.shift(1).expanding().sum())
    prior_bf = g["bf"].transform(lambda s: s.shift(1).expanding().sum())
    gm["prev_kp"] = prior_k / prior_bf

    # directional hit: pred and actual on the same side of his baseline
    pred_up = gm["pred_k"] > gm["prev_kp"]
    act_up = gm["actual_k"] > gm["prev_kp"]
    gm["direction_hit"] = (pred_up == act_up).astype(int)
    gm.loc[gm["prev_kp"].isna(), "direction_hit"] = np.nan  # first outing: no baseline

    print(f"game K% RMSE {np.sqrt(mean_squared_error(gm.actual_k, gm.pred_k)):.4f}")
    print(
        f"game weighted K% RMSE {np.sqrt(mean_squared_error(gm.actual_k, gm.pred_k, sample_weight=gm.bf))}"
    )
    print(
        f"Total pred Ks {gm.pred_total.sum():.0f} vs actual Ks {gm.actual_total.sum()}"
    )
    print(f"direction accuracy {gm['direction_hit'].mean():.3f}")
    return chosen, data, gm


# TODO Configure
def train_bbs_logistic(df):
    """
    Calculates a few additional fields like pitcher abd batter strikeout talent as wll as their log5 combination.
    Then from a list of candidates, features are forward selected using log loss. Finally these features are tested and total log loss,
    BB% RMSE (for game), and BB% RMSE weighted by batters faced are calculated.

    df: Has every pre-game feature available for both the batter and the pitcher at the given at bat
    """

    import numpy as np
    from sklearn.metrics import mean_squared_error

    df = df[df["is_pa"] == 1].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = add_league_k(df)
    df["pit_talent_k"] = talent_k(
        df,
        "pit_k_szn_todate",
        "pit_bf_szn_todate",
        "pit_avg_k_percent_career",
        "league_k",
    )
    df["bat_talent_k"] = talent_k(
        df,
        "bat_k_szn_todate",
        "bat_pa_szn_todate",
        "bat_avg_k_percent_career",
        "league_k",
    )
    df = add_log5_k(df, pit_col="pit_talent_k", bat_col="bat_talent_k")

    candidates = [
        "log5_k",
        "bat_avg_chase_percent_career",
        "bat_avg_whiff_percent_career",
        "bat_avg_first_pitch_strike_rate_career",
        "bat_avg_avg_swing_length_career",
        "bat_avg_chase_percent_szn",
        "bat_avg_whiff_percent_szn",
        "bat_avg_first_pitch_strike_rate_szn",
        "bat_avg_avg_swing_length_szn",
        "pit_avg_chase_percent_career",
        "pit_avg_whiff_percent_career",
        "pit_avg_chase_percent_szn",
        "pit_avg_whiff_percent_szn",
        "pit_avg_avg_velocity_career",
        "pit_avg_avg_velocity_l5",
        "pit_avg_first_pitch_strike_rate_career",
        "pit_avg_first_pitch_strike_rate_l5",
        "pit_avg_max_velocity_career",
        "pit_avg_max_velocity_l5",
        "tto",
        "pit_avg_release_consistency_szn",
        "pit_avg_velo_spread_szn",
        "pit_avg_movement_spread_szn",
        "pit_avg_movement_spread_l5",
        "pit_put_away_pct_career",
        "pit_put_away_pct_l5",
        "pit_put_away_pct_breaking_career",
        "pit_ts_usage_breaking_szn",
        "pit_arsenal_entropy_szn",
    ]
    keep = [
        "game_pk",
        "pitcher_id",
        "game_date",
        "pitcher_name",
        "strikeout",
        *candidates,
    ]
    data = df[keep].dropna().sort_values("game_date").reset_index(drop=True)
    y = data["strikeout"].to_numpy()
    groups = data["game_pk"].to_numpy()

    chosen = forward_select(
        data,
        candidates,
        y,
        groups,
        base=["log5_k", "pit_avg_whiff_percent_career", "pit_avg_movement_spread_szn"],
    )
    print("selected:", chosen)

    ll, oof = oof_logloss(data, chosen, y, groups)
    data["p_k"] = oof
    print(f"final OOF logloss {ll:.5f}")
    gm = (
        data.groupby(["game_pk", "pitcher_id"])
        .agg(
            pred_k=("p_k", "mean"),
            actual_k=("strikeout", "mean"),
            pitcher_name=("pitcher_name", "first"),
            game_date=("game_date", "first"),
            bf=("strikeout", "size"),
            actual_total=("strikeout", "sum"),
        )
        .reset_index()
    )
    gm["pred_total"] = gm["pred_k"] * gm["bf"]

    gm = gm.sort_values(["pitcher_id", "game_date"])
    g = gm.groupby("pitcher_id")
    prior_k = g["actual_total"].transform(lambda s: s.shift(1).expanding().sum())
    prior_bf = g["bf"].transform(lambda s: s.shift(1).expanding().sum())
    gm["prev_kp"] = prior_k / prior_bf

    # directional hit: pred and actual on the same side of his baseline
    pred_up = gm["pred_k"] > gm["prev_kp"]
    act_up = gm["actual_k"] > gm["prev_kp"]
    gm["direction_hit"] = (pred_up == act_up).astype(int)
    gm.loc[gm["prev_kp"].isna(), "direction_hit"] = np.nan  # first outing: no baseline

    print(f"game K% RMSE {np.sqrt(mean_squared_error(gm.actual_k, gm.pred_k)):.4f}")
    print(
        f"game weighted K% RMSE {np.sqrt(mean_squared_error(gm.actual_k, gm.pred_k, sample_weight=gm.bf))}"
    )
    print(
        f"Total pred Ks {gm.pred_total.sum():.0f} vs actual Ks {gm.actual_total.sum()}"
    )
    print(f"direction accuracy {gm['direction_hit'].mean():.3f}")
    return chosen, data, gm


def pregame_feats(log, entity_col, prefix):
    """Keep only leak-safe pregame features (rolling avg_*, count-derived rates,
    entropy) + season-to-date counts + keys, prefixed."""
    keep = ("avg_", "arsenal_entropy_", "put_away_pct_", "ts_usage_")
    feats = [c for c in log.columns if c.startswith(keep)]
    feats.append("bf_szn_todate" if entity_col == "pitcher" else "pa_szn_todate")
    feats.append("k_szn_todate")
    out = log[["game_pk", entity_col, *feats]].copy()
    return out.rename(columns={c: f"{prefix}{c}" for c in feats})


if __name__ == "__main__":
    for year in range(2021, 2027):
        pull_year(year)

    # duckdb.sql("""DESCRIBE 'data/training_data/statcast_2025.parquet'""").show(
    #     max_rows=120
    # )
    # duckdb.sql("""
    #     SELECT DISTINCT pitch_type FROM 'data/training_data/statcast_2025.parquet'
    #     """).show()
    dfs = []
    for year in range(2024, 2027):
        dfs.append(_read_parquet(f"../data/training_data/statcast_{year}.parquet"))
    raw = pd.concat(dfs, ignore_index=True)
    df = build_features(raw)
    # pitcher_outings = build_outing_level(df)
    # batter_outings = build_batter_game_level(df)
    # pitcher_outings.to_csv("./pitcher_outings.csv")
    # batter_outings.to_csv("./batter_outings.csv")
    pitcher_outings = pd.read_csv("./pitcher_outings.csv")
    batter_outings = pd.read_csv("./batter_outings.csv")

    abs = build_at_bat_level(df)

    bat_feat = pregame_feats(batter_outings, "batter", "bat_")
    pit_feat = pregame_feats(pitcher_outings, "pitcher", "pit_")

    m1 = abs.merge(
        bat_feat,
        how="left",
        left_on=["game_pk", "batter_id"],
        right_on=["game_pk", "batter"],
        validate="m:1",
    )
    m2 = m1.merge(
        pit_feat,
        how="left",
        left_on=["game_pk", "pitcher_id"],
        right_on=["game_pk", "pitcher"],
        validate="m:1",
    )

    assert len(m2) == len(abs), "merged changed row count -- duplicate key in log"
    # print("unmatched batter rows:", m2["bat_avg_k_percent_szn"].isna().mean())
    # print("unmatched pitcher rows:", m2["pit_avg_k_percent_szn"].isna().mean())

    chosen, data, games = train_ks_logistic(m2)
    games.to_csv("./playground/log_k_test_results.csv")

    # chosen, data, games = train_bbs_logistic(m2)
    # games.to_csv("./playground/log_bb_test_results.csv")
