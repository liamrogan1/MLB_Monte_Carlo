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


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Statcast returns newest-first; sort to true order so prev-pitch is correct.
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(
        drop=True
    )
    d = df["description"]

    # ---- targets ----------------------------------------------------------
    df["swing"] = d.isin(SWING_DESCS).astype(int)
    df["swing_result"] = np.select(
        [d.isin(WHIFF_DESCS), d.isin(FOUL_DESCS), d.isin(INPLAY_DESCS)],
        ["whiff", "foul", "in_play"],
        default="take",
    )
    df["total_bases"] = np.where(d.eq("hit_into_play"), df["events"].map(BASES), np.nan)

    # ---- count / situational features ------------------------------------
    df["count_state"] = (
        df["balls"].astype("Int64").astype(str)
        + "-"
        + df["strikes"].astype("Int64").astype(str)
    )
    df["two_strike"] = (df["strikes"] == 2).astype(int)
    df["three_ball"] = (df["balls"] == 3).astype(int)
    df["ahead_in_count"] = (df["strikes"] > df["balls"]).astype(int)

    # ---- prev-pitch features (within the same at-bat) --------------------
    grp = df.groupby(["game_pk", "at_bat_number"], sort=False)
    for col in ["pitch_type", "release_speed", "plate_x", "plate_z", "description"]:
        df[f"prev_{col}"] = grp[col].shift(1)
    df["velo_diff_prev"] = df["release_speed"] - df["prev_release_speed"]
    df["first_pitch"] = df["prev_pitch_type"].isna().astype(int)

    return df


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


if __name__ == "__main__":
    for year in range(2021, 2027):
        pull_year(year)

    print(duckdb.sql("""
        SELECT COUNT(*)
        FROM 'data/training_data/statcast_*.parquet'
        """))
    print(duckdb.sql("""
            SELECT player_name, COUNT(DISTINCT pitch_name) as diff_pitches
            FROM 'data/training_data/statcast_*.parquet'
            GROUP BY player_name
            ORDER BY diff_pitches DESC
            LIMIT 10
            """))
