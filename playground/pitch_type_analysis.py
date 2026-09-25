import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
import os
import time


def _find_data_dir():
    start = Path(__file__).resolve()
    for parent in start.parents:  # script's dir, then up the tree
        candidate = parent / "data" / "training_data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate data/training_data above "
        f"{start} — is the data folder in a sibling or parent directory?"
    )


DATA_DIR = _find_data_dir()


@lru_cache
def _read_parquet(year):
    target_file = os.path.join(DATA_DIR, f"statcast_{year}.parquet")

    return pd.read_parquet(target_file)


def gather_pitch_types(years=[2026]) -> pd.DataFrame:
    results = []
    for year in years:
        stats = _read_parquet(year)
        stats = stats.copy()

        pitches = (
            stats.groupby(by=["pitcher", "pitch_type"])
            .agg(
                player_name=("player_name", "first"),
                season=("game_year", "first"),
                eff_speed=("effective_speed", "mean"),
                release_speed=("release_speed", "mean"),
                extension=("release_extension", "mean"),
                arm_angle=("arm_angle", "mean"),
                spin_rate=("release_spin_rate", "mean"),
                v_break=("api_break_z_with_gravity", "mean"),
                h_break=("api_break_x_arm", "mean"),
                accel_x=("ax", "mean"),
                accel_y=("ay", "mean"),
                accel_z=("az", "mean"),
                velo_x=("vx0", "mean"),
                velo_y=("vy0", "mean"),
                velo_z=("vz0", "mean"),
                n_times_thrown=("pitch_type", "size"),
            )
            .reset_index()
        )

        two_strikes = stats[stats["strikes"] == 2]
        ts = (
            two_strikes.groupby(by=["pitcher", "pitch_type"])
            .agg(total=("pitch_type", "size"))
            .reset_index()
        )
        ts["ts_thrown_pct"] = ts["total"] / ts.groupby("pitcher")["total"].transform(
            "sum"
        )
        ts = ts.drop(columns="total")

        in_play = stats[stats["bb_type"].notna()]
        bip = (
            in_play.groupby(by=["pitcher", "pitch_type"])
            .agg(
                bip=("bb_type", "size"),
                gb=("bb_type", lambda x: (x == "ground_ball").sum()),
                fb=("bb_type", lambda x: (x == "fly_ball").sum()),
                pop=("bb_type", lambda x: (x == "pop_up").sum()),
                ld=("bb_type", lambda x: (x == "line_drive").sum()),
                xwoba=("estimated_woba_using_speedangle", "mean"),
                xba=("estimated_ba_using_speedangle", "mean"),
                xslg=("estimated_slg_using_speedangle", "mean"),
            )
            .reset_index()
        )
        bip["gb%"] = bip["gb"] / bip["bip"]
        bip["fb%"] = bip["fb"] / bip["bip"]
        bip["pop%"] = bip["pop"] / bip["bip"]
        bip["ld%"] = bip["ld"] / bip["bip"]
        bip["air%"] = 1 - bip["gb%"]
        bip["gb/fb"] = np.clip(bip["gb"] / bip["fb"], 0, 1)

        pitches = pitches.merge(ts, how="left", on=["pitcher", "pitch_type"])
        pitches = pitches.merge(bip, how="left", on=["pitcher", "pitch_type"])
        pitches = pitches[
            (pitches["pitch_type"] != "EP") & (pitches["n_times_thrown"] >= 5)
        ]

        results.append(pitches)
    return pd.concat(results, ignore_index=True)


def knearest(
    df: pd.DataFrame, feature_cols, num_neighbors=5, exclude_same_pitcher=True
):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    keep_cols = (
        ["pitcher", "season", "pitch_type", "n_times_thrown", "player_name"]
        + feature_cols
        + ["xwoba"]
    )

    # reset_index so positional row numbers line up with the numpy idxs later
    data = df[keep_cols].dropna(axis=0).reset_index(drop=True)

    ids = data[["pitcher", "season", "pitch_type", "n_times_thrown", "player_name"]]
    X = data[feature_cols]
    y = data["xwoba"]  # Series -> y.iloc[i] is a bare float

    # Scale the WHOLE pool once. The query rows are just a subset of this same
    # matrix, so one scaler covers both the reference set and the query set.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # numpy array = the reference set

    # Query = most recent season only. Reference pool = every season.
    target_season = data["season"].max()
    query_mask = (data["season"] == target_season).to_numpy()
    query_pos = np.flatnonzero(query_mask)  # reference positions of the query rows
    print(
        f"Querying {query_mask.sum()} pitches from {target_season} "
        f"against {len(data)} pitches across {data['season'].nunique()} seasons"
    )

    ref_pitcher = ids["pitcher"].to_numpy()

    # We drop every neighbor that shares the query row's pitcher (his self,
    # his other seasons, his other pitch types). Worst case that's one
    # pitcher's entire row count, so over-fetch by that much to still be left
    # with num_neighbors survivors.
    buffer = int(ids.groupby("pitcher").size().max()) if exclude_same_pitcher else 1
    m = min(len(data), num_neighbors + buffer + 1)

    model = NearestNeighbors(n_neighbors=m).fit(X_scaled)
    dists, idxs = model.kneighbors(X_scaled[query_mask], n_neighbors=m)
    # NOTE: idxs indexes into the REFERENCE (all seasons); the loop below
    # iterates over QUERY rows. Two different index spaces — keep them straight.

    columns = ["pitcher_name", "season", "pitch_type", "n_times_thrown", "pitch_xwoba"]
    for n in range(1, num_neighbors + 1):
        columns += [
            f"n{n}_pitcher_name",
            f"n{n}_season",
            f"n{n}_pitch_type",
            f"n{n}_pitch_xwoba",
            f"n{n}_distance",
        ]
    results = {col: [] for col in columns}

    for q in range(len(idxs)):
        src = query_pos[q]  # reference position of this query pitch
        src_pitcher = ref_pitcher[src]

        # source fields come from the query row
        results["pitcher_name"].append(ids.iloc[src]["player_name"])
        results["season"].append(ids.iloc[src]["season"])
        results["pitch_type"].append(ids.iloc[src]["pitch_type"])
        results["n_times_thrown"].append(ids.iloc[src]["n_times_thrown"])
        results["pitch_xwoba"].append(y.iloc[src])

        kept = 0
        for j in range(m):
            nb = idxs[q, j]
            if exclude_same_pitcher and ref_pitcher[nb] == src_pitcher:
                continue  # skips self, his other seasons, his other pitches
            k = kept + 1
            results[f"n{k}_pitcher_name"].append(ids.iloc[nb]["player_name"])
            results[f"n{k}_season"].append(ids.iloc[nb]["season"])
            results[f"n{k}_pitch_type"].append(ids.iloc[nb]["pitch_type"])
            results[f"n{k}_pitch_xwoba"].append(y.iloc[nb])
            results[f"n{k}_distance"].append(dists[q, j])
            kept += 1
            if kept == num_neighbors:
                break

        # pad if the reference pool couldn't supply enough neighbors, so every
        # column stays the same length and pd.DataFrame won't choke
        while kept < num_neighbors:
            k = kept + 1
            results[f"n{k}_pitcher_name"].append(None)
            results[f"n{k}_season"].append(None)
            results[f"n{k}_pitch_type"].append(None)
            results[f"n{k}_pitch_xwoba"].append(np.nan)
            results[f"n{k}_distance"].append(np.nan)
            kept += 1

    nndf = pd.DataFrame(results)

    # Inverse-distance weighted xwoba prediction (closer neighbor -> more weight)
    dist_cols = [f"n{i}_distance" for i in range(1, num_neighbors + 1)]
    xwoba_cols = [f"n{i}_pitch_xwoba" for i in range(1, num_neighbors + 1)]

    weights = 1 / (nndf[dist_cols] + 1e-9)  # epsilon guards distance==0
    weights = weights.div(weights.sum(axis=1), axis=0)  # normalize each row to 1
    # .values drops labels so the two frames multiply by position, not by name;
    # nansum so any padded NaN slots don't poison the row
    nndf["pred_xwoba"] = np.nansum(weights.values * nndf[xwoba_cols].values, axis=1)
    nndf["diff_xwoba"] = nndf["pred_xwoba"] - nndf["pitch_xwoba"]
    nndf["confidence_area"] = np.max(nndf[dist_cols], axis=1) ** 2 * np.pi

    rmse = np.sqrt(((nndf["pred_xwoba"] - nndf["pitch_xwoba"]) ** 2).mean())
    print(f"RMSE: {rmse:.3f}")

    return nndf


if __name__ == "__main__":
    df = gather_pitch_types(years=[2021, 2022, 2023, 2024, 2025, 2026])
    feature_cols = ["eff_speed", "spin_rate", "arm_angle", "v_break", "h_break"]
    thresholds = df[
        (df["n_times_thrown"] > df["n_times_thrown"].quantile(0.25))
        & (df["bip"] > df["bip"].quantile(0.25))
    ]
    nndf = knearest(thresholds, feature_cols, num_neighbors=5)
    nndf.to_csv("./test_knearest.csv")

    # Analyze pitchers xwoba differences
    denom = nndf.groupby("pitcher_name")["n_times_thrown"].transform("sum")
    nndf["usage_share"] = nndf["n_times_thrown"] / denom
    nndf["weighted_diff"] = nndf["diff_xwoba"] * nndf["usage_share"]
    nndf["weighted_xwoba"] = nndf["pitch_xwoba"] * nndf["usage_share"]
    nndf["weighted_pred_xwoba"] = nndf["pred_xwoba"] * nndf["usage_share"]
    print(nndf.head())

    by_pitcher = nndf.groupby(by="pitcher_name").agg(
        season=("season", "first"),
        n_pitch_types=("pitch_type", "size"),
        total_pitches=("n_times_thrown", "sum"),
        xwoba=("weighted_xwoba", "sum"),
        pred_xwoba=("weighted_pred_xwoba", "sum"),
        total_xwoba_diff=("weighted_diff", "sum"),
    )
    by_pitcher = by_pitcher.sort_values(by="total_xwoba_diff")

    print(by_pitcher.head(30))
