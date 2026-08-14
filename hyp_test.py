import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

team_colors = {
    "ARI": "#A71930",
    "ATL": "#CE1141",
    "BAL": "#DF4601",
    "BOS": "#BD3039",
    "CHC": "#0E3386",
    "CWS": "#27251F",
    "CIN": "#C6011F",
    "CLE": "#E31937",
    "COL": "#33006F",
    "DET": "#0C2340",
    "HOU": "#EB6E1F",
    "KC": "#004687",
    "LAA": "#BA0021",
    "LAD": "#005A9C",
    "MIA": "#00A3E0",
    "MIL": "#12284B",
    "MIN": "#002B5C",
    "NYM": "#FF5910",
    "NYY": "#003087",
    "ATH": "#EFB21E",
    "PHI": "#E81828",
    "PIT": "#FDB827",
    "SDP": "#2F241D",
    "SFG": "#FD5A1E",
    "SEA": "#005C5C",
    "STL": "#C41E3A",
    "TB": "#092C5C",
    "TEX": "#003278",
    "TOR": "#134A8E",
    "WSN": "#AB0003",
    # bref-style aliases → same colors
    "CHW": "#27251F",
    "KCR": "#004687",
    "TBR": "#092C5C",
    "OAK": "#EFB21E",
    "WSH": "#AB0003",
    "SD": "#2F241D",
    "SF": "#FD5A1E",
}

# Hypothesis: If a batter/pitcher has a higher whiff% (percentage of swings that are misses) leads to later counts
# Which leads to more Ks and more BBs


def gather_rows(t: str, threshold: int, bubble_metric: str):
    today = pd.to_datetime("today").strftime("%Y%m%d")
    bref = pd.read_csv(
        f"./data/daily_data/bref_{'pitching' if t == 'p' else 'hitting'}_{today}.csv"
    )
    sc = pd.read_csv(
        f"./data/daily_data/statcast_{'pitcher' if t == 'p' else 'hitter'}_arsenal_{today}.csv"
    )
    exp = pd.read_csv(
        f"./data/daily_data/statcast_{'pitching' if t == 'p' else 'hitting'}_{today}.csv"
    )
    pdb = pd.read_csv("./data/player_database.csv")

    sc["wp"] = sc["whiff_percent"] * sc["pitches"]
    g = sc.groupby("player_id").agg(wp=("wp", "sum"), p=("pitches", "sum"))
    sc = (g["wp"] / g["p"]).rename("whiff_percent")

    df = pd.merge(bref, sc, how="left", left_on="mlbID", right_index=True)
    df = pd.merge(
        df,
        pdb[["team", "key_mlbam"]],
        how="left",
        left_on="mlbID",
        right_on="key_mlbam",
    )

    # est_woba lives in the expected-stats file, keyed by player_id
    exp = exp.rename(columns={"player_id": "woba_id"})
    df = pd.merge(df, exp, how="left", left_on="mlbID", right_on="woba_id")

    # bigger metric -> bigger bubble: hitters as-is, pitchers inverted
    df["bubble_metric"] = df[bubble_metric]
    df["k-bb"] = df["blended_Kp"] - df["blended_BBp"]
    df["loc_skill"] = df["StL"] / df["Str"]
    df["meatballer"] = df["loc_skill"] / df["blended_Kp"]
    df["meatballer+bb"] = (df["loc_skill"] / df["blended_Kp"]) + df["blended_BBp"]
    df["meatballer+bb^2"] = df["meatballer+bb"] ** 2
    df["BFpG"] = df["BF"] / df["G"]
    df["K%"] = df["SO"] / df["BF"]
    df.reset_index()

    print(df.head())
    if t == "p":
        return df[(df["BF"] >= threshold)]
    return df[(df["PA"] >= threshold)]


def plot(df: pd.DataFrame, statx: str, staty: str):
    d = df.dropna(subset=[staty, statx]).reset_index(drop=True)

    colors = d["team"].map(team_colors).fillna("gray")

    # scale bubble_metric to marker area; fill missing woba with the median size
    metric = d["bubble_metric"].fillna(d["bubble_metric"].median())
    lo, hi = 20, 1200
    gamma = 3
    rng = metric.max() - metric.min()
    norm = (metric - metric.min()) / rng if rng else metric * 0
    sizes = lo + (norm**gamma) * (hi - lo)

    fig, ax = plt.subplots(figsize=(12, 9))
    pts = ax.scatter(d[statx], d[staty], c=colors, s=sizes, alpha=0.7)
    ax.set_xlabel(statx)
    ax.set_ylabel(staty)
    xpad = (d[statx].max() - d[statx].min()) * 0.1
    ypad = (d[staty].max() - d[staty].min()) * 0.1
    ax.set_xlim(d[statx].min() - xpad, d[statx].max() + xpad)
    ax.set_ylim(d[staty].min() - ypad, d[staty].max() + ypad)

    m, b = np.polyfit(d[statx], d[staty], 1)
    x = np.array([d[statx].min(), d[statx].max()])
    ax.plot(x, m * x + b, color="red")
    ax.axvline(d[statx].mean(), color="gray", linestyle="--", linewidth=1)
    ax.axhline(d[staty].mean(), color="gray", linestyle="--", linewidth=1)

    r = d[statx].corr(d[staty])
    ax.set_title(f"{statx} vs {staty} (r^2 = {r**2:.3f})")

    cursor = mplcursors.cursor(pts, hover=True)

    @cursor.connect("add")
    def on_sel(sel):
        sel.annotation.set_text(d["Name"].iloc[sel.index])

    plt.show()


if __name__ == "__main__":
    df = gather_rows(t="p", threshold=400, bubble_metric="BF")
    # plot(df, "whiff_percent", "blended_Kp")
    plot(df, "BF", "SO")
    plot(df, "BFpG", "blended_Kp")
    plot(df, "BFpG", "K%")
    plot(df, "K%", "blended_Kp")
    # plot(df, "blended_Kp", "loc_skill")
    # plot(df, "k-bb", "xera")
    # plot(df, "xera", "meatballer+bb")
    # plot(df, "xera", "meatballer+bb^2")
    # plot(df, "whiff_percent", "meatballer")

    # Hypotheses
    # 1. Get away games (Game before travel or next series) are said to have more strikes thrown and more swings --> BIP
    # Is this true?
