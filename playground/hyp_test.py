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


def analyze_hooks():
    logs = pd.read_csv("./data/pitcher_outing_logs.csv")

    starts = logs[logs["GS"] == 1][["Outs", "pitch_count"]]
    print("All starts")
    print(f"Mean Outs: {starts["Outs"].mean()} SD Outs: {starts["Outs"].std()}")
    print(
        f"Mean PC: {starts["pitch_count"].mean()} SD PC: {starts["pitch_count"].std()}"
    )

    full_starts = logs[(logs["GS"] == 1) & (logs["IP"] > 3)][["Outs", "pitch_count"]]
    print("Starts longer than 3 innings")
    print(
        f"Mean Outs: {full_starts["Outs"].mean()} SD Outs: {full_starts["Outs"].std()}"
    )
    print(
        f"Mean PC: {full_starts["pitch_count"].mean()} SD PC: {full_starts["pitch_count"].std()}"
    )

    big_timers = logs.groupby("Name", as_index=False)["Outs"].mean()
    nl = big_timers.sort_values("Outs", ascending=False).head(50)["Name"].tolist()
    nl2 = big_timers.sort_values("Outs", ascending=False).head(25)["Name"].tolist()

    big_names = logs[(logs["GS"] == 1) & (logs["Name"].isin(nl))][
        ["Outs", "pitch_count"]
    ]
    big_names2 = logs[(logs["GS"] == 1) & (logs["Name"].isin(nl2))][
        ["Outs", "pitch_count"]
    ]
    print("Starts from Top 20 pitchers in average outing length (Outs)")
    print(f"Mean Outs: {big_names["Outs"].mean()} SD Outs: {big_names["Outs"].std()}")
    print(
        f"Mean PC: {big_names["pitch_count"].mean()} SD PC: {big_names["pitch_count"].std()}"
    )

    print()

    counts = starts["Outs"].value_counts(normalize=True).sort_index()
    cdf = counts.reindex(range(0, 28), fill_value=0).cumsum()
    print(cdf.round(3).tolist())

    outs = np.arange(28)
    plt.step(outs, cdf, label="All Starts", color="red", where="post")
    plt.xlabel("Outs")
    plt.title("Starts Length CDF")
    plt.ylabel("P(Outs <= x)")

    counts = big_names["Outs"].value_counts(normalize=True).sort_index()
    cdf = counts.reindex(range(0, 28), fill_value=0).cumsum()
    plt.step(outs, cdf, label="Top 50", color="pink", where="post")

    counts = big_names2["Outs"].value_counts(normalize=True).sort_index()
    cdf = counts.reindex(range(0, 28), fill_value=0).cumsum()
    plt.step(outs, cdf, label="Top 25", color="blue", where="post")

    plt.legend()
    plt.xticks(range(0, 28))
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
