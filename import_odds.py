# Uses https://the-odds-api.com
from collections import defaultdict
from datetime import datetime
import os
import numpy as np
import pandas as pd
import requests
import json
from zoneinfo import ZoneInfo
import time
from dotenv import load_dotenv


def get_event_ids(api_key, sport_key):
    URL = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events?apiKey={api_key}"

    try:
        response = requests.get(URL)
        if response.status_code == 200:
            sports_data = response.json()
            return [event["id"] for event in sports_data]
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            print(f"Error message: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the most recent fetched_at rows for each unique line.

    Use this in downstream files (pred_tb.py, import_data.py) so they only
    operate on the latest prices instead of every historical snapshot:

        odds_df = latest_snapshot(pd.read_csv(path))
    """
    if "fetched_at" not in df.columns:
        return df
    key_cols = ["date", "home_team", "away_team", "type", "player", "side", "point"]
    df = df.sort_values("fetched_at")
    return df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def save_props(date: str, new_rows_df: pd.DataFrame):
    """Append new snapshot rows to the existing props file (never overwrite history)."""
    path = f"./data/odds/{date}_props.csv"
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_rows_df], ignore_index=True)
    else:
        combined = new_rows_df
    combined.to_csv(path, index=False)
    print(f"Saved {len(new_rows_df)} new rows -> {path} ({len(combined)} total rows)")


def get_prices(event_ids, sport_key, api_key):
    # Buffer rows per date so a single run can span multiple slates safely
    props_by_date = defaultdict(list)

    # One timestamp per run so all rows from this pull share the same snapshot id
    now_local = datetime.now().astimezone()
    today_str = now_local.strftime("%Y%m%d")
    fetched_at = now_local.strftime("%Y-%m-%d %H:%M:%S")

    for event_id in event_ids:
        URL = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds?apiKey={api_key}&regions=us&markets=pitcher_strikeouts,pitcher_walks"
        time.sleep(5)
        try:
            response = requests.get(URL)
            if response.status_code != 200:
                print(f"Failed to retrieve data: {response.status_code}")
                print(f"Error message: {response.text}")
                continue

            odds_data = response.json()

            home_team = odds_data["home_team"]
            away_team = odds_data["away_team"]
            commence_time = odds_data["commence_time"]  # e.g. "2026-07-19T23:21:00Z"

            # Parse commence time as UTC, convert to local
            date_obj_utc = datetime.strptime(
                commence_time, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=ZoneInfo("UTC"))
            date_obj = date_obj_utc.astimezone(now_local.tzinfo)

            # Only append prices for games that have NOT started yet.
            # Rows already saved for started games remain untouched in the CSV.
            if date_obj <= now_local:
                print(f"Skipping started event: {away_team} @ {home_team} ({date_obj})")
                continue

            date = date_obj.strftime("%Y%m%d")
            if date != today_str:
                print(f"Skipping non-today event: {away_team} @ {home_team} ({date})")
                break
            print(f"{date} | {away_team} @ {home_team}")

            # market key -> label written into the csv "type" column
            market_labels = {
                "pitcher_strikeouts": "pitcher strikeouts",
                "pitcher_walks": "pitcher walks",
            }
            # {market_key: {player: {line: {"o": [...], "u": [...]}}}}
            prop_prices = {
                key: defaultdict(lambda: defaultdict(lambda: {"o": [], "u": []}))
                for key in market_labels
            }
            for book in odds_data["bookmakers"]:
                for market in book["markets"]:
                    key = market["key"]
                    if key in prop_prices:
                        for outcome in market["outcomes"]:
                            pitcher_name = outcome["description"]
                            side = outcome["name"]
                            line = float(outcome["point"])
                            price = outcome["price"]
                            if side == "Over":
                                prop_prices[key][pitcher_name][line]["o"].append(price)
                            elif side == "Under":
                                prop_prices[key][pitcher_name][line]["u"].append(price)

            for market_key, players in prop_prices.items():
                for player, lines in players.items():
                    for line, sides in lines.items():
                        o_prices = sides["o"]
                        u_prices = sides["u"]

                        o_avg = (
                            np.median([1 / p for p in o_prices]) if o_prices else None
                        )
                        u_avg = (
                            np.median([1 / p for p in u_prices]) if u_prices else None
                        )

                        if o_avg is not None and u_avg is not None:
                            fair_o, fair_u, z = shin_devig(o_avg, u_avg)

                        base = {
                            "date": date,
                            "home_team": home_team,
                            "away_team": away_team,
                            "type": market_labels[market_key],
                            "player": player,
                            "point": line,
                            "fetched_at": fetched_at,
                            "commence_time": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        if o_avg is not None:
                            props_by_date[date].append(
                                {
                                    **base,
                                    "side": "over",
                                    "price": round(1 / o_avg, 3),
                                    "fair_prob": round(fair_o, 4),
                                    "fair_price": round(1 / fair_o, 3),
                                    "z": z,
                                }
                            )
                        if u_avg is not None:
                            props_by_date[date].append(
                                {
                                    **base,
                                    "side": "under",
                                    "price": round(1 / u_avg, 3),
                                    "fair_prob": round(fair_u, 4),
                                    "fair_price": round(1 / fair_u, 3),
                                    "z": z,
                                }
                            )

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            continue

    # Save each date's rows, appending to any existing file
    col_order = [
        "date",
        "home_team",
        "away_team",
        "type",
        "player",
        "side",
        "point",
        "price",
        "fair_prob",
        "fair_price",
        "z",
        "fetched_at",
        "commence_time",
    ]
    for date, rows in props_by_date.items():
        df = pd.DataFrame(rows)[col_order]
        save_props(date, df)


def shin_devig(prob1, prob2):
    """
    Shin (1992/93) devig for a two-way market.
    prob1, prob2: implied probabilities, i.e. 1/decimal_odds per side.
    Returns (fair_prob1, fair_prob2, z), z = estimated insider proportion.
    """
    booksum = prob1 + prob2  # Pi > 1 when the book has margin
    if booksum <= 1:  # no vig (or an arb) -> nothing to strip
        return prob1, prob2, 0.0

    # Closed-form z for the 2-outcome case, derived from D1 + D2 = 2
    num = 2 * ((prob1**2 + prob2**2) - booksum) / booksum
    den = (prob2 - prob1) ** 2 - 1
    z = 1 - num / den

    def fair(pi):
        D = np.sqrt(z**2 + 4 * (1 - z) * pi**2 / booksum)
        return (D - z) / (2 * (1 - z))

    return fair(prob1), fair(prob2), z


if __name__ == "__main__":
    sport_key = "baseball_mlb"
    load_dotenv()
    api_key = os.environ.get("ODDS_API_KEY", "")

    event_ids = get_event_ids(api_key, sport_key)
    if event_ids:
        get_prices(event_ids, sport_key, api_key)
