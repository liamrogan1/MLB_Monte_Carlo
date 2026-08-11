import os
import unicodedata

import numpy as np
import pandas as pd
import pybaseball

SEASON = 2026
PDB_PATH = "./data/player_database.csv"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Columns we keep in the player database, in order.
PDB_COLUMNS = [
    "team",
    "season",
    "rotowire_name",
    "name_last",
    "name_first",
    "key_mlbam",
    "key_retro",
    "key_bbref",
    "key_fangraphs",
    "mlb_played_first",
    "mlb_played_last",
]


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------
def strip_accents(text: str) -> str:
    """Fold accents so 'Acuña' -> 'Acuna', 'Alcántara' -> 'Alcantara'."""
    if not isinstance(text, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def split_name(name: str, statcast: bool = False):
    """Return (first_token, last_key), both lowercased/accent-free/suffix-free.

    Handles rotowire abbreviations ('X. Edwards' -> ('x', 'edwards')),
    multi-word last names ('E. De Jesus' -> ('e', 'de jesus')), and suffixes
    ('Ronald Acuna Jr.' -> ('ronald', 'acuna')). When statcast=True the name is
    'Last, First' and the comma is used to split cleanly.
    """
    s = strip_accents(str(name)).lower().replace(".", " ")
    if statcast and "," in s:
        last, _, first = s.partition(",")
        first_parts = first.split()
        first_tok = first_parts[0] if first_parts else ""
        last_key = " ".join(p for p in last.split() if p not in SUFFIXES)
        return first_tok, last_key

    parts = [p for p in s.replace(",", " ").split() if p and p not in SUFFIXES]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], " ".join(parts[1:])


# ---------------------------------------------------------------------------
# Daily-file id index (resolves debutants / accents with no chadwick lookup)
# ---------------------------------------------------------------------------
def build_daily_id_index(date: str) -> dict:
    """Map last_key -> [(first_token, mlbam), ...] from today's bref+statcast files.

    bref uses 'Name'/'mlbID', statcast uses 'last_name, first_name'/'player_id';
    both ids are MLBAM, so a matched name gives us the id directly.
    """
    root = "./data/daily_data"
    frames = [
        (f"{root}/bref_hitting_{date}.csv", "Name", "mlbID", False),
        (f"{root}/bref_pitching_{date}.csv", "Name", "mlbID", False),
        (
            f"{root}/statcast_hitting_{date}.csv",
            "last_name, first_name",
            "player_id",
            True,
        ),
        (
            f"{root}/statcast_pitching_{date}.csv",
            "last_name, first_name",
            "player_id",
            True,
        ),
    ]
    index: dict = {}
    for path, name_col, id_col, is_sc in frames:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for name, pid in zip(df[name_col], df[id_col]):
            if pd.isna(name) or pd.isna(pid):
                continue
            first_tok, last_key = split_name(name, statcast=is_sc)
            if last_key:
                index.setdefault(last_key, []).append((first_tok, int(pid)))
    return index


def resolve_from_daily(name: str, index: dict):
    """Return (mlbam, status). mlbam is None unless a single candidate matches."""
    first_tok, last_key = split_name(name)
    cands = index.get(last_key)
    if not cands:
        return None, "no_last_match"

    # Group the candidate first-name tokens by id.
    by_id: dict = {}
    for cf, cid in cands:
        by_id.setdefault(cid, set()).add(cf)

    is_initial = len(first_tok) == 1
    matches = set()
    for cid, firsts in by_id.items():
        if is_initial:
            if any(cf.startswith(first_tok) for cf in firsts):
                matches.add(cid)
        else:
            if any(cf == first_tok for cf in firsts):
                matches.add(cid)
            # single-candidate leniency: same first initial is good enough
            elif len(by_id) == 1 and any(cf[:1] == first_tok[:1] for cf in firsts):
                matches.add(cid)

    if len(matches) == 1:
        return matches.pop(), "ok"
    if not matches:
        return None, "last_only_no_first"
    return None, f"ambiguous({len(matches)})"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------
def row_from_existing(existing_row: pd.Series, name: str, team: str) -> dict:
    """Copy id keys from an existing DB row (any team/season) onto a new row.

    This is the trade path: the player already has keys in the DB, so we reuse
    them and just stamp the new team/season -- no lookup at all.
    """
    row = {c: existing_row.get(c) for c in PDB_COLUMNS}
    row["team"] = team
    row["season"] = SEASON
    row["rotowire_name"] = name
    row["mlb_played_last"] = SEASON
    return row


def row_from_mlbam(mlbam: int, name: str, team: str) -> dict:
    """Build a full DB row from an MLBAM id, filling secondary keys via one
    deterministic reverse lookup (no user prompt). Degrades to id+name only if
    the register is unavailable (e.g. offline)."""
    first_tok, last_key = split_name(name)
    row = {c: -1 for c in PDB_COLUMNS}
    row.update(
        {
            "team": team,
            "season": SEASON,
            "rotowire_name": name,
            "name_first": first_tok,
            "name_last": last_key,
            "key_mlbam": int(mlbam),
            "mlb_played_first": SEASON,
            "mlb_played_last": SEASON,
        }
    )
    try:
        res = pybaseball.playerid_reverse_lookup([int(mlbam)], key_type="mlbam")
        if not res.empty:
            r = res.iloc[0]
            for col in [
                "name_last",
                "name_first",
                "key_retro",
                "key_bbref",
                "key_fangraphs",
                "mlb_played_first",
                "mlb_played_last",
            ]:
                if col in r and pd.notna(r[col]):
                    row[col] = r[col]
    except Exception:
        pass  # keep the minimal id+name row
    return row


def resolve_via_chadwick(name: str, daily_mlbam=None):
    """Interactive fallback for players missing from the DB and daily files.

    If a daily-file id is available it is used to auto-pick the right row with no
    prompt. Otherwise this keeps the original manual escape hatches:
      row -1  -> re-search by last name (accent cases)
      row -2  -> skip this player
    """
    last_tok = split_name(name)[1].split()[-1]
    lookup = pybaseball.playerid_lookup(last_tok, fuzzy=True)
    if lookup.empty:
        return None

    lookup["mlb_played_last"] = pd.to_numeric(
        lookup["mlb_played_last"], errors="coerce"
    )

    # Auto-disambiguate against the daily-file id when we have one.
    if daily_mlbam is not None and "key_mlbam" in lookup:
        hit = lookup[lookup["key_mlbam"] == daily_mlbam]
        if not hit.empty:
            return hit.iloc[0:1]

    lookup = lookup[lookup["mlb_played_last"].fillna(0) >= 2021].sort_values(
        "mlb_played_last", ascending=False, ignore_index=True
    )
    if lookup.empty:
        return None
    if len(lookup) == 1:
        return lookup.iloc[0:1]

    print(lookup)
    row_number = int(
        input("Multiple players found. Enter row number (-1 accent search, -2 skip): ")
    )
    if row_number == -1:
        return _last_name_prompt()
    if row_number == -2:
        return None
    return lookup.iloc[row_number : row_number + 1]


def _last_name_prompt():
    last_name = input("Enter the player's last name: ")
    lookup = pybaseball.playerid_lookup(last_name, fuzzy=True)
    if lookup.empty:
        return None
    if len(lookup) == 1:
        return lookup.iloc[0:1]
    print(lookup)
    row_number = int(input("Enter row number for the correct player (-1 to retry): "))
    if row_number == -1:
        return _last_name_prompt()
    return lookup.iloc[row_number : row_number + 1]


def _row_from_lookup(lookup_row: pd.DataFrame, name: str, team: str) -> dict:
    r = lookup_row.iloc[0]
    row = {c: r.get(c, -1) for c in PDB_COLUMNS}
    row["team"] = team
    row["season"] = SEASON
    row["rotowire_name"] = name
    row["mlb_played_last"] = SEASON
    return row


# ---------------------------------------------------------------------------
# Single resolution entry point
# ---------------------------------------------------------------------------
def resolve_player(name, team, pdb, seen_keys, index):
    """Return a new DB row dict, or None if the player should be skipped
    (already present, or user-skipped)."""
    if not isinstance(name, str) or not name.strip():
        return None

    # 1. Already present for this team/season (or added earlier this run).
    if (name, team) in seen_keys:
        return None
    if (
        (pdb["season"] == SEASON)
        & (pdb["rotowire_name"] == name)
        & (pdb["team"] == team)
    ).any():
        return None

    # 2. Trade path: same rotowire name already in the DB under another
    #    team/season -> copy its id keys, no lookup.
    prior = pdb[pdb["rotowire_name"] == name]
    if not prior.empty:
        prior = prior.sort_values("mlb_played_last", ascending=False)
        return row_from_existing(prior.iloc[0], name, team)

    # 3. Debutant path: resolve the MLBAM id straight from today's stat files.
    mlbam, status = resolve_from_daily(name, index)
    if mlbam is not None:
        return row_from_mlbam(mlbam, name, team)

    # 4. Fallback: chadwick lookup (auto-picks with daily id if the name was
    #    ambiguous in the files; otherwise prompts).
    print(f"\n{name}  [{status}] -> chadwick lookup")
    lookup = resolve_via_chadwick(name, daily_mlbam=None)
    if lookup is None:
        return None
    return _row_from_lookup(lookup, name, team)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def update_player_database(date):
    pdb = pd.read_csv(PDB_PATH)
    lineup_df = pd.read_csv(f"./data/lineups/{date}_lineup.csv")
    bullpen_df = pd.read_csv(f"./data/daily_data/bullpen_activity_{date}.csv")
    index = build_daily_id_index(date)

    # (name, team) work list from hitters, opposing pitchers, and the bullpen.
    triples = []
    triples += list(zip(lineup_df["Player"], lineup_df["Team"]))
    triples += list(zip(lineup_df["Opposing Pitcher"], lineup_df["Opponent"]))
    triples += list(zip(bullpen_df["player"], bullpen_df["team"]))

    seen_keys = set()
    new_rows = []
    for name, team in dict.fromkeys(triples):  # de-dup, keep order
        row = resolve_player(name, team, pdb, seen_keys, index)
        if row is None:
            continue
        new_rows.append(row)
        seen_keys.add((name, team))
        # Make this player visible to later trade/skip checks in the same run.
        pdb = pd.concat([pdb, pd.DataFrame([row])], ignore_index=True)
        print(f"  + {row['rotowire_name']:22} {team}  mlbam={row['key_mlbam']}")

    if new_rows:
        pdb.to_csv(PDB_PATH, index=False)
        print(f"\nAdded {len(new_rows)} player(s); wrote {PDB_PATH}")
    else:
        print("\nNo new players to add.")


if __name__ == "__main__":
    today = pd.to_datetime("today").strftime("%Y%m%d")
    update_player_database(today)
