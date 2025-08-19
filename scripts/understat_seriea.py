#!/usr/bin/env python3
"""
scripts/understat_seriea.py

Fetch per-player (and per-match history if available) data from Understat for a league + seasons,
flatten the nested history if present, and write CSVs to data/.

Usage:
    python scripts/understat_seriea.py
    python scripts/understat_seriea.py --league Serie_A --seasons 2022,2023,2024 --output data --delay 0.8
"""
import ssl
import certifi

import argparse
import asyncio
import aiohttp
import pandas as pd
import os
import time
from understat import Understat

def parse_args():
    p = argparse.ArgumentParser(description="Understat league player/history fetcher")
    p.add_argument("--league", type=str, default="Serie_A", help="Understat league slug (Serie_A, EPL, La_liga, etc.)")
    p.add_argument("--seasons", type=str, default="2024", help="Comma-separated seasons (e.g. 2022,2023,2024)")
    p.add_argument("--output", type=str, default="data", help="Output directory")
    p.add_argument("--delay", type=float, default=0.8, help="Delay (s) between season requests")
    p.add_argument("--retries", type=int, default=3, help="Number of retries per season on failure")
    return p.parse_args()

def flatten_players(players):
    """
    Convert the list of player dicts returned by understat into a flat DataFrame.
    If 'history' is present, each history item becomes a row merged with top-level player info.
    """
    rows = []
    for p in players:
        base = {k: v for k, v in p.items() if k not in ("history", "shots", "stats")}
        history = p.get("history", [])
        if history:
            for h in history:
                r = dict(base)
                # history items frequently contain date, xG, result, etc.
                r.update(h)
                rows.append(r)
        else:
            rows.append(base)
    # Use pandas.json_normalize to flatten nested dicts if any
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)

async def fetch_league_players(session, league, season):
    u = Understat(session)
    players = await u.get_league_players(league, season)
    return players

async def fetch_and_save(session, league, season, out_dir, retries, delay):
    print(f"[+] Fetching {league} season {season} ...")
    tries = 0
    while tries < retries:
        try:
            players = await fetch_league_players(session, league, season)
            if not players:
                print(f"[!] No data returned for {league} {season}")
            df = flatten_players(players)
            df["season"] = season
            os.makedirs(out_dir, exist_ok=True)
            outfn = os.path.join(out_dir, f"understat_{league}_{season}.csv")
            df.to_csv(outfn, index=False)
            print(f"[+] Saved: {outfn}  (rows: {len(df)})")
            return df
        except Exception as e:
            tries += 1
            wait = 2 * tries
            print(f"[!] Error fetching {league} {season} (try {tries}/{retries}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed to fetch {league} {season} after {retries} retries")

async def main_async(args):
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    out_dir = args.output
    # Use a UA hint so Understat won't block us unnecessarily
    headers = {"User-Agent": "SerieA-Ingest/1.0 (+https://github.com/)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        results = []
        for season in seasons:
            df = await fetch_and_save(session, args.league, season, out_dir, args.retries, args.delay)
            results.append(df)
            # polite delay between seasons
            time.sleep(args.delay)
        # combine and save combined file
        combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        if not combined.empty:
            combined_fn = os.path.join(out_dir, f"understat_{args.league}_all.csv")
            combined.to_csv(combined_fn, index=False)
            print(f"[+] Saved combined file: {combined_fn} (rows: {len(combined)})")
        else:
            print("[!] No results to combine.")

def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    except Exception as e:
        print(f"[!] Fatal error: {e}")

if __name__ == "__main__":
    main()
