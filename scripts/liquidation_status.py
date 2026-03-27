#!/usr/bin/env python3
"""
Liquidation Status Report
Checks DB count and live API for recent liquidations across BTC, ETH, SOL.
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

import psycopg2

DB_CONFIG = dict(host="localhost", port=5432, dbname="trader", user="trader", password="changeme")
HL_API = "https://api.hyperliquid.xyz/info"
ASSETS = ["BTC", "ETH", "SOL"]
LIQUIDATION_ADDRESS = "0x0000000000000000000000000000000000000000"


def fetch_recent_trades(coin: str, n: int = 10) -> list:
    payload = json.dumps({"type": "recentTrades", "coin": coin}).encode()
    req = urllib.request.Request(
        HL_API,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "crypto-trader-research/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)[:n]
    except Exception as e:
        print(f"  [WARN] Failed to fetch trades for {coin}: {e}")
        return []


def check_liquidations_in_trades(trades: list) -> list:
    liqs = []
    for t in trades:
        users = t.get("users", [])
        if isinstance(users, list) and LIQUIDATION_ADDRESS in users:
            liqs.append(t)
    return liqs


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*60}")
    print(f"  Liquidation Status Report — {now}")
    print(f"{'='*60}\n")

    # --- DB check ---
    print("[ Database ]")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM liquidations")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM liquidations WHERE time > now() - interval '24 hours'")
        last_24h = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM liquidations WHERE time > now() - interval '1 hour'")
        last_1h = cur.fetchone()[0]
        cur.execute("SELECT MAX(time) FROM liquidations")
        last_ts = cur.fetchone()[0]
        conn.close()
        print(f"  Total rows     : {total}")
        print(f"  Last 24h       : {last_24h}")
        print(f"  Last 1h        : {last_1h}")
        print(f"  Latest entry   : {last_ts or 'N/A'}")
    except Exception as e:
        print(f"  [ERROR] DB connection failed: {e}")

    # --- API check ---
    print("\n[ Live API — recentTrades (last 10 per asset) ]")
    api_summary = {}
    for coin in ASSETS:
        trades = fetch_recent_trades(coin, n=10)
        liqs = check_liquidations_in_trades(trades)
        api_summary[coin] = len(liqs)
        status = "✓ liquidations found" if liqs else "no liquidations"
        print(f"  {coin:4s} — {len(trades):2d} trades fetched, {len(liqs)} liquidations — {status}")
        if liqs:
            sample = liqs[0]
            print(f"          Sample: side={sample.get('side')}, px={sample.get('px')}, sz={sample.get('sz')}")

    # --- Summary ---
    total_api_liqs = sum(api_summary.values())
    print(f"\n[ Summary ]")
    print(f"  DB total rows  : {total if 'total' in dir() else 'N/A'}")
    print(f"  API liqs found : {total_api_liqs} across {ASSETS}")
    if total_api_liqs == 0:
        print("  → Market is calm, no liquidations in most recent trades. This is expected.")
    else:
        print("  → Liquidations detected in live API — poller should be capturing these.")
    print()


if __name__ == "__main__":
    main()
