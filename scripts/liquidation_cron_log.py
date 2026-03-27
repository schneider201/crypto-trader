#!/usr/bin/env python3
"""
Liquidation Cron Logger
Runs every 6 hours via cron. Appends a one-line status to liquidation_log.txt.
Format: 2026-03-27 19:00 | DB rows: 0 | API BTC liqs last hour: 0 | ETH: 0 | SOL: 0
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

DB_CONFIG = dict(host="localhost", port=5432, dbname="trader", user="trader", password="changeme")
HL_API = "https://api.hyperliquid.xyz/info"
ASSETS = ["BTC", "ETH", "SOL"]
LIQUIDATION_ADDRESS = "0x0000000000000000000000000000000000000000"
LOG_FILE = Path("/home/etienne/projects/crypto-trader/data/liquidation_log.txt")


def get_db_count() -> int:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM liquidations")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return -1


def get_api_liq_count(coin: str) -> int:
    payload = json.dumps({"type": "recentTrades", "coin": coin}).encode()
    req = urllib.request.Request(
        HL_API,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "crypto-trader-research/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            trades = json.load(resp)
            return sum(
                1 for t in trades
                if LIQUIDATION_ADDRESS in t.get("users", [])
            )
    except Exception:
        return -1


def main():
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M")

    db_count = get_db_count()
    api_counts = {coin: get_api_liq_count(coin) for coin in ASSETS}

    parts = [ts, f"DB rows: {db_count}"]
    for coin in ASSETS:
        parts.append(f"API {coin} liqs last hour: {api_counts[coin]}")

    line = " | ".join(parts)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    print(line)


if __name__ == "__main__":
    main()
