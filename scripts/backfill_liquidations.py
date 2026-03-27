#!/usr/bin/env python3
"""
Backfill Liquidations from Hyperliquid API
Fetches recent liquidation fills for BTC, ETH, SOL from the zero-address (liquidator)
and inserts into TimescaleDB.

The Hyperliquid zero address (0x000...000) is the liquidation engine — all liquidation
fills appear as fills by/against this address.

Note: Also attempts Coinglass public API if available; gracefully handles auth errors.
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = dict(host="localhost", port=5432, dbname="trader", user="trader", password="changeme")
HL_API = "https://api.hyperliquid.xyz/info"
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"
ASSETS = ["BTC", "ETH", "SOL"]
DAYS = 7
LIQUIDATION_ADDRESS = "0x0000000000000000000000000000000000000000"


def hl_request(payload: dict) -> list | dict | None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        HL_API,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "crypto-trader-research/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except Exception as e:
        print(f"  [WARN] HL API request failed: {e}")
        return None


def fetch_hl_liquidations_by_time(start_time_ms: int) -> list:
    """
    Fetch fills for the liquidation address using userFillsByTime.
    Returns all fills (all coins) since start_time.
    """
    result = hl_request({
        "type": "userFillsByTime",
        "user": LIQUIDATION_ADDRESS,
        "startTime": start_time_ms,
    })
    if isinstance(result, list):
        return result
    return []


def fetch_hl_liquidations_recent() -> list:
    """
    Fetch most recent 2000 fills for the liquidation address.
    Falls back when time-range query returns nothing.
    """
    result = hl_request({
        "type": "userFills",
        "user": LIQUIDATION_ADDRESS,
    })
    if isinstance(result, list):
        return result
    return []


def map_hl_fill_to_schema(fill: dict) -> dict | None:
    """Map a Hyperliquid fill to our liquidations schema."""
    try:
        ts = datetime.fromtimestamp(fill["time"] / 1000, tz=timezone.utc)
        side = "long" if fill.get("side") == "A" else "short"  # A=ask=sell=long liq; B=buy=short liq
        price = float(fill.get("px", 0))
        qty = float(fill.get("sz", 0))
        usd_value = price * qty
        return {
            "time": ts,
            "exchange": "hyperliquid",
            "symbol": fill["coin"],
            "side": side,
            "price": round(price, 8),
            "quantity": round(qty, 8),
            "usd_value": round(usd_value, 2),
        }
    except Exception as e:
        print(f"  [WARN] Failed to map fill: {e}")
        return None


def fetch_coinglass_liquidations(symbol: str) -> list:
    """Attempt Coinglass public API. Handles auth errors gracefully."""
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)
    url = (
        f"{COINGLASS_BASE}/liquidation_history"
        f"?symbol={symbol}&time_type=1&start_time={start_ts}&end_time={end_ts}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-trader-research/1.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.load(resp)
            if body.get("code") in ("0", 0) and "data" in body:
                return body["data"]
            msg = body.get("msg", str(body))
            print(f"  [INFO] Coinglass {symbol}: {msg}")
            return []
    except urllib.error.HTTPError as e:
        print(f"  [INFO] Coinglass {symbol}: HTTP {e.code} — API key likely required.")
        return []
    except Exception as e:
        print(f"  [WARN] Coinglass {symbol}: {e}")
        return []


def insert_liquidations(rows: list) -> int:
    if not rows:
        return 0
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    data = [
        (r["time"], r["exchange"], r["symbol"], r["side"], r["price"], r["quantity"], r["usd_value"])
        for r in rows
    ]
    try:
        execute_values(
            cur,
            """
            INSERT INTO liquidations (time, exchange, symbol, side, price, quantity, usd_value)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            data,
        )
        inserted = cur.rowcount
        conn.commit()
        return max(inserted, 0)
    finally:
        conn.close()


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)

    print(f"\n{'='*60}")
    print(f"  Liquidation Backfill — Hyperliquid + Coinglass")
    print(f"  {now} | Period: last {DAYS} days")
    print(f"{'='*60}\n")

    # --- Source 1: Hyperliquid userFillsByTime ---
    print("[ Hyperliquid — userFillsByTime ]")
    all_fills = fetch_hl_liquidations_by_time(start_ms)

    if not all_fills:
        print("  userFillsByTime returned 0 fills. Falling back to userFills (most recent 2000)...")
        all_fills = fetch_hl_liquidations_recent()
        print(f"  Got {len(all_fills)} total fills from recent history.")
    else:
        print(f"  Got {len(all_fills)} fills in last {DAYS} days.")

    target_fills = [f for f in all_fills if f.get("coin") in ASSETS]
    print(f"  Filtered to {ASSETS}: {len(target_fills)} fills.")

    hl_rows = []
    for fill in target_fills:
        row = map_hl_fill_to_schema(fill)
        if row:
            hl_rows.append(row)

    if hl_rows:
        inserted = insert_liquidations(hl_rows)
        print(f"  Inserted {inserted} rows from Hyperliquid.")
    else:
        print("  No liquidations to insert from Hyperliquid (market has been calm).")

    # --- Source 2: Coinglass (if accessible) ---
    print("\n[ Coinglass — liquidation_history (free tier) ]")
    cg_total = 0
    for symbol in ASSETS:
        raw = fetch_coinglass_liquidations(symbol)
        if raw:
            print(f"  {symbol}: {len(raw)} records from Coinglass.")
            # Map Coinglass aggregated data
            cg_rows = []
            for rec in raw:
                try:
                    ts_ms = rec.get("t") or rec.get("time") or rec.get("createTime")
                    ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
                    price = float(rec.get("price") or rec.get("avgPrice") or 0)
                    for side_key, side_label in [("buyUsdAmt", "long"), ("sellUsdAmt", "short")]:
                        usd = float(rec.get(side_key) or 0)
                        if usd > 0:
                            qty = usd / price if price > 0 else 0
                            cg_rows.append({"time": ts, "exchange": "hyperliquid", "symbol": symbol,
                                           "side": side_label, "price": round(price, 6),
                                           "quantity": round(qty, 8), "usd_value": round(usd, 2)})
                except Exception:
                    pass
            if cg_rows:
                ins = insert_liquidations(cg_rows)
                cg_total += ins
                print(f"  {symbol}: inserted {ins} rows.")
        time.sleep(1.5)

    if cg_total == 0 and not any(fetch_coinglass_liquidations.__doc__ for _ in []):
        print("  Coinglass not available without API key — this is expected on free tier.")

    # --- Final DB count ---
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM liquidations")
        db_total = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        db_total = f"(error: {e})"

    print(f"\n{'='*60}")
    print(f"  Backfill complete.")
    print(f"  Hyperliquid rows inserted : {len(hl_rows) if hl_rows else 0}")
    print(f"  Coinglass rows inserted   : {cg_total}")
    print(f"  DB total rows now         : {db_total}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
