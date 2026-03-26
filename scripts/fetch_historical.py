#!/usr/bin/env python3
"""
Historical OHLCV + Funding Rate downloader.

Sources:
- Hyperliquid REST (/info endpoint) — candlestick + funding history
- Binance REST — klines endpoint

Stores to:
  1. TimescaleDB (for live system use)
  2. Parquet files under /app/data/historical/ (for backtesting — fast, compressed)

Shows tqdm progress bars.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
import asyncpg
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://trader:changeme@db:5432/trader")
HL_ASSETS = [a.strip() for a in os.getenv("HL_ASSETS", "BTC,ETH,SOL").split(",")]
BINANCE_ASSETS = [a.strip() for a in os.getenv("BINANCE_ASSETS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")]

HL_REST = "https://api.hyperliquid.xyz/info"
BINANCE_REST = "https://api.binance.com/api/v3/klines"
HEADERS = {"User-Agent": "crypto-trader-research/1.0"}

# How many days of history to fetch
DAYS_BACK = int(os.getenv("HISTORY_DAYS", "365"))
INTERVAL = "1m"

# Parquet output directory
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", "/app/data/historical"))
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


# ─── DB helpers ───────────────────────────────────────────────────────────────
async def insert_candles(pool: asyncpg.Pool, rows: list[tuple]) -> int:
    sql = """
        INSERT INTO candles (time, exchange, symbol, interval, open, high, low, close,
                             volume, quote_volume, num_trades, is_closed)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (time, exchange, symbol, interval) DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.executemany(sql, rows)
    return len(rows)


# ─── Hyperliquid ──────────────────────────────────────────────────────────────
async def fetch_hl_candles(
    session: aiohttp.ClientSession,
    coin: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    async with session.post(HL_REST, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data if isinstance(data, list) else []


async def fetch_hl_funding(
    session: aiohttp.ClientSession,
    coin: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    payload = {
        "type": "fundingHistory",
        "coin": coin,
        "startTime": start_ms,
        "endTime": end_ms,
    }
    async with session.post(HL_REST, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data if isinstance(data, list) else []


async def insert_funding(pool: asyncpg.Pool, rows: list[tuple]) -> int:
    sql = """
        INSERT INTO funding_rates (time, exchange, symbol, funding_rate, mark_price, open_interest)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (time, exchange, symbol) DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.executemany(sql, rows)
    return len(rows)


async def ingest_hl_history(pool: asyncpg.Pool) -> None:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - DAYS_BACK * 86400 * 1000
    chunk_ms = 86400 * 1000  # 1-day chunks

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for coin in HL_ASSETS:
            chunks = []
            t = start_ms
            while t < now_ms:
                chunks.append((t, min(t + chunk_ms, now_ms)))
                t += chunk_ms

            pbar = tqdm(chunks, desc=f"HL candles {coin}", unit="day", ncols=80)
            total_rows = 0
            all_candles: list[dict] = []

            for chunk_start, chunk_end in pbar:
                try:
                    candles = await fetch_hl_candles(session, coin, chunk_start, chunk_end)
                    if not candles:
                        continue
                    rows = []
                    for c in candles:
                        ts = datetime.fromtimestamp(c["t"] / 1000, tz=timezone.utc)
                        rows.append((
                            ts, "hyperliquid", coin, INTERVAL,
                            float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]),
                            float(c["v"]), 0.0, 0, True,
                        ))
                        all_candles.append({
                            "time": ts, "open": float(c["o"]), "high": float(c["h"]),
                            "low": float(c["l"]), "close": float(c["c"]), "volume": float(c["v"]),
                        })
                    n = await insert_candles(pool, rows)
                    total_rows += n
                    pbar.set_postfix(rows=total_rows)
                    await asyncio.sleep(0.1)
                except Exception as exc:
                    tqdm.write(f"  ⚠ HL {coin} candle error: {exc}")

            # Save Parquet
            if all_candles:
                df = pd.DataFrame(all_candles).set_index("time").sort_index()
                path = PARQUET_DIR / f"hl_{coin.lower()}_candles_1m.parquet"
                df.to_parquet(path, compression="snappy")
                tqdm.write(f"✅ HL {coin}: {total_rows} candles → DB + {path.name} ({path.stat().st_size // 1024}KB)")

            # Fetch funding rates
            pbar2 = tqdm(chunks, desc=f"HL funding {coin}", unit="day", ncols=80)
            funding_rows_total = 0
            all_funding: list[dict] = []

            for chunk_start, chunk_end in pbar2:
                try:
                    funding = await fetch_hl_funding(session, coin, chunk_start, chunk_end)
                    if not funding:
                        continue
                    rows = []
                    for f in funding:
                        ts = datetime.fromtimestamp(f["time"] / 1000, tz=timezone.utc)
                        rate = float(f.get("fundingRate", 0))
                        rows.append((ts, "hyperliquid", coin, rate, None, None))
                        all_funding.append({"time": ts, "funding_rate": rate,
                                           "premium": float(f.get("premium", 0))})
                    n = await insert_funding(pool, rows)
                    funding_rows_total += n
                    pbar2.set_postfix(rows=funding_rows_total)
                    await asyncio.sleep(0.05)
                except Exception as exc:
                    tqdm.write(f"  ⚠ HL {coin} funding error: {exc}")

            if all_funding:
                df_f = pd.DataFrame(all_funding).set_index("time").sort_index()
                path_f = PARQUET_DIR / f"hl_{coin.lower()}_funding_1h.parquet"
                df_f.to_parquet(path_f, compression="snappy")
                tqdm.write(f"✅ HL {coin}: {funding_rows_total} funding rows → DB + {path_f.name}")


# ─── Binance ──────────────────────────────────────────────────────────────────
async def fetch_binance_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list[list]:
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    async with session.get(BINANCE_REST, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def ingest_binance_history(pool: asyncpg.Pool) -> None:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - DAYS_BACK * 86400 * 1000
    chunk_ms = 1000 * 60 * 1000  # 1000 minutes per request (Binance limit)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for symbol in BINANCE_ASSETS:
            chunks = []
            t = start_ms
            while t < now_ms:
                chunks.append((t, min(t + chunk_ms, now_ms)))
                t += chunk_ms

            pbar = tqdm(chunks, desc=f"BN {symbol}", unit="chunk", ncols=80)
            total_rows = 0
            all_candles: list[dict] = []

            for chunk_start, chunk_end in pbar:
                try:
                    klines = await fetch_binance_klines(session, symbol, chunk_start, chunk_end)
                    if not klines:
                        continue
                    rows = []
                    for k in klines:
                        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                        rows.append((
                            ts, "binance", symbol, INTERVAL,
                            float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                            float(k[5]), float(k[7]), int(k[8]), True,
                        ))
                        all_candles.append({
                            "time": ts, "open": float(k[1]), "high": float(k[2]),
                            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                            "quote_volume": float(k[7]), "num_trades": int(k[8]),
                        })
                    n = await insert_candles(pool, rows)
                    total_rows += n
                    pbar.set_postfix(rows=total_rows)
                    await asyncio.sleep(0.05)
                except Exception as exc:
                    tqdm.write(f"  ⚠ Binance {symbol} error: {exc}")

            if all_candles:
                df = pd.DataFrame(all_candles).set_index("time").sort_index()
                path = PARQUET_DIR / f"binance_{symbol.lower()}_candles_1m.parquet"
                df.to_parquet(path, compression="snappy")
                tqdm.write(f"✅ Binance {symbol}: {total_rows} candles → DB + {path.name} ({path.stat().st_size // 1024}KB)")


# ─── Entry ────────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"\n🚀 Fetching {DAYS_BACK} days of 1m OHLCV + funding data")
    print(f"   Parquet output: {PARQUET_DIR}\n")
    pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
    try:
        await ingest_hl_history(pool)
        await ingest_binance_history(pool)
    finally:
        await pool.close()
    print(f"\n✅ Done. Parquet files saved to {PARQUET_DIR}\n")


if __name__ == "__main__":
    asyncio.run(main())
