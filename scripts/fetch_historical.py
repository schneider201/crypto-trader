#!/usr/bin/env python3
"""
Historical OHLCV downloader.

Sources:
- Hyperliquid REST (/info endpoint) — candlestick history
- Binance REST — klines endpoint

Stores directly to TimescaleDB.
Shows tqdm progress bars.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import asyncpg
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://trader:changeme@db:5432/trader")
HL_ASSETS = [a.strip() for a in os.getenv("HL_ASSETS", "BTC,ETH,SOL").split(",")]
BINANCE_ASSETS = [a.strip() for a in os.getenv("BINANCE_ASSETS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")]

HL_REST = "https://api.hyperliquid.xyz/info"
BINANCE_REST = "https://api.binance.com/api/v3/klines"

# How many days of history to fetch
DAYS_BACK = int(os.getenv("HISTORY_DAYS", "30"))
INTERVAL = "1m"


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


async def ingest_hl_history(pool: asyncpg.Pool) -> None:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - DAYS_BACK * 86400 * 1000
    # Chunk into 1-day windows to avoid huge responses
    chunk_ms = 86400 * 1000

    async with aiohttp.ClientSession() as session:
        for coin in HL_ASSETS:
            chunks = []
            t = start_ms
            while t < now_ms:
                chunks.append((t, min(t + chunk_ms, now_ms)))
                t += chunk_ms

            pbar = tqdm(chunks, desc=f"HL {coin}", unit="day", ncols=80)
            total_rows = 0

            for chunk_start, chunk_end in pbar:
                try:
                    candles = await fetch_hl_candles(session, coin, chunk_start, chunk_end)
                    if not candles:
                        continue
                    rows = []
                    for c in candles:
                        ts = datetime.fromtimestamp(c["t"] / 1000, tz=timezone.utc)
                        rows.append((
                            ts,
                            "hyperliquid",
                            coin,
                            INTERVAL,
                            float(c["o"]),
                            float(c["h"]),
                            float(c["l"]),
                            float(c["c"]),
                            float(c["v"]),
                            0.0,
                            0,
                            True,
                        ))
                    n = await insert_candles(pool, rows)
                    total_rows += n
                    pbar.set_postfix(rows=total_rows)
                    await asyncio.sleep(0.1)   # rate-limit courtesy
                except Exception as exc:
                    tqdm.write(f"  ⚠ HL {coin} error: {exc}")

            tqdm.write(f"✅ HL {coin}: {total_rows} candles inserted")


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
    # Binance returns max 1000 candles per request; 1m candles = ~16.7h per request
    chunk_ms = 1000 * 60 * 1000   # 1000 minutes

    async with aiohttp.ClientSession() as session:
        for symbol in BINANCE_ASSETS:
            chunks = []
            t = start_ms
            while t < now_ms:
                chunks.append((t, min(t + chunk_ms, now_ms)))
                t += chunk_ms

            pbar = tqdm(chunks, desc=f"BN {symbol}", unit="chunk", ncols=80)
            total_rows = 0

            for chunk_start, chunk_end in pbar:
                try:
                    klines = await fetch_binance_klines(session, symbol, chunk_start, chunk_end)
                    if not klines:
                        continue
                    rows = []
                    for k in klines:
                        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                        rows.append((
                            ts,
                            "binance",
                            symbol,
                            INTERVAL,
                            float(k[1]),   # open
                            float(k[2]),   # high
                            float(k[3]),   # low
                            float(k[4]),   # close
                            float(k[5]),   # volume
                            float(k[7]),   # quote asset volume
                            int(k[8]),     # num trades
                            True,
                        ))
                    n = await insert_candles(pool, rows)
                    total_rows += n
                    pbar.set_postfix(rows=total_rows)
                    await asyncio.sleep(0.05)
                except Exception as exc:
                    tqdm.write(f"  ⚠ Binance {symbol} error: {exc}")

            tqdm.write(f"✅ Binance {symbol}: {total_rows} candles inserted")


# ─── Entry ────────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"\n🚀 Fetching {DAYS_BACK} days of 1m OHLCV data\n")
    pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
    try:
        await ingest_hl_history(pool)
        await ingest_binance_history(pool)
    finally:
        await pool.close()
    print("\n✅ Historical data fetch complete\n")


if __name__ == "__main__":
    asyncio.run(main())
