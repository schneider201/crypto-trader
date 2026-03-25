#!/usr/bin/env python3
"""
Feed health check script.

Prints:
- Feed status (connected/stale)
- Messages per minute
- Last message time
- DB row counts

Exit 0 = healthy, Exit 1 = degraded/down
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

import asyncpg
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://trader:changeme@db:5432/trader")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

STREAMS = [
    "feed:hyperliquid:trades",
    "feed:hyperliquid:orderbook",
    "feed:hyperliquid:funding",
    "feed:hyperliquid:liquidations",
    "feed:binance:kline",
    "feed:binance:trades",
]

STALE_THRESHOLD_S = 120    # seconds — flag if last msg older than this

TABLES = [
    "trades",
    "candles",
    "funding_rates",
    "liquidations",
    "orderbook_snapshots",
    "feed_health",
]


def _ago(ts_ms: int | None) -> str:
    if not ts_ms:
        return "never"
    age = time.time() - ts_ms / 1000
    if age < 60:
        return f"{age:.0f}s ago"
    if age < 3600:
        return f"{age / 60:.1f}m ago"
    return f"{age / 3600:.1f}h ago"


async def check_redis(r: aioredis.Redis) -> tuple[bool, list[dict]]:
    results = []
    healthy = True
    for stream in STREAMS:
        try:
            info = await r.xinfo_stream(stream)
            length = info.get("length", 0)
            last_entry = info.get("last-entry")
            last_ts_ms = None
            if last_entry:
                # last-entry is [id, fields_dict]
                entry_id = last_entry[0] if isinstance(last_entry, list) else None
                if entry_id:
                    # Redis stream ID format: <ms>-<seq>
                    last_ts_ms = int(entry_id.split("-")[0])

            age_s = (time.time() - last_ts_ms / 1000) if last_ts_ms else None
            is_stale = age_s is None or age_s > STALE_THRESHOLD_S

            if is_stale:
                healthy = False

            results.append({
                "stream": stream,
                "length": length,
                "last_msg": _ago(last_ts_ms),
                "stale": is_stale,
            })
        except Exception as exc:
            results.append({
                "stream": stream,
                "length": 0,
                "last_msg": "error",
                "stale": True,
                "error": str(exc),
            })
            healthy = False

    return healthy, results


async def check_db(pool: asyncpg.Pool) -> tuple[bool, dict[str, int]]:
    counts = {}
    healthy = True
    for table in TABLES:
        try:
            row = await pool.fetchrow(f"SELECT count(*) AS n FROM {table}")
            counts[table] = row["n"] if row else 0
        except Exception as exc:
            counts[table] = -1
            healthy = False
    return healthy, counts


async def main() -> int:
    print("=" * 60)
    print("  crypto-trader — Feed Health Check")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    overall_healthy = True

    # ── Redis streams ──
    print("\n📡 Redis Streams\n")
    try:
        r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        redis_ok, stream_results = await check_redis(r)
        await r.aclose()

        for s in stream_results:
            status = "⚠️  STALE" if s["stale"] else "✅ OK   "
            err = f"  [{s.get('error', '')}]" if s.get("error") else ""
            print(f"  {status}  {s['stream']:<40}  len={s['length']:<6}  last={s['last_msg']}{err}")

        if not redis_ok:
            overall_healthy = False
    except Exception as exc:
        print(f"  ❌ Redis connection failed: {exc}")
        overall_healthy = False

    # ── DB row counts ──
    print("\n🗄️  TimescaleDB Row Counts\n")
    try:
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=3)
        db_ok, counts = await check_db(pool)
        await pool.close()

        for table, count in counts.items():
            status = "✅" if count >= 0 else "❌"
            display = f"{count:,}" if count >= 0 else "ERROR"
            print(f"  {status}  {table:<30}  {display}")

        if not db_ok:
            overall_healthy = False
    except Exception as exc:
        print(f"  ❌ DB connection failed: {exc}")
        overall_healthy = False

    print("\n" + "=" * 60)
    if overall_healthy:
        print("  ✅ Overall: HEALTHY")
    else:
        print("  ⚠️  Overall: DEGRADED — check stale feeds above")
    print("=" * 60 + "\n")

    return 0 if overall_healthy else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
