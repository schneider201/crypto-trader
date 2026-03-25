"""
Redis Streams consumer + TimescaleDB batch writer.

Reads from: feed:hyperliquid:* and feed:binance:*
Flush trigger: 100 messages OR 1 second (whichever comes first)
Strategy: UPSERT on conflict
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

import asyncpg
import redis.asyncio as aioredis
import structlog
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# ─── Prometheus metrics ───────────────────────────────────────────────────────
ingestor_writes_total = Counter(
    "ingestor_writes_total",
    "Total rows written to TimescaleDB",
    ["table"],
)
ingestor_write_errors = Counter(
    "ingestor_write_errors_total",
    "Write errors to TimescaleDB",
    ["table"],
)
ingestor_write_latency = Histogram(
    "ingestor_write_latency_seconds",
    "DB write batch latency",
    ["table"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
ingestor_queue_size = Gauge(
    "ingestor_queue_size",
    "Current number of messages in ingestor buffer",
)

STREAMS = [
    "feed:hyperliquid:trades",
    "feed:hyperliquid:orderbook",
    "feed:hyperliquid:funding",
    "feed:hyperliquid:liquidations",
    "feed:binance:kline",
    "feed:binance:trades",
]

BATCH_SIZE = 100
FLUSH_INTERVAL = 1.0   # seconds


def _ts_from_ms(ts_ms: int | float) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


class Ingestor:
    def __init__(self, db_url: str, redis_url: str) -> None:
        self._db_url = db_url
        self._redis_url = redis_url
        self._pool: asyncpg.Pool | None = None
        self._redis: aioredis.Redis | None = None
        self._running = False
        self._buffer: list[tuple[str, dict[str, Any]]] = []
        self._stream_cursors: dict[str, str] = {s: "$" for s in STREAMS}
        self.log = structlog.get_logger("Ingestor")

    async def start(self) -> None:
        self._pool = await asyncpg.create_pool(self._db_url, min_size=2, max_size=10)
        self._redis = await aioredis.from_url(self._redis_url, decode_responses=True)
        self._running = True

        # Initialize stream cursors to latest (don't replay old data)
        for stream in STREAMS:
            try:
                info = await self._redis.xinfo_stream(stream)
                last_id = info.get("last-generated-id") or info.get("last-entry", [None])[0]
                if last_id:
                    self._stream_cursors[stream] = last_id
            except Exception:
                self._stream_cursors[stream] = "0"

        self.log.info("ingestor.started", streams=STREAMS)
        await asyncio.gather(
            self._consume_loop(),
            self._flush_timer(),
        )

    async def stop(self) -> None:
        self._running = False
        if self._buffer:
            await self._flush()
        if self._pool:
            await self._pool.close()
        if self._redis:
            await self._redis.aclose()
        self.log.info("ingestor.stopped")

    # ─── Consumer loop ────────────────────────────────────────────────────
    async def _consume_loop(self) -> None:
        streams_arg = {s: self._stream_cursors[s] for s in STREAMS}

        while self._running:
            try:
                results = await self._redis.xread(
                    streams=streams_arg,
                    count=100,
                    block=100,   # ms
                )
                if not results:
                    continue

                for stream_name, messages in results:
                    for msg_id, fields in messages:
                        self._stream_cursors[stream_name] = msg_id
                        streams_arg[stream_name] = msg_id
                        try:
                            data = json.loads(fields["payload"])
                            self._buffer.append((stream_name, data))
                        except Exception as exc:
                            self.log.warning("ingestor.parse_error", error=str(exc))

                ingestor_queue_size.set(len(self._buffer))

                if len(self._buffer) >= BATCH_SIZE:
                    await self._flush()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log.error("ingestor.consume_error", error=str(exc))
                await asyncio.sleep(1)

    async def _flush_timer(self) -> None:
        while self._running:
            await asyncio.sleep(FLUSH_INTERVAL)
            if self._buffer:
                await self._flush()

    # ─── Flush to DB ──────────────────────────────────────────────────────
    async def _flush(self) -> None:
        if not self._buffer or not self._pool:
            return

        batch = self._buffer[:]
        self._buffer.clear()
        ingestor_queue_size.set(0)

        # Group by table
        trades: list[dict] = []
        candles: list[dict] = []
        funding: list[dict] = []
        liquidations: list[dict] = []
        orderbooks: list[dict] = []

        for stream_name, msg in batch:
            if ":trades" in stream_name:
                trades.append(msg)
            elif ":kline" in stream_name:
                candles.append(msg)
            elif ":funding" in stream_name:
                funding.append(msg)
            elif ":liquidations" in stream_name:
                liquidations.append(msg)
            elif ":orderbook" in stream_name:
                orderbooks.append(msg)

        await asyncio.gather(
            self._write_trades(trades),
            self._write_candles(candles),
            self._write_funding(funding),
            self._write_liquidations(liquidations),
            self._write_orderbooks(orderbooks),
            return_exceptions=True,
        )

    async def _write_trades(self, rows: list[dict]) -> None:
        if not rows:
            return
        t0 = time.time()
        sql = """
            INSERT INTO trades (time, exchange, symbol, trade_id, price, quantity, side, is_maker)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (time, exchange, symbol, trade_id) DO NOTHING
        """
        values = [
            (
                _ts_from_ms(r["ts"]),
                r["exchange"],
                r["symbol"],
                r.get("trade_id", ""),
                r["price"],
                r["quantity"],
                r["side"],
                r.get("is_maker", False),
            )
            for r in rows
        ]
        await self._execute_many("trades", sql, values, t0)

    async def _write_candles(self, rows: list[dict]) -> None:
        if not rows:
            return
        t0 = time.time()
        sql = """
            INSERT INTO candles (time, exchange, symbol, interval, open, high, low, close,
                                 volume, quote_volume, num_trades, is_closed)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (time, exchange, symbol, interval)
            DO UPDATE SET
                high = GREATEST(EXCLUDED.high, candles.high),
                low = LEAST(EXCLUDED.low, candles.low),
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                quote_volume = EXCLUDED.quote_volume,
                num_trades = EXCLUDED.num_trades,
                is_closed = EXCLUDED.is_closed
        """
        values = [
            (
                _ts_from_ms(r["ts"]),
                r["exchange"],
                r["symbol"],
                r.get("interval", "1m"),
                r["open"],
                r["high"],
                r["low"],
                r["close"],
                r["volume"],
                r.get("quote_volume", 0),
                r.get("num_trades", 0),
                r.get("is_closed", False),
            )
            for r in rows
        ]
        await self._execute_many("candles", sql, values, t0)

    async def _write_funding(self, rows: list[dict]) -> None:
        if not rows:
            return
        t0 = time.time()
        sql = """
            INSERT INTO funding_rates (time, exchange, symbol, funding_rate, mark_price, open_interest)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (time, exchange, symbol) DO UPDATE SET
                funding_rate = EXCLUDED.funding_rate,
                mark_price = EXCLUDED.mark_price,
                open_interest = EXCLUDED.open_interest
        """
        values = [
            (
                _ts_from_ms(r["ts"]),
                r["exchange"],
                r["symbol"],
                r.get("funding_rate", 0),
                r.get("mark_price"),
                r.get("open_interest"),
            )
            for r in rows
        ]
        await self._execute_many("funding_rates", sql, values, t0)

    async def _write_liquidations(self, rows: list[dict]) -> None:
        if not rows:
            return
        t0 = time.time()
        sql = """
            INSERT INTO liquidations (time, exchange, symbol, side, price, quantity, usd_value)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (time, exchange, symbol, price, quantity) DO NOTHING
        """
        values = [
            (
                _ts_from_ms(r["ts"]),
                r["exchange"],
                r["symbol"],
                r["side"],
                r["price"],
                r["quantity"],
                r.get("usd_value"),
            )
            for r in rows
        ]
        await self._execute_many("liquidations", sql, values, t0)

    async def _write_orderbooks(self, rows: list[dict]) -> None:
        if not rows:
            return
        t0 = time.time()
        sql = """
            INSERT INTO orderbook_snapshots
                (time, exchange, symbol, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (time, exchange, symbol) DO UPDATE SET
                bid_price = EXCLUDED.bid_price,
                bid_qty = EXCLUDED.bid_qty,
                ask_price = EXCLUDED.ask_price,
                ask_qty = EXCLUDED.ask_qty,
                mid_price = EXCLUDED.mid_price,
                spread = EXCLUDED.spread
        """
        values = [
            (
                _ts_from_ms(r["ts"]),
                r["exchange"],
                r["symbol"],
                r.get("bid_price"),
                r.get("bid_qty"),
                r.get("ask_price"),
                r.get("ask_qty"),
                r.get("mid_price"),
                r.get("spread"),
            )
            for r in rows
        ]
        await self._execute_many("orderbook_snapshots", sql, values, t0)

    async def _execute_many(
        self,
        table: str,
        sql: str,
        values: list[tuple],
        t0: float,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(sql, values)
            latency = time.time() - t0
            ingestor_writes_total.labels(table=table).inc(len(values))
            ingestor_write_latency.labels(table=table).observe(latency)
            self.log.debug(
                "ingestor.flushed",
                table=table,
                rows=len(values),
                latency_ms=round(latency * 1000, 2),
            )
        except Exception as exc:
            ingestor_write_errors.labels(table=table).inc()
            self.log.error("ingestor.write_error", table=table, error=str(exc))
