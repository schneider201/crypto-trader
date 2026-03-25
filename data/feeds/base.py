"""
Abstract WebSocket feed base class.

Features:
- Connect / disconnect / reconnect with exponential backoff (max 10 retries)
- Message counting and latency tracking
- Redis Stream publishing (key: feed:<exchange>:<type>)
- Telegram alert on persistent disconnect
"""
from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import redis.asyncio as aioredis
import structlog
import websockets
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# ─── Prometheus metrics ───────────────────────────────────────────────────────
feed_messages_total = Counter(
    "feed_messages_total",
    "Total messages received from exchange",
    ["exchange", "feed_type"],
)
feed_reconnects_total = Counter(
    "feed_reconnects_total",
    "Total reconnection attempts",
    ["exchange"],
)
feed_connected = Gauge(
    "feed_connected",
    "Whether the feed is currently connected (1=yes, 0=no)",
    ["exchange"],
)
feed_latency_seconds = Histogram(
    "feed_latency_seconds",
    "Message processing latency",
    ["exchange", "feed_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


class BaseFeed(ABC):
    """Abstract base for all exchange WS feeds."""

    MAX_RETRIES: int = 10
    BASE_BACKOFF: float = 1.0      # seconds
    MAX_BACKOFF: float = 60.0      # seconds

    def __init__(
        self,
        redis_url: str,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        self._redis_url = redis_url
        self._telegram_bot_token = telegram_bot_token
        self._telegram_chat_id = telegram_chat_id

        self._redis: Optional[aioredis.Redis] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False

        # Stats
        self._msg_count: int = 0
        self._error_count: int = 0
        self._last_msg_at: Optional[float] = None
        self._connect_time: Optional[float] = None

        self.log = structlog.get_logger(self.__class__.__name__)

    # ─── Abstract interface ────────────────────────────────────────────────
    @property
    @abstractmethod
    def exchange(self) -> str:
        """Exchange name (e.g. 'hyperliquid', 'binance')."""

    @property
    @abstractmethod
    def ws_url(self) -> str:
        """WebSocket endpoint URL."""

    @abstractmethod
    async def on_connect(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Called after successful connection — send subscriptions here."""

    @abstractmethod
    async def handle_message(self, raw: str) -> None:
        """Parse a raw WS message and publish to Redis."""

    # ─── Public API ────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Run feed with automatic reconnection."""
        self._running = True
        self._redis = await aioredis.from_url(self._redis_url, decode_responses=True)
        await self._connect_loop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._redis:
            await self._redis.aclose()
        self.log.info("feed.stopped", exchange=self.exchange)

    # ─── Internal ──────────────────────────────────────────────────────────
    async def _connect_loop(self) -> None:
        retries = 0
        while self._running:
            try:
                self.log.info("feed.connecting", exchange=self.exchange, url=self.ws_url)
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._connect_time = time.time()
                    retries = 0
                    feed_connected.labels(exchange=self.exchange).set(1)
                    self.log.info("feed.connected", exchange=self.exchange)
                    await self.on_connect(ws)
                    await self._recv_loop(ws)

            except (websockets.ConnectionClosed, websockets.WebSocketException) as exc:
                self.log.warning("feed.disconnected", exchange=self.exchange, error=str(exc))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log.error("feed.error", exchange=self.exchange, error=str(exc))
                self._error_count += 1
            finally:
                feed_connected.labels(exchange=self.exchange).set(0)
                self._ws = None

            if not self._running:
                break

            retries += 1
            feed_reconnects_total.labels(exchange=self.exchange).inc()

            if retries >= self.MAX_RETRIES:
                self.log.error(
                    "feed.max_retries_exceeded",
                    exchange=self.exchange,
                    retries=retries,
                )
                await self._send_telegram_alert(
                    f"🚨 [{self.exchange}] Feed permanently disconnected after "
                    f"{retries} retries. Manual intervention required."
                )
                break

            backoff = min(self.BASE_BACKOFF * (2 ** (retries - 1)), self.MAX_BACKOFF)
            self.log.info(
                "feed.reconnecting",
                exchange=self.exchange,
                retry=retries,
                backoff=backoff,
            )
            await asyncio.sleep(backoff)

    async def _recv_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            if not self._running:
                break
            t0 = time.time()
            self._last_msg_at = t0
            self._msg_count += 1
            try:
                await self.handle_message(raw)
            except Exception as exc:
                self.log.warning("feed.parse_error", exchange=self.exchange, error=str(exc))
                self._error_count += 1

    # ─── Redis Stream publishing ───────────────────────────────────────────
    async def publish(self, feed_type: str, data: dict[str, Any]) -> None:
        """Publish a normalised message to a Redis Stream."""
        if not self._redis:
            return
        stream_key = f"feed:{self.exchange}:{feed_type}"
        t0 = time.time()
        try:
            await self._redis.xadd(
                stream_key,
                {"payload": json.dumps(data)},
                maxlen=50_000,
                approximate=True,
            )
            latency = time.time() - t0
            feed_messages_total.labels(exchange=self.exchange, feed_type=feed_type).inc()
            feed_latency_seconds.labels(exchange=self.exchange, feed_type=feed_type).observe(latency)
        except Exception as exc:
            self.log.error("feed.redis_publish_error", error=str(exc))

    # ─── Telegram alerts ───────────────────────────────────────────────────
    async def _send_telegram_alert(self, message: str) -> None:
        if not (self._telegram_bot_token and self._telegram_chat_id):
            return
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self._telegram_bot_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    "chat_id": self._telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                })
        except Exception as exc:
            self.log.error("telegram.alert_failed", error=str(exc))

    # ─── Stats accessors ──────────────────────────────────────────────────
    @property
    def msg_count(self) -> int:
        return self._msg_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def last_msg_at(self) -> Optional[float]:
        return self._last_msg_at

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    def stats(self) -> dict[str, Any]:
        uptime = time.time() - self._connect_time if self._connect_time else 0
        msg_per_min = (self._msg_count / uptime * 60) if uptime > 0 else 0
        return {
            "exchange": self.exchange,
            "connected": self.is_connected,
            "msg_count": self._msg_count,
            "msg_per_min": round(msg_per_min, 2),
            "error_count": self._error_count,
            "last_msg_at": self._last_msg_at,
            "uptime_s": round(uptime, 1),
        }
