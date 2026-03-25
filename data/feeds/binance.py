"""
Binance WebSocket feed.

Connects to wss://stream.binance.com:9443/stream
Subscribes to: kline_1m and aggTrade streams
Assets: BTCUSDT, ETHUSDT, SOLUSDT (configurable via BINANCE_ASSETS env var)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import structlog
import websockets

from data.feeds.base import BaseFeed

logger = structlog.get_logger(__name__)


class BinanceFeed(BaseFeed):
    WS_BASE = "wss://stream.binance.com:9443/stream"

    def __init__(self, redis_url: str, assets: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(redis_url, **kwargs)
        env_assets = os.getenv("BINANCE_ASSETS", "BTCUSDT,ETHUSDT,SOLUSDT")
        self._assets = assets or [a.strip() for a in env_assets.split(",")]
        self.log = structlog.get_logger("BinanceFeed")

    # ─── BaseFeed interface ────────────────────────────────────────────────
    @property
    def exchange(self) -> str:
        return "binance"

    @property
    def ws_url(self) -> str:
        """Build combined stream URL."""
        streams = []
        for asset in self._assets:
            symbol = asset.lower()
            streams.append(f"{symbol}@kline_1m")
            streams.append(f"{symbol}@aggTrade")
        return f"{self.WS_BASE}?streams={'/'.join(streams)}"

    async def on_connect(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Binance combined streams auto-subscribe via URL — nothing to send."""
        self.log.info(
            "binance.connected",
            assets=self._assets,
            stream_count=len(self._assets) * 2,
        )

    async def handle_message(self, raw: str) -> None:
        """Parse Binance combined stream envelope."""
        msg = json.loads(raw)
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        event_type = data.get("e", "")

        if event_type == "kline":
            await self._handle_kline(data)
        elif event_type == "aggTrade":
            await self._handle_agg_trade(data)
        else:
            self.log.debug("binance.unknown_event", stream=stream, event=event_type)

    # ─── Message handlers ─────────────────────────────────────────────────
    async def _handle_kline(self, data: dict[str, Any]) -> None:
        """Normalise kline (candlestick) message."""
        k = data.get("k", {})
        symbol = data.get("s", "UNKNOWN")

        normalized = {
            "ts": data.get("E", int(time.time() * 1000)),
            "exchange": "binance",
            "symbol": symbol,
            "interval": k.get("i", "1m"),
            "open": float(k.get("o", 0)),
            "high": float(k.get("h", 0)),
            "low": float(k.get("l", 0)),
            "close": float(k.get("c", 0)),
            "volume": float(k.get("v", 0)),
            "quote_volume": float(k.get("q", 0)),
            "num_trades": int(k.get("n", 0)),
            "is_closed": bool(k.get("x", False)),
            "kline_open_time": k.get("t"),
            "kline_close_time": k.get("T"),
        }
        await self.publish("kline", normalized)

    async def _handle_agg_trade(self, data: dict[str, Any]) -> None:
        """Normalise aggregated trade message."""
        symbol = data.get("s", "UNKNOWN")

        normalized = {
            "ts": data.get("E", int(time.time() * 1000)),
            "exchange": "binance",
            "symbol": symbol,
            "trade_id": str(data.get("a", "")),          # agg trade id
            "price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "side": "sell" if data.get("m", False) else "buy",   # m=True means buyer is maker → taker is seller
            "is_maker": bool(data.get("m", False)),
            "first_trade_id": data.get("f"),
            "last_trade_id": data.get("l"),
        }
        await self.publish("trades", normalized)
