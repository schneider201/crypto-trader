"""
Hyperliquid WebSocket feed.

Connects to wss://api.hyperliquid.xyz/ws
Subscribes to: l2Book, trades, activeAssetCtx, liquidations
Assets: BTC, ETH, SOL (configurable via HL_ASSETS env var)
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


class HyperliquidFeed(BaseFeed):
    WS_URL = "wss://api.hyperliquid.xyz/ws"

    def __init__(self, redis_url: str, assets: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(redis_url, **kwargs)
        env_assets = os.getenv("HL_ASSETS", "BTC,ETH,SOL")
        self._assets = assets or [a.strip() for a in env_assets.split(",")]
        self.log = structlog.get_logger("HyperliquidFeed")

    # ─── BaseFeed interface ────────────────────────────────────────────────
    @property
    def exchange(self) -> str:
        return "hyperliquid"

    @property
    def ws_url(self) -> str:
        return self.WS_URL

    async def on_connect(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Subscribe to all required streams."""
        for asset in self._assets:
            # Order book
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "l2Book", "coin": asset},
            }))
            # Trades
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": asset},
            }))
            # Asset context (funding rate, open interest, mark price, etc.)
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "activeAssetCtx", "coin": asset},
            }))
            self.log.debug("hl.subscribed", asset=asset)

        # All liquidations (not per-asset on HL)
        await ws.send(json.dumps({
            "method": "subscribe",
            "subscription": {"type": "liquidations"},
        }))
        self.log.info("hl.all_subscriptions_sent", assets=self._assets)

    async def handle_message(self, raw: str) -> None:
        """Parse and route incoming Hyperliquid WS messages."""
        msg = json.loads(raw)
        channel = msg.get("channel", "")
        data = msg.get("data", {})

        if channel == "l2Book":
            await self._handle_orderbook(data)
        elif channel == "trades":
            await self._handle_trades(data)
        elif channel == "activeAssetCtx":
            await self._handle_asset_ctx(data)
        elif channel == "liquidations":
            await self._handle_liquidations(data)
        elif channel == "subscriptionResponse":
            self.log.debug("hl.subscription_ack", data=data)
        else:
            self.log.debug("hl.unknown_channel", channel=channel)

    # ─── Message handlers ─────────────────────────────────────────────────
    async def _handle_orderbook(self, data: dict[str, Any]) -> None:
        """Normalise L2 book snapshot."""
        coin = data.get("coin", "UNKNOWN")
        levels = data.get("levels", [[], []])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []

        best_bid = bids[0] if bids else {}
        best_ask = asks[0] if asks else {}

        bid_price = float(best_bid.get("px", 0)) if best_bid else None
        bid_qty   = float(best_bid.get("sz", 0)) if best_bid else None
        ask_price = float(best_ask.get("px", 0)) if best_ask else None
        ask_qty   = float(best_ask.get("sz", 0)) if best_ask else None

        mid = ((bid_price + ask_price) / 2) if (bid_price and ask_price) else None
        spread = (ask_price - bid_price) if (ask_price and bid_price) else None

        normalized = {
            "ts": int(time.time() * 1000),
            "exchange": "hyperliquid",
            "symbol": coin,
            "bid_price": bid_price,
            "bid_qty": bid_qty,
            "ask_price": ask_price,
            "ask_qty": ask_qty,
            "mid_price": mid,
            "spread": spread,
            "levels": {
                "bids": [[float(l["px"]), float(l["sz"])] for l in bids[:10]],
                "asks": [[float(l["px"]), float(l["sz"])] for l in asks[:10]],
            },
        }
        await self.publish("orderbook", normalized)

    async def _handle_trades(self, data: list[dict[str, Any]]) -> None:
        """Normalise trade fills."""
        if not isinstance(data, list):
            data = [data]
        for trade in data:
            normalized = {
                "ts": trade.get("time", int(time.time() * 1000)),
                "exchange": "hyperliquid",
                "symbol": trade.get("coin", "UNKNOWN"),
                "trade_id": str(trade.get("tid", "")),
                "price": float(trade.get("px", 0)),
                "quantity": float(trade.get("sz", 0)),
                "side": "buy" if trade.get("side", "") == "B" else "sell",
                "is_maker": False,  # HL doesn't expose this in WS
            }
            await self.publish("trades", normalized)

    async def _handle_asset_ctx(self, data: dict[str, Any]) -> None:
        """Normalise asset context (funding rate, OI, mark price)."""
        coin = data.get("coin", "UNKNOWN")
        ctx = data.get("ctx", {})

        normalized = {
            "ts": int(time.time() * 1000),
            "exchange": "hyperliquid",
            "symbol": coin,
            "funding_rate": float(ctx.get("funding", 0)),
            "mark_price": float(ctx.get("markPx", 0)) if ctx.get("markPx") else None,
            "open_interest": float(ctx.get("openInterest", 0)) if ctx.get("openInterest") else None,
            "mid_price": float(ctx.get("midPx", 0)) if ctx.get("midPx") else None,
        }
        await self.publish("funding", normalized)

    async def _handle_liquidations(self, data: Any) -> None:
        """Normalise liquidation events."""
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return
        for liq in data:
            normalized = {
                "ts": liq.get("time", int(time.time() * 1000)),
                "exchange": "hyperliquid",
                "symbol": liq.get("coin", "UNKNOWN"),
                "side": "long" if liq.get("side", "") == "A" else "short",
                "price": float(liq.get("px", 0)),
                "quantity": float(liq.get("sz", 0)),
                "usd_value": float(liq.get("px", 0)) * float(liq.get("sz", 0)),
            }
            await self.publish("liquidations", normalized)
