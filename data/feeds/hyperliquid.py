"""
Hyperliquid WebSocket feed.

Connects to wss://api.hyperliquid.xyz/ws
Subscribes to: l2Book, trades, activeAssetCtx (per asset)
Liquidations: polled via REST API every 60s (no WS subscription available)
Assets: BTC, ETH, SOL (configurable via HL_ASSETS env var)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import aiohttp
import structlog
import websockets

from data.feeds.base import BaseFeed

logger = structlog.get_logger(__name__)

HL_REST_URL = "https://api.hyperliquid.xyz/info"


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

        self.log.info("hl.all_subscriptions_sent", assets=self._assets)

        # Start liquidation poller as background task (REST-based, not WS)
        asyncio.create_task(self._liquidation_poll_loop())

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

    async def _liquidation_poll_loop(self) -> None:
        """
        Poll Hyperliquid recentTrades REST endpoint for liquidation events every 30s.

        Hyperliquid does NOT have a dedicated liquidations WS or REST endpoint.
        Liquidated trades are identified by checking if the trade hash appears in
        the clearinghouse liquidation records, OR by using the 'recentTrades' endpoint
        and cross-referencing with known liquidator vault addresses.

        Simpler approach: use recentTrades per asset and flag trades where one of the
        two user addresses is the known HL liquidator vault:
        0x0000000000000000000000000000000000000000 or matches liquidator pattern.

        Best production approach: subscribe to userEvents for the liquidator vault address.
        We use recentTrades + hash inspection as a practical approximation.
        """
        POLL_INTERVAL = 30  # seconds
        # Known Hyperliquid liquidator/system addresses (zero address is liquidator)
        LIQUIDATOR_ADDRESSES = {
            "0x0000000000000000000000000000000000000000",
        }
        last_seen_tids: set[str] = set()

        self.log.info("hl.liquidation_poller_started", interval_s=POLL_INTERVAL)

        while True:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                now_ms = int(time.time() * 1000)
                new_count = 0

                async with aiohttp.ClientSession() as session:
                    for asset in self._assets:
                        payload = {"type": "recentTrades", "coin": asset}
                        async with session.post(
                            HL_REST_URL, json=payload,
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as resp:
                            if resp.status != 200:
                                continue
                            trades = await resp.json()

                        if not isinstance(trades, list):
                            continue

                        for trade in trades:
                            tid = str(trade.get("tid", ""))
                            if tid in last_seen_tids:
                                continue
                            last_seen_tids.add(tid)

                            # Check if any party is the liquidator (zero address)
                            users = trade.get("users", [])
                            is_liquidation = any(
                                u.lower() in LIQUIDATOR_ADDRESSES for u in users
                            )
                            if not is_liquidation:
                                continue

                            # Liquidated party is the non-liquidator
                            # side "A" = ask/sell side was liquidated (long position)
                            # side "B" = bid/buy side was liquidated (short position)
                            side_raw = trade.get("side", "")
                            side = "long" if side_raw == "A" else "short"

                            px = float(trade.get("px", 0))
                            sz = float(trade.get("sz", 0))

                            normalized = {
                                "ts": trade.get("time", now_ms),
                                "exchange": "hyperliquid",
                                "symbol": trade.get("coin", asset),
                                "side": side,
                                "price": px,
                                "quantity": sz,
                                "usd_value": px * sz,
                            }
                            await self.publish("liquidations", normalized)
                            new_count += 1

                # Keep seen set bounded
                if len(last_seen_tids) > 50000:
                    last_seen_tids = set(list(last_seen_tids)[-25000:])

                if new_count > 0:
                    self.log.info("hl.liquidations_detected", count=new_count)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log.error("hl.liquidation_poll_exception", error=str(exc))
                await asyncio.sleep(10)
