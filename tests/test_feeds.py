"""
Unit tests for feed message normalization, validator staleness, and outlier detection.
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.pipeline.validator import Validator


# ─── Helpers ─────────────────────────────────────────────────────────────────
def make_hl_trade_msg(coin="BTC", price=50000.0, sz=0.1, side="B", tid=12345):
    """Raw Hyperliquid trade WS message."""
    return {
        "channel": "trades",
        "data": [
            {
                "coin": coin,
                "px": str(price),
                "sz": str(sz),
                "side": side,
                "time": int(time.time() * 1000),
                "tid": tid,
            }
        ],
    }


def make_binance_kline_msg(symbol="BTCUSDT", close=50000.0, is_closed=True):
    """Raw Binance kline WS message."""
    return {
        "stream": f"{symbol.lower()}@kline_1m",
        "data": {
            "e": "kline",
            "E": int(time.time() * 1000),
            "s": symbol,
            "k": {
                "t": int(time.time() * 1000) - 60000,
                "T": int(time.time() * 1000),
                "s": symbol,
                "i": "1m",
                "o": "49900.0",
                "c": str(close),
                "h": "50100.0",
                "l": "49800.0",
                "v": "10.5",
                "q": "525000.0",
                "n": 350,
                "x": is_closed,
            },
        },
    }


def make_binance_trade_msg(symbol="BTCUSDT", price=50000.0, qty=0.01, is_maker=False):
    """Raw Binance aggTrade WS message."""
    return {
        "stream": f"{symbol.lower()}@aggTrade",
        "data": {
            "e": "aggTrade",
            "E": int(time.time() * 1000),
            "s": symbol,
            "a": 99999,
            "p": str(price),
            "q": str(qty),
            "f": 1,
            "l": 1,
            "T": int(time.time() * 1000),
            "m": is_maker,
        },
    }


# ─── Normalization: Hyperliquid ───────────────────────────────────────────────
class TestHyperliquidNormalization:

    @pytest.mark.asyncio
    async def test_trade_normalization(self):
        """HL trade should normalize side B → buy, A → sell."""
        from data.feeds.hyperliquid import HyperliquidFeed

        feed = HyperliquidFeed.__new__(HyperliquidFeed)
        feed._redis = None
        published = []

        async def mock_publish(feed_type, data):
            published.append((feed_type, data))

        feed.publish = mock_publish

        msg = make_hl_trade_msg(coin="BTC", price=50000.0, side="B")
        await feed._handle_trades(msg["data"])

        assert len(published) == 1
        ft, data = published[0]
        assert ft == "trades"
        assert data["exchange"] == "hyperliquid"
        assert data["symbol"] == "BTC"
        assert data["price"] == 50000.0
        assert data["side"] == "buy"

    @pytest.mark.asyncio
    async def test_trade_sell_side(self):
        feed = object.__new__(__import__("data.feeds.hyperliquid", fromlist=["HyperliquidFeed"]).HyperliquidFeed)
        from data.feeds.hyperliquid import HyperliquidFeed
        feed = HyperliquidFeed.__new__(HyperliquidFeed)
        published = []

        async def mock_publish(feed_type, data):
            published.append((feed_type, data))

        feed.publish = mock_publish

        await feed._handle_trades([{
            "coin": "ETH", "px": "3000.0", "sz": "1.5", "side": "A",
            "time": int(time.time() * 1000), "tid": 1
        }])
        assert published[0][1]["side"] == "sell"

    @pytest.mark.asyncio
    async def test_orderbook_normalization(self):
        from data.feeds.hyperliquid import HyperliquidFeed
        feed = HyperliquidFeed.__new__(HyperliquidFeed)
        published = []

        async def mock_publish(ft, data):
            published.append((ft, data))

        feed.publish = mock_publish

        ob_data = {
            "coin": "BTC",
            "levels": [
                [{"px": "49990.0", "sz": "0.5"}, {"px": "49980.0", "sz": "1.0"}],
                [{"px": "50010.0", "sz": "0.3"}, {"px": "50020.0", "sz": "0.8"}],
            ],
        }
        await feed._handle_orderbook(ob_data)

        assert len(published) == 1
        ft, data = published[0]
        assert ft == "orderbook"
        assert data["bid_price"] == 49990.0
        assert data["ask_price"] == 50010.0
        assert data["mid_price"] == pytest.approx(50000.0)
        assert data["spread"] == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_funding_normalization(self):
        from data.feeds.hyperliquid import HyperliquidFeed
        feed = HyperliquidFeed.__new__(HyperliquidFeed)
        published = []

        async def mock_publish(ft, data):
            published.append((ft, data))

        feed.publish = mock_publish

        ctx_data = {
            "coin": "BTC",
            "ctx": {
                "funding": "0.0001",
                "markPx": "50100.0",
                "openInterest": "12345.6",
                "midPx": "50090.0",
            },
        }
        await feed._handle_asset_ctx(ctx_data)

        assert len(published) == 1
        ft, data = published[0]
        assert ft == "funding"
        assert data["funding_rate"] == pytest.approx(0.0001)
        assert data["mark_price"] == pytest.approx(50100.0)


# ─── Normalization: Binance ───────────────────────────────────────────────────
class TestBinanceNormalization:

    @pytest.mark.asyncio
    async def test_kline_normalization(self):
        from data.feeds.binance import BinanceFeed
        feed = BinanceFeed.__new__(BinanceFeed)
        published = []

        async def mock_publish(ft, data):
            published.append((ft, data))

        feed.publish = mock_publish

        msg = make_binance_kline_msg("BTCUSDT", close=50000.0, is_closed=True)
        await feed._handle_kline(msg["data"])

        assert len(published) == 1
        ft, data = published[0]
        assert ft == "kline"
        assert data["exchange"] == "binance"
        assert data["symbol"] == "BTCUSDT"
        assert data["close"] == pytest.approx(50000.0)
        assert data["is_closed"] is True
        assert data["interval"] == "1m"

    @pytest.mark.asyncio
    async def test_trade_buy_maker_false(self):
        from data.feeds.binance import BinanceFeed
        feed = BinanceFeed.__new__(BinanceFeed)
        published = []

        async def mock_publish(ft, data):
            published.append((ft, data))

        feed.publish = mock_publish

        msg = make_binance_trade_msg("ETHUSDT", price=3000.0, is_maker=False)
        await feed._handle_agg_trade(msg["data"])

        ft, data = published[0]
        assert ft == "trades"
        assert data["side"] == "buy"     # m=False → taker is buyer
        assert data["price"] == pytest.approx(3000.0)

    @pytest.mark.asyncio
    async def test_trade_sell_maker_true(self):
        from data.feeds.binance import BinanceFeed
        feed = BinanceFeed.__new__(BinanceFeed)
        published = []

        async def mock_publish(ft, data):
            published.append((ft, data))

        feed.publish = mock_publish

        msg = make_binance_trade_msg("SOLUSDT", price=150.0, is_maker=True)
        await feed._handle_agg_trade(msg["data"])

        _, data = published[0]
        assert data["side"] == "sell"    # m=True → taker is seller


# ─── Validator: staleness ─────────────────────────────────────────────────────
class TestValidatorStaleness:

    def test_no_staleness_when_fresh(self):
        v = Validator()
        msg = {
            "ts": int(time.time() * 1000),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "quantity": 0.1,
            "side": "buy",
        }
        v.validate("trades", msg)
        stale = v.check_staleness()
        assert stale == []

    def test_staleness_detected(self):
        v = Validator()
        # Manually plant a stale timestamp
        v._last_msg_time["binance:trades"] = time.time() - 999.0   # way past 2× 5s
        stale = v.check_staleness()
        assert "binance:trades" in stale

    def test_freshly_received_not_stale(self):
        v = Validator()
        v._last_msg_time["hyperliquid:orderbook"] = time.time() - 1.0   # 1 second ago, threshold is 4s
        stale = v.check_staleness()
        assert "hyperliquid:orderbook" not in stale


# ─── Validator: outlier ───────────────────────────────────────────────────────
class TestValidatorOutlier:

    def test_normal_price_accepted(self):
        v = Validator()
        msg = {
            "ts": int(time.time() * 1000),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "quantity": 0.1,
            "side": "buy",
        }
        assert v.validate("trades", msg) is True
        # Price within 5% — should pass
        msg2 = dict(msg)
        msg2["price"] = 52000.0
        assert v.validate("trades", msg2) is True

    def test_outlier_price_rejected(self):
        v = Validator()
        msg = {
            "ts": int(time.time() * 1000),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "quantity": 0.1,
            "side": "buy",
        }
        v.validate("trades", msg)
        # 20% jump — should be rejected
        msg2 = dict(msg)
        msg2["price"] = 60001.0
        result = v.validate("trades", msg2)
        assert result is False
        assert v.rejected_count >= 1

    def test_zero_price_rejected(self):
        v = Validator()
        msg = {
            "ts": int(time.time() * 1000),
            "exchange": "hyperliquid",
            "symbol": "ETH",
            "price": 0.0,
            "quantity": 1.0,
            "side": "sell",
        }
        assert v.validate("trades", msg) is False

    def test_missing_field_rejected(self):
        v = Validator()
        # Missing 'price' field
        msg = {
            "ts": int(time.time() * 1000),
            "exchange": "binance",
            "symbol": "SOLUSDT",
            "quantity": 10.0,
            "side": "buy",
        }
        assert v.validate("trades", msg) is False
