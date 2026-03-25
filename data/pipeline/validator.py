"""
Message validator.

Checks:
1. Staleness — alert if no message received in 2× expected interval
2. Outlier guard — reject if price >10% from last known price
3. Schema validation — required fields must be present and correct type
"""
from __future__ import annotations

import time
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# Expected message interval (seconds) per stream type
EXPECTED_INTERVALS: dict[str, float] = {
    "trades": 5.0,
    "kline": 60.0,
    "orderbook": 2.0,
    "funding": 3600.0,
    "liquidations": 60.0,
}

# Required fields per feed type
REQUIRED_FIELDS: dict[str, list[str]] = {
    "trades": ["ts", "exchange", "symbol", "price", "quantity", "side"],
    "kline": ["ts", "exchange", "symbol", "open", "high", "low", "close", "volume"],
    "orderbook": ["ts", "exchange", "symbol"],
    "funding": ["ts", "exchange", "symbol", "funding_rate"],
    "liquidations": ["ts", "exchange", "symbol", "side", "price", "quantity"],
}

PRICE_OUTLIER_THRESHOLD = 0.10   # 10%


class Validator:
    def __init__(
        self,
        alert_callback: Optional[Any] = None,
    ) -> None:
        """
        alert_callback: async callable(message: str) to send alerts.
        """
        self._alert_callback = alert_callback
        self._last_msg_time: dict[str, float] = {}     # stream_key -> unix ts
        self._last_price: dict[str, float] = {}         # symbol -> last price
        self.log = structlog.get_logger("Validator")

        self._rejected_count = 0
        self._stale_alerts_sent: set[str] = set()

    # ─── Public API ────────────────────────────────────────────────────────
    def validate(self, feed_type: str, msg: dict[str, Any]) -> bool:
        """
        Validate a message synchronously.
        Returns True if valid, False if should be rejected.
        """
        if not self._check_schema(feed_type, msg):
            self._rejected_count += 1
            return False

        if not self._check_outlier(feed_type, msg):
            self._rejected_count += 1
            return False

        # Record message time for staleness tracking
        stream_key = f"{msg.get('exchange', '')}:{feed_type}"
        self._last_msg_time[stream_key] = time.time()

        return True

    def check_staleness(self) -> list[str]:
        """
        Check all tracked streams for staleness.
        Returns list of stale stream keys.
        Should be called periodically (e.g., every 60s).
        """
        stale = []
        now = time.time()
        for stream_key, last_time in self._last_msg_time.items():
            feed_type = stream_key.split(":")[-1]
            expected = EXPECTED_INTERVALS.get(feed_type, 60.0)
            age = now - last_time
            if age > expected * 2:
                stale.append(stream_key)
                self.log.warning(
                    "validator.stale_feed",
                    stream=stream_key,
                    age_s=round(age, 1),
                    threshold_s=expected * 2,
                )
        return stale

    async def check_staleness_and_alert(self) -> None:
        """Check staleness and fire alerts for newly stale feeds."""
        stale_streams = self.check_staleness()
        for stream_key in stale_streams:
            if stream_key not in self._stale_alerts_sent:
                self._stale_alerts_sent.add(stream_key)
                await self._alert(
                    f"⚠️ Stale feed detected: {stream_key} — no messages for >{self._stale_threshold_str(stream_key)}"
                )
        # Clear recovered streams from sent-alerts set
        recovered = self._stale_alerts_sent - set(stale_streams)
        for stream_key in recovered:
            self._stale_alerts_sent.discard(stream_key)
            await self._alert(f"✅ Feed recovered: {stream_key}")

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def get_last_msg_times(self) -> dict[str, float]:
        return dict(self._last_msg_time)

    # ─── Internal ──────────────────────────────────────────────────────────
    def _check_schema(self, feed_type: str, msg: dict[str, Any]) -> bool:
        required = REQUIRED_FIELDS.get(feed_type, [])
        for field in required:
            if field not in msg:
                self.log.warning(
                    "validator.missing_field",
                    feed_type=feed_type,
                    field=field,
                )
                return False
        # Type check price-like fields
        for field in ("price", "open", "high", "low", "close", "quantity", "funding_rate"):
            if field in msg:
                val = msg[field]
                if val is not None and not isinstance(val, (int, float)):
                    self.log.warning(
                        "validator.bad_type",
                        feed_type=feed_type,
                        field=field,
                        value=val,
                    )
                    return False
        return True

    def _check_outlier(self, feed_type: str, msg: dict[str, Any]) -> bool:
        """Reject prices >10% from last known price for same symbol."""
        price_field = "price" if feed_type in ("trades", "liquidations") else "close"
        if price_field not in msg or msg[price_field] is None:
            return True

        price = float(msg[price_field])
        if price <= 0:
            self.log.warning("validator.zero_or_negative_price", feed_type=feed_type, price=price)
            return False

        symbol = msg.get("symbol", "")
        cache_key = f"{msg.get('exchange', '')}:{symbol}"

        if cache_key in self._last_price:
            last = self._last_price[cache_key]
            change = abs(price - last) / last
            if change > PRICE_OUTLIER_THRESHOLD:
                self.log.warning(
                    "validator.outlier_rejected",
                    symbol=symbol,
                    price=price,
                    last_price=last,
                    change_pct=round(change * 100, 2),
                )
                return False

        # Update last price
        self._last_price[cache_key] = price
        return True

    def _stale_threshold_str(self, stream_key: str) -> str:
        feed_type = stream_key.split(":")[-1]
        expected = EXPECTED_INTERVALS.get(feed_type, 60.0)
        threshold = expected * 2
        if threshold >= 60:
            return f"{threshold / 60:.0f}m"
        return f"{threshold:.0f}s"

    async def _alert(self, message: str) -> None:
        self.log.warning("validator.alert", message=message)
        if self._alert_callback:
            try:
                await self._alert_callback(message)
            except Exception as exc:
                self.log.error("validator.alert_failed", error=str(exc))
