"""
crypto-trader Phase 0 — async entrypoint.

Starts:
- Hyperliquid WS feed
- Binance WS feed
- Redis → TimescaleDB ingestor
- Health reporter (every 60s)

Handles: SIGTERM, SIGINT gracefully.
Telegram: notify on start/stop.
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any

import structlog
from dotenv import load_dotenv
from prometheus_client import start_http_server

from data.feeds.binance import BinanceFeed
from data.feeds.hyperliquid import HyperliquidFeed
from data.pipeline.ingestor import Ingestor
from data.pipeline.validator import Validator

load_dotenv()

# ─── Logging setup ────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("ENVIRONMENT") == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger("main")


# ─── Telegram helper ──────────────────────────────────────────────────────────
async def send_telegram(message: str) -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_ALERT_CHAT_ID", "")
    if not (bot_token and chat_id):
        return
    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
            })
    except Exception as exc:
        log.warning("telegram.send_failed", error=str(exc))


# ─── Health reporter ──────────────────────────────────────────────────────────
async def health_reporter(feeds: list[Any], interval: float = 60.0) -> None:
    while True:
        await asyncio.sleep(interval)
        for feed in feeds:
            stats = feed.stats()
            log.info("health.report", **stats)


# ─── Main ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    db_url = os.getenv("DATABASE_URL", "postgresql://trader:changeme@db:5432/trader")
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.getenv("TELEGRAM_ALERT_CHAT_ID", "")

    # Start Prometheus metrics server
    metrics_port = int(os.getenv("METRICS_PORT", "8000"))
    start_http_server(metrics_port)
    log.info("metrics.server_started", port=metrics_port)

    validator = Validator(alert_callback=send_telegram)

    hl_feed = HyperliquidFeed(
        redis_url=redis_url,
        telegram_bot_token=tg_token,
        telegram_chat_id=tg_chat,
    )
    bn_feed = BinanceFeed(
        redis_url=redis_url,
        telegram_bot_token=tg_token,
        telegram_chat_id=tg_chat,
    )
    ingestor = Ingestor(db_url=db_url, redis_url=redis_url)

    # Graceful shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        log.info("main.shutdown_signal_received")
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    loop.add_signal_handler(signal.SIGINT, _signal_handler)

    await send_telegram("🚀 crypto-trader started — Phase 0 data pipeline is live.")
    log.info("main.starting", db_url=db_url, redis_url=redis_url)
    start_time = time.time()

    # Start all tasks
    tasks = [
        asyncio.create_task(hl_feed.start(), name="hl_feed"),
        asyncio.create_task(bn_feed.start(), name="bn_feed"),
        asyncio.create_task(ingestor.start(), name="ingestor"),
        asyncio.create_task(health_reporter([hl_feed, bn_feed]), name="health_reporter"),
        asyncio.create_task(
            _staleness_monitor(validator, send_telegram), name="staleness_monitor"
        ),
    ]

    # Wait for shutdown signal
    await shutdown_event.wait()
    log.info("main.shutting_down")

    # Cancel all tasks
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    # Flush ingestor
    await ingestor.stop()
    await hl_feed.stop()
    await bn_feed.stop()

    uptime = time.time() - start_time
    await send_telegram(
        f"🛑 crypto-trader stopped after {uptime:.0f}s uptime.\n"
        f"HL msgs: {hl_feed.msg_count} | Binance msgs: {bn_feed.msg_count}"
    )
    log.info("main.stopped", uptime_s=round(uptime, 1))


async def _staleness_monitor(validator: Validator, alert_fn: Any) -> None:
    while True:
        await asyncio.sleep(60)
        await validator.check_staleness_and_alert()


if __name__ == "__main__":
    asyncio.run(main())
