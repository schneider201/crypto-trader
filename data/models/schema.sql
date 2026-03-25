-- crypto-trader Phase 0 — TimescaleDB schema
-- Run via: make schema

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ─────────────────────────────────────────────
-- candles (OHLCV)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS candles (
    time        TIMESTAMPTZ     NOT NULL,
    exchange    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    interval    TEXT            NOT NULL,   -- '1m', '5m', etc.
    open        NUMERIC(24, 8)  NOT NULL,
    high        NUMERIC(24, 8)  NOT NULL,
    low         NUMERIC(24, 8)  NOT NULL,
    close       NUMERIC(24, 8)  NOT NULL,
    volume      NUMERIC(32, 8)  NOT NULL,
    quote_volume NUMERIC(32, 8) DEFAULT 0,
    num_trades  INTEGER         DEFAULT 0,
    is_closed   BOOLEAN         DEFAULT FALSE,
    PRIMARY KEY (time, exchange, symbol, interval)
);

SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_candles_exchange_symbol ON candles (exchange, symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles (symbol, time DESC);

-- ─────────────────────────────────────────────
-- trades (individual fills)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    time        TIMESTAMPTZ     NOT NULL,
    exchange    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    trade_id    TEXT,
    price       NUMERIC(24, 8)  NOT NULL,
    quantity    NUMERIC(32, 8)  NOT NULL,
    side        TEXT            NOT NULL,   -- 'buy' | 'sell'
    is_maker    BOOLEAN         DEFAULT FALSE,
    PRIMARY KEY (time, exchange, symbol, trade_id)
);

SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_trades_exchange_symbol ON trades (exchange, symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, time DESC);

-- ─────────────────────────────────────────────
-- funding_rates (perps)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS funding_rates (
    time            TIMESTAMPTZ     NOT NULL,
    exchange        TEXT            NOT NULL,
    symbol          TEXT            NOT NULL,
    funding_rate    NUMERIC(18, 10) NOT NULL,
    mark_price      NUMERIC(24, 8),
    index_price     NUMERIC(24, 8),
    open_interest   NUMERIC(32, 8),
    PRIMARY KEY (time, exchange, symbol)
);

SELECT create_hypertable('funding_rates', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_funding_exchange_symbol ON funding_rates (exchange, symbol, time DESC);

-- ─────────────────────────────────────────────
-- liquidations
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS liquidations (
    time        TIMESTAMPTZ     NOT NULL,
    exchange    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    side        TEXT            NOT NULL,   -- 'long' | 'short'
    price       NUMERIC(24, 8)  NOT NULL,
    quantity    NUMERIC(32, 8)  NOT NULL,
    usd_value   NUMERIC(32, 2),
    PRIMARY KEY (time, exchange, symbol, price, quantity)
);

SELECT create_hypertable('liquidations', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_liquidations_exchange_symbol ON liquidations (exchange, symbol, time DESC);

-- ─────────────────────────────────────────────
-- orderbook_snapshots (L2 top-of-book)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    time        TIMESTAMPTZ     NOT NULL,
    exchange    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    bid_price   NUMERIC(24, 8),
    bid_qty     NUMERIC(32, 8),
    ask_price   NUMERIC(24, 8),
    ask_qty     NUMERIC(32, 8),
    mid_price   NUMERIC(24, 8),
    spread      NUMERIC(24, 8),
    levels_json JSONB,                      -- full book levels if stored
    PRIMARY KEY (time, exchange, symbol)
);

SELECT create_hypertable('orderbook_snapshots', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ob_exchange_symbol ON orderbook_snapshots (exchange, symbol, time DESC);

-- ─────────────────────────────────────────────
-- feed_health (monitoring)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feed_health (
    time            TIMESTAMPTZ     NOT NULL,
    exchange        TEXT            NOT NULL,
    feed_type       TEXT            NOT NULL,   -- 'trades', 'kline', 'orderbook', etc.
    msg_count       BIGINT          DEFAULT 0,
    msg_per_min     NUMERIC(10, 2)  DEFAULT 0,
    last_msg_at     TIMESTAMPTZ,
    is_connected    BOOLEAN         DEFAULT FALSE,
    latency_ms      NUMERIC(10, 2),
    error_count     INTEGER         DEFAULT 0,
    PRIMARY KEY (time, exchange, feed_type)
);

SELECT create_hypertable('feed_health', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_feed_health_exchange ON feed_health (exchange, feed_type, time DESC);
