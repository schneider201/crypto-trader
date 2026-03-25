# 🚀 crypto-trader — Phase 0: Data Pipeline

Real-time crypto market data ingestion pipeline using WebSocket feeds (Hyperliquid + Binance),
TimescaleDB for time-series storage, and Redis Streams for internal message passing.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        crypto-trader                         │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │   WS Feeds      │    │        Redis Streams         │    │
│  │                 │    │                              │    │
│  │  Hyperliquid ──►│───►│  feed:hyperliquid:trades     │    │
│  │  (BTC/ETH/SOL)  │    │  feed:hyperliquid:orderbook  │    │
│  │                 │    │  feed:hyperliquid:funding     │    │
│  │  Binance ───────│───►│  feed:binance:kline          │    │
│  │  (BTC/ETH/SOL)  │    │  feed:binance:trades         │    │
│  └─────────────────┘    └──────────────┬───────────────┘    │
│                                        │                     │
│                                        ▼                     │
│                         ┌─────────────────────────┐         │
│                         │  Pipeline Ingestor       │         │
│                         │  (batch writer, 100msg   │         │
│                         │   or 1s flush)           │         │
│                         └────────────┬────────────┘         │
│                                      │                       │
│                                      ▼                       │
│                         ┌─────────────────────────┐         │
│                         │   TimescaleDB            │         │
│                         │   - candles              │         │
│                         │   - trades               │         │
│                         │   - funding_rates        │         │
│                         │   - liquidations         │         │
│                         │   - orderbook_snapshots  │         │
│                         │   - feed_health          │         │
│                         └─────────────────────────┘         │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │   Prometheus    │◄───│   /metrics endpoint          │    │
│  └────────┬────────┘    └──────────────────────────────┘    │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │    Grafana      │  :3000                                  │
│  └─────────────────┘                                         │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/schneider201/crypto-trader.git
cd crypto-trader
cp .env.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN and TELEGRAM_ALERT_CHAT_ID if desired
```

### 2. Start the stack

```bash
make up
```

This starts: app, TimescaleDB, Redis, Prometheus (:9090), Grafana (:3000).

### 3. Apply database schema

```bash
make schema
```

### 4. Fetch historical data

```bash
make historical
```

### 5. Verify feeds are healthy

```bash
make health
```

## Services

| Service    | URL / Port         | Description                   |
|------------|--------------------|-------------------------------|
| App        | -                  | Feed collector + ingestor     |
| TimescaleDB| localhost:5432     | Time-series database          |
| Redis      | localhost:6379     | Stream message bus            |
| Prometheus | http://localhost:9090 | Metrics scraping           |
| Grafana    | http://localhost:3000 | Dashboards (admin/admin)   |

## Makefile Targets

```bash
make up           # Start all services
make down         # Stop all services
make logs         # Tail all logs (SERVICE=app for one)
make db-shell     # Open psql shell
make redis-cli    # Open redis-cli
make health       # Run health check
make historical   # Fetch historical OHLCV data
make test         # Run unit tests
make schema       # Apply DB schema
make restart-app  # Restart app container only
```

## Verification Steps

After `make up` + `make schema`:

```bash
# Check feed health
make health

# Inspect Redis streams
make redis-cli
> XLEN feed:hyperliquid:trades
> XLEN feed:binance:kline

# Check DB rows
make db-shell
trader=# SELECT count(*) FROM trades;
trader=# SELECT count(*) FROM candles;

# View logs
make logs SERVICE=app
```

## Project Structure

```
crypto-trader/
├── data/
│   ├── feeds/         # WS feed clients (Hyperliquid, Binance)
│   ├── models/        # DB schema (TimescaleDB hypertables)
│   └── pipeline/      # Ingestor + validator
├── monitoring/        # Prometheus + Grafana configs
├── scripts/           # CLI utilities
├── tests/             # Unit tests
└── main.py            # Async entrypoint
```
