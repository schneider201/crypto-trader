.PHONY: up down logs db-shell redis-cli health historical test schema restart-app

# Start all services
up:
	docker compose up -d --build

# Stop all services
down:
	docker compose down

# Tail logs (all services, or pass SERVICE=app)
logs:
	docker compose logs -f $(SERVICE)

# Open psql shell in db container
db-shell:
	docker compose exec db psql -U $${POSTGRES_USER:-trader} -d $${POSTGRES_DB:-trader}

# Open redis-cli in redis container
redis-cli:
	docker compose exec redis redis-cli

# Run health check script
health:
	docker compose exec app python scripts/health_check.py

# Fetch historical data
historical:
	docker compose exec app python scripts/fetch_historical.py

# Run tests
test:
	docker compose exec app pytest tests/ -v

# Apply DB schema
schema:
	docker compose exec db psql -U $${POSTGRES_USER:-trader} -d $${POSTGRES_DB:-trader} -f /app/data/models/schema.sql

# Restart app only
restart-app:
	docker compose restart app
