-- Enable TimescaleDB extension. Runs automatically on first container start
-- via docker-entrypoint-initdb.d. Table creation itself is handled by
-- scripts/init_db.py (SQLAlchemy metadata) + this script's hypertable calls,
-- which must run AFTER the tables exist. See scripts/init_db.py.
CREATE EXTENSION IF NOT EXISTS timescaledb;
