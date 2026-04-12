#!/bin/bash
set -e

echo "Waiting for database..."
sleep 3

echo "Running migrations..."
alembic upgrade head

echo "Seeding demo locations..."
python -m src.demo.seed_locations

echo "Starting API server..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
