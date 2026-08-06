#!/usr/bin/env bash
# Standardized benchmark run: 50 users, spawn rate 5/s, 120 seconds, CSV output.
#
# Prerequisites: API on :8000 (override with HOST=...), worker running,
# Postgres + Redis up. Writes benchmarks/bench_*.csv and prints a summary.
set -euo pipefail

HOST="${HOST:-http://localhost:8000}"
OUT_DIR="benchmarks"
CSV_PREFIX="$OUT_DIR/bench"

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"

echo "==> checking API readiness at $HOST/ready"
if ! curl -sf --max-time 5 "$HOST/ready" > /dev/null; then
    echo "ERROR: API at $HOST is not ready. Start it first (see README)." >&2
    exit 1
fi

echo "==> running locust: 50 users, spawn 5/s, 120s, host=$HOST"
.venv/bin/locust -f scripts/load_test.py \
    --headless \
    --users 50 \
    --spawn-rate 5 \
    --run-time 120s \
    --csv "$CSV_PREFIX" \
    --host "$HOST" \
    --loglevel INFO

echo ""
echo "==> summary (from ${CSV_PREFIX}_stats.csv)"
# Aggregated row is named "Aggregated"; columns (locust 2.32):
# 3 Request Count, 4 Failure Count, 10 Requests/s,
# percentiles start at col 12 (50%): 12=p50, 17=p95, 19=p99.
awk -F',' '
    NR == 1 { next }
    $2 ~ /Aggregated/ {
        printf "  requests:   %s\n", $3
        printf "  failures:   %s\n", $4
        printf "  p50 (ms):   %s\n", $12
        printf "  p95 (ms):   %s\n", $17
        printf "  p99 (ms):   %s\n", $19
        printf "  RPS:        %s\n", $10
    }
' "${CSV_PREFIX}_stats.csv"

echo ""
echo "Full per-endpoint stats: ${CSV_PREFIX}_stats.csv"
