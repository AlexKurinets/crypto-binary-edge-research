#!/usr/bin/env bash
# Runs every command block in the README against a running Compose stack.
set -euo pipefail
cd "$(dirname "$0")/.."
DASH_PORT="${EDGELAB_DASHBOARD_PORT:-8501}"
psql() { docker compose exec -T db psql -U edgelab -d edgelab -v ON_ERROR_STOP=1 -tA "$@"; }

echo "== dashboard is up"
curl -fsS "http://localhost:${DASH_PORT}/_stcore/health" | grep -q ok

echo "== the pipeline produced a research run and a paper period"
docker compose run --rm -T pipeline edgelab status > /tmp/status.json  # not piped into head: an early close would fail the run under pipefail
head -5 /tmp/status.json
grep -q '"frozen_hypothesis": "H02"' /tmp/status.json
grep -q '"paper"' /tmp/status.json
[ "$(psql -c 'select count(*) from research.hypotheses where run_id = 1')" = "12" ]
[ "$(psql -c "select count(*) > 0 from paper.decisions where action = 'filled'")" = "t" ]

echo "== the report"
docker compose run --rm -T pipeline edgelab report > /tmp/report.md
grep -q "## Hypotheses" /tmp/report.md
grep -q "## Paper period" /tmp/report.md

echo "== sample size and fee arithmetic"
docker compose run --rm -T pipeline edgelab power --edge 0.02 | tee /tmp/power.json
grep -q '"independent_outcomes_needed": 3865' /tmp/power.json

echo "== a logged paper decision cannot be edited"
if psql -c "update paper.decisions set p_model = 0.99 where decision_id = (select min(decision_id) from paper.decisions)" 2>/tmp/trigger.err; then
  echo "expected the trigger to refuse the edit"; exit 1
fi
grep -q "cannot be changed" /tmp/trigger.err

echo "== re-running the pipeline is a no-op"
docker compose run --rm -T pipeline sh -c "edgelab research --if-none && edgelab paper --if-none" | grep -c '"skipped": true' | grep -q 2

echo "smoke ok"
