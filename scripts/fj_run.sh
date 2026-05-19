#!/bin/bash -l

set -a
source .env
set +a

source venv/bin/activate
echo "Running heuristic simulation..."

for TOPIC in all $(seq 0 4); do
    echo "Running with topic ${TOPIC}"

    CONFIG="configs/fj_model_config_${TOPIC}.ini"

    python -m simulation.heuristic_simulation -c "$CONFIG"
done