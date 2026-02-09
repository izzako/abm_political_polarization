#!/bin/bash -l

CONFIG="configs/fj_model_config.ini"

set -a
source .env
set +a

source venv/bin/activate

echo "Running heuristic simulation..."
python -m simulation.heuristic_simulation -c "$CONFIG"