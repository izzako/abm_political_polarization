#!/bin/bash -l

# Set SCC project
#$ -P llamagrp

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=48:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N abm_pp_qwen3

# Combine output and error files into a single file
#$ -j y

# Request 4 core
#$ -pe omp 4

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.6

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "WORKING DIR: $TMPDIR"
echo "Job ID : $JOB_ID"
echo "=========================================================="

MODEL="Qwen/Qwen3-8B-FP8"
PORT=8000
CONFIG="configs/vllm_config.ini"
SIMULATE="false" # "false" or "true"

MODEL_SAFE=$(echo "$MODEL" | tr '[:upper:]/' '[:lower:]_' )
LOG_DIR="logs/${MODEL_SAFE}"
SERVER_URL="http://localhost:${PORT}/v1"

module load gcc/12.2.0 python3/3.10.12 cuda/12.8

set -a
source .env
set +a

source "${ENV_PATH}/bin/activate"
hf auth login --token "$HF_TOKEN" --add-to-git-credential
mkdir -p "$LOG_DIR"
nohup vllm serve "$MODEL" \
        --reasoning-parser deepseek_r1 \
        --tensor-parallel-size 1 \
        --kv-cache-dtype fp8 \
        --enable-chunked-prefill \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 256 \
        --port $PORT \
        > "$LOG_DIR/vllm_$JOB_ID.log" 2>&1 &

echo "Starting vLLM for $MODEL (logs in $LOG_DIR)..."

# Wait loop
until curl -s $SERVER_URL/models | grep -q "id"; do
    echo "Waiting for model to load... Retrying in 1 minute"
    sleep 60
done

echo "vLLM is ready!"

if [[ "$SIMULATE" == "true" ]]; then
    echo "Running synthetic simulation..."
    python -m simulation.synthetic_simulation_run -c "$CONFIG"
else
    echo "Running simulation based on real data..."
    python -m simulation.simulation_run --model $MODEL \
                                        --server_url $SERVER_URL \
                                        -c "$CONFIG"
fi