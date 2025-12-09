#!/bin/bash -l

# Set SCC project
#$ -P llamagrp

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=60:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N abm_pp_llama_nemotron

# Combine output and error files into a single file
#$ -j y

# Request 4 core
#$ -pe omp 4

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "WORKING DIR: $TMPDIR"
echo "Job ID : $JOB_ID"
echo "=========================================================="

MODEL="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
CONFIG="configs/llama_config.ini"
SIMULATE="true" # "false" or "true"

MODEL_SAFE=$(echo "$MODEL" | tr '[:upper:]/' '[:lower:]_' )
LOG_DIR="logs/${MODEL_SAFE}"

module load cuda/12.2 gcc/12.2.0 python3/3.10.12

set -a
source .env
set +a

source /projectnb/llamagrp/izzan/env/bin/activate
hf auth login --token "$HF_TOKEN" --add-to-git-credential
mkdir -p "$LOG_DIR"
nohup vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    > "$LOG_DIR/vllm_$JOB_ID.log" 2>&1 &

echo "Starting vLLM for $MODEL (logs in $LOG_DIR)..."

# Wait loop
until curl -s http://localhost:8000/v1/models | grep -q "id"; do
    echo "Waiting for model to load... Retrying in 1 minute"
    sleep 60
done

echo "vLLM is ready!"

if [[ "$SIMULATE" == "true" ]]; then
    echo "Running synthetic simulation..."
    python -m simulation.synthetic_simulation_run -c "$CONFIG"
else
    echo "Running simulation based on real data..."
    python -m simulation.simulation_run -c "$CONFIG"
fi