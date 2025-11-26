#!/bin/bash -l

# Set SCC project
#$ -P llamagrp

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=120:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N abm_pp_qwen3

# Combine output and error files into a single file
#$ -j y

# Request 4 core
#$ -pe omp 4

# Request 2 GPU 
#$ -l gpus=2

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "WORKING DIR: $TMPDIR"
echo "Job ID : $JOB_ID"
echo "=========================================================="



module load cuda/12.2 gcc/12.2.0 python3/3.10.12

set -a
source .env
set +a

MODEL="Qwen/Qwen3-8B"
MODEL_SAFE=$(echo "$MODEL" | tr '[:upper:]/' '[:lower:]_' )
LOG_DIR="logs/${MODEL_SAFE}"

mkdir -p "$LOG_DIR"

source /projectnb/llamagrp/izzan/env/bin/activate
nohup vllm serve "$MODEL" \
    --reasoning-parser deepseek_r1 \
    --tensor-parallel-size 2 \
    > "$LOG_DIR/vllm.log" 2>&1 &

echo "Starting vLLM for $MODEL (logs in $LOG_DIR)..."

# Wait loop
until curl -s http://localhost:8000/v1/models | grep -q "id"; do
    echo "Waiting for model to load... Retrying in 30s"
    sleep 30
done

echo "vLLM is ready!"

python -m simulation.simulation_run -c configs/qwen_config.ini