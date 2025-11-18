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
#$ -N abm_pp_qwen3

# Combine output and error files into a single file
#$ -j y

# Request 2 core
#$ -pe omp 2

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

source /projectnb/llamagrp/izzan/env/bin/activate
nohup vllm serve Qwen/Qwen3-8B --reasoning-parser deepseek_r1 --tensor-parallel-size 2 > logs/vllm.log 2>&1 &

echo "Starting vLLM... waiting for server to become ready."

# Wait loop
until curl -s http://localhost:8000/v1/models | grep -q "id"; do
    echo "Waiting for model to load... Retrying in 10s"
    sleep 10
done

echo "vLLM is ready!"

python -m simulation.simulation_real_data