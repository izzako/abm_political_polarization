#!/bin/bash -l

# Set SCC project
#$ -P llamagrp

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=80:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N abm_pp_gpt5_nano

# Combine output and error files into a single file
#$ -j y

# Request 4 core
#$ -pe omp 4


# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "WORKING DIR: $TMPDIR"
echo "Job ID : $JOB_ID"
echo "=========================================================="

CONFIG="configs/gpt_config.ini"
SIMULATE="true" # "false" or "true"

module load gcc/12.2.0 python3/3.10.12 cuda/12.8

set -a
source .env
set +a

source "${ENV_PATH}/bin/activate"
if [[ "$SIMULATE" == "true" ]]; then
    echo "Running synthetic simulation..."
    python -m simulation.synthetic_simulation_run -c "$CONFIG" --debug
else
    echo "Running simulation based on real data..."
    python -m simulation.simulation_run -c "$CONFIG" --debug
fi