#!/bin/bash
set -a
source .env
set +a

source /projectnb/llamagrp/izzan/env/bin/activate
vllm serve Qwen/Qwen3-8B --reasoning-parser deepseek_r1