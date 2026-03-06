# abm_political_polarization
Measuring Political Polarization using LLM-Based Agent Based Simulation

## Environment

CUDA version: 12.8
Compute capability: 8.6
gcc: 12.2.0
Python: 3.10.12

## Setup Instructions

1. Clone the repository with `git clone`
2. Create a virtual environment `python3 -m venv venv` and activate it.
3. Install the required packages using pip `pip install -r requirements.txt`
4. Create a .env by copying the .env.example file and filling in the required values
5. Modified the config.ini accordingly
6. run the simulation with the choosen model accordingly `scripts/qwen_run.sh`