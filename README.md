# abm_political_polarization
Measuring Political Polarization using LLM-Based Agent Based Simulation

## Setup Instructions

1. Clone the repository with `git clone`
2. Create a virtual environment `python3 -m venv venv` and activate it.
3. Install the required packages using pip
4. Create a .env by copying the .env.example file and filling in the required values
5. Modified the config.ini accordingly
6. If using local model, you can modified the `scripts/qwen_server.sh` to start a vLLM server
7. run the real data simulation using `python -m simulation.simulation_real_data`