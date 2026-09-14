# How Good Are LLM Agents? Benchmarking Opinion Dynamics Against Indonesian Electoral Discourse on Social Media

This repository contains the code and configuration for the paper *"How Good Are LLM Agents? Benchmarking Opinion Dynamics Against Indonesian Electoral Discourse on Social Media"*. We benchmark three open-weight LLMs (Gemma-3-12B, Qwen3-8B, Nemotron-Nano-8B) against Friedkin-Johnsen and per-agent Ridge regression on 1,000 Twitter users and 70,336 interactions from the 2024 Indonesian Presidential Election.

## Repository Structure

```
src/                  # Core modules (agents, opinion classifier, utilities)
simulation/           # Simulation runners (LLM-ABM, Ridge, Friedkin-Johnsen, grid search)
configs/              # Hyperparameter configs for FJ, Ridge, LLM, and vLLM serving
scripts/              # Shell scripts to launch each model's simulation
data/                 # Interaction data, author data, and ground-truth opinion trajectories
persona/              # Generated agent personas and summarized memories
prompts/              # Prompt templates for opinion classifier and memory summarizer
figures/              # Output figures (pipeline diagram, trend plots, teaser)
```

## Environment

- CUDA: 12.8
- Compute capability: 8.6
- GCC: 12.2.0
- Python: 3.10.12

## Setup

1. Clone the repository.
2. Create and activate a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in the required values.
5. Start the vLLM inference server with the desired model checkpoint.
6. Adjust the config file in `configs/` for your experiment.

## Running Simulations

Each model has a dedicated launch script in `scripts/`:

```bash
bash scripts/qwen_run.sh       # Qwen3-8B
bash scripts/gemma_run.sh      # Gemma-3-12B
bash scripts/llama_run.sh      # Nemotron-Nano-8B
bash scripts/fj_run.sh         # Friedkin-Johnsen baseline
```

Ridge regression is run via `simulation/regression_simulation.py`.

## Evaluation

Scoring (Wasserstein distance, macro/weighted F1) and analysis notebooks:

- `compute_score.ipynb` — main evaluation metrics and lag analysis
- `cross_topic_correlation.ipynb` — cross-topic Pearson correlation and Frobenius distance
- `visualization.ipynb` — figure generation
