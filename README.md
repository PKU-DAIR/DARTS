
# DARTS Code Structure

This repository contains the full codebase for DARTS. Below is a guide to the main code components, modules, and usage instructions.

## Requirements

- The code is based on VeRL. Please install dependencies as described in [VeRL](https://github.com/volcengine/verl?tab=readme-ov-file#contribution-guide).
- Environment configuration is provided in `env.yml`.

## Directory Overview

- `recipe/darts/` : Main training logic, recipes, and experiment scripts. Includes baseline and overlap methods.
- `verl/` : Core VeRL code, including distributed training, model utilities, and worker implementations. Some files are modified for DARTS.

## Key Modules & Scripts

- `verl/workers/actor/dp_actor.py` : Modified for token-level overlap support.
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py` : Implements Async LLM Engine for token-level response streaming and forward computation during generation.
- `verl/workers/rollout/vllm_rollout/ray_trainer_darts.py` : Redundant rollout logic, including repeat number control and Ray actor synchronization.

## Usage

1. Prepare environment and install dependencies.
2. Configure training recipes in `recipe/darts/`.
3. Run training scripts as described in the recipe folder.


## Notes

- Most core RL and distributed logic is inherited from VeRL. Only key files are modified for DARTS-specific features.
- For details on token-level overlap and redundant rollout, see comments in the relevant Python files.

## Citation

If you use this codebase, please cite our paper.

