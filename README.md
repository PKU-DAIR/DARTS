# [ICML'26] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning



## Abstract

Reinforcement Learning (RL) has become pivotal for improving model capabilities yet suffers from rollout efficiency bottlenecks due to the long-tail response length distribution.
While existing works mitigate the impact of long tails via prompt-level tail scheduling, we focus on the root source of inefficiency: the distribution itself.
Specifically, we characterize the long-tail distribution at a finer granularity, identifying intra-prompt long tails, and revealing that they frequently consist of ineffective verbosity.
To address this, we propose a novel paradigm of active distribution shaping to shape the rollout distribution towards conciseness and certainty, thereby fundamentally resolving tail-induced overheads.
We achieve this through a distribution-aware trajectory sampling mechanism, which selects trajectories from a redundant exploration space for each prompt, and an adaptive redundancy allocation scheme to maximize both shaping effectiveness and system efficiency.
Experiments demonstrate significant acceleration over state-of-the-art systems by up to 1.77x without compromising model performance.

## Requirements

- The code is based on VeRL. Please install dependencies as described in [VeRL](https://github.com/volcengine/verl?tab=readme-ov-file#contribution-guide).
- Environment configuration is provided in `env.yml`.

## Code Structure

- `recipe/darts/` : Main training logic, recipes, and experiment scripts. Includes baseline and overlap methods.
- `verl/` : Core VeRL code, including distributed training, model utilities, and worker implementations. Some files are modified for DARTS.

- `verl/workers/actor/dp_actor.py` : Modified for token-level overlap support.
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py` : Implements Async LLM Engine for token-level response streaming and forward computation during generation.
- `verl/workers/rollout/vllm_rollout/ray_trainer_darts.py` : Redundant rollout logic, including repeat number control and Ray actor synchronization.

## Usage

1. Prepare environment and install dependencies.
2. Configure training recipes in `recipe/darts/`. Replace model and datasets path.
3. Run training scripts as described in the recipe folder.


## Notes

- Most core RL and distributed logic is inherited from VeRL. Only key files are modified for DARTS-specific features.
- For details on token-level overlap and redundant rollout, see comments in the relevant Python files.

## Citation

If you use this codebase, please cite our paper.

