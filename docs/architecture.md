# Project architecture

## RL and agents

In RL, an agent observes an environment, chooses an action, and learns from the reward. In MARL, multiple agents act in the same environment. Here, controlled vehicles are agents; background traffic is environment-controlled. MAPPO and WocaR-DQN share model weights across vehicles and append a one-hot agent ID to each vehicle's observation.

## Training and execution

- **MAPPO — CTDE:** `src/marl/training/trainer.py:train` creates the environment through `src/marl/env.py:make_env`. The shared actor uses each vehicle's observation and ID. The critic receives concatenated observations for all vehicles during training; execution uses the actor's local input.
- **WocaR-DQN — CTDE with parameter sharing:** `src/marl/woca_r_rl/dqn.py:train_dqn` selects Highway or Intersection through `make_env`, pools each vehicle's transitions in one replay buffer, and updates shared Q-networks. Each action uses only that vehicle's observation and ID; there is no joint-state critic.
- **WocaR-PPO — not applicable to MARL categories in this implementation:** `src/marl/woca_r_rl/ppo.py:train_ppo` creates one Gym environment with one observation and action stream. It does not use `--agents` or a multi-agent environment.

CTDE means centralized training with decentralized execution. The project uses CTDE for MAPPO because of its joint-observation critic, and for WocaR-DQN because one learner trains on pooled agent transitions while action selection stays local. No current multi-agent path uses CTCE (centralized action selection) or DTDE (independent local learners).

---
See [training details](training-details.md) for observation/action shapes, rewards, schedules, and implementation limits.
