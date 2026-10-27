# MAPPO (Multi-Agent PPO) for multi-agent highway-env.

## Design:
* Parameter-shared actor: one policy network used by all agents. A one-hot
  agent id is appended to each agent's observation so they can specialise.
* Centralised critic: V(s) takes the concatenation of ALL agents' observations.
* Team reward: highway-env returns one shared scalar reward, so GAE is computed
  at team level and the same advantage is used for every agent's policy loss.

## Install:
    `uv sync`

## Train:
    `uv run main.py --total-steps XXX --save ABC.pt`

## Evaluate:
    `uv run main.py --eval ABC.pt --render`
