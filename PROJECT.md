# Project objective

This document records the intended research direction. The current implementation
is a starting point and does not define or necessarily fulfil that objective.

## Confirmed direction

- Extend WocaR-RL (Worst-case-aware Robust Reinforcement Learning) to a
  multi-agent reinforcement learning (MARL) setting.
- Use `highway-env` as the experimental environment.
- Use a shared team reward for the controlled agents.
- Run experiments to evaluate the multi-agent extension. The specific hypotheses,
  comparisons, metrics, and experiment budget remain to be decided.

The reference paper is [Efficient Adversarial Training without Attacking:
Worst-Case-Aware Robust Reinforcement Learning](https://arxiv.org/abs/2210.05927)
(Liang et al., NeurIPS 2022). WocaR-RL targets robustness to bounded adversarial
observation perturbations by estimating and optimising worst-case policy return
without training a separate attacker. How to extend this approach to multiple
agents is the research problem, rather than an already settled design.

## Tentative direction

- During robust training, account for possible observation perturbations to all
  controlled agents. This is the current working assumption, not a confirmed
  training design. It does not require explicitly running an attacker during
  training; the mechanism depends on the WocaR-RL extension.
- We **may** investigate and compare different Bellman operators.
  This is an optional research direction, not a committed requirement.
  No particular operators or comparison protocol have been selected.

## Planned evaluation scenarios

Evaluate robustness in separate experiments where observation attacks affect:

1. One controlled agent.
2. A subset of controlled agents.
3. All controlled agents simultaneously.

These are evaluation scenarios, not a requirement to train a separate policy for
each scenario. Agent selection, subset sizes, attack methods, and budgets remain
open. With very few agents, the one-agent and subset cases may overlap; specify
the attacked-agent count explicitly in each experiment.

## Current starting point

The repository currently implements a MAPPO baseline with a parameter-shared
actor, agent IDs, and a centralised critic. It does not yet implement the WocaR-RL
extension.

The shared scalar reward is the mean of the existing per-vehicle `highway-v0`
driving reward across all controlled vehicles. The episode terminates when any
controlled vehicle crashes (or leaves the road, if `offroad_terminal` is enabled).

These are implementation facts, not permanent research constraints. See
`README.md` for setup and execution instructions.

## Open decisions

- **Research question:** What specific claim or hypothesis should the MARL
  extension test?
- **Threat model:** Confirm the tentative all-agent training assumption and define
  which observation features can be perturbed, the perturbation budgets, and
  whether attacks can coordinate across agents. For evaluation, decide how agents
  and subset sizes are selected in the planned scenarios.
- **Method:** Which parts of WocaR-RL should be adapted, and should MAPPO remain
  the underlying learning algorithm?
- **Evaluation:** Which baselines, attack methods, metrics, random seeds, and
  training budgets should be used? How should nominal performance, robustness,
  and computational cost be assessed?
- **Optional Bellman study:** Is this needed, and if so, which operators and
  hypotheses should be compared?

## Guidance for contributors and coding agents

- Read this document before proposing or changing algorithms, rewards, or
  evaluation criteria.
- Distinguish confirmed objectives, tentative ideas, and current implementation.
- Do not treat existing code or an assistant's recommendation as a confirmed
  research decision.
- Keep unresolved choices explicit; record agreed decisions here as the project
  develops.
