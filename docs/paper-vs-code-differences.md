# Differences between the paper (WocaR-MARL section) and the current code

Line numbers refer to the current `my-main` branch and will shift once the code changes.

## Differences that change the method

1. **The vanilla Q-network is missing.** The paper uses four networks: vanilla $Q_v$, its target $Q_{v'}$, robust $Q_r$, and the worst-attack critic $\underline{Q}_\phi$. The code has only three: $Q_r$, a target copy of $Q_r$, and $\underline{Q}_\phi$ (`src/marl/woca_r_rl/dqn.py`, `_build_trainer`, lines 616-618). The vanilla network and its training step are therefore not implemented.

2. **The robust target $y^r$ is computed differently.** In the paper, the target mixes the *values* of $Q_{v'}$ and $\underline{Q}_\phi$, weighted by $\kappa_{\text{wst}}$. In the code, $\underline{Q}_\phi$ is only used to *choose* the next action, through a fixed mix (`worst_ratio = 0.1`, `scripts/train.py`, `build_parser`, line 85). The target value itself then comes from the target copy of $Q_r$ (`dqn.py`, `WocaRDQNTrainer._robust_targets`, line 203).

3. **$\kappa_{\text{wst}}$ has a different role.** In the paper, $\kappa_{\text{wst}}$ is the weight of the worst-case value inside the target. In the code, it weights two *losses* against each other, the normal TD loss and an extra bound loss (`dqn.py`, `WocaRDQNTrainer._update_robust_q`, line 195). The TD weight goes from 1 down to 0.1 during training (`scripts/train.py`, `build_parser`, line 89).

4. **The code has an extra loss term.** The robust network's loss includes a bound loss (`dqn.py`, `WocaRDQNTrainer._worst_case_loss`, line 212) that is not part of the paper's training objective.

5. **The state importance weight $w$ is computed from the wrong network.** The paper computes $w$ from the vanilla network $Q_v$. The code computes it from $Q_r$ (`dqn.py`, `WocaRDQNTrainer._state_importance`, line 235), because $Q_v$ doesn't exist (see point 1). The normalization, dividing by the largest value in the minibatch, matches the paper.

6. **By default, the adversarial action set uses almost no perturbation.** The paper computes this set with the scheduled perturbation radius $\epsilon_t$. The code's default radius is `1e-7`, copied from the official WocaR code (`scripts/train.py`, `build_parser`, line 90), so the set almost always contains only the agent's normal action. The paper's behavior is only used with the flag `--reachable-eps-scheduled` (`scripts/train.py`, `build_parser`, line 91).

7. **The rewards differ.** The paper gives each vehicle its own reward. The environment currently gives every vehicle the same reward: the average over all controlled vehicles (`src/marl/env.py`, `TeamHighwayEnv._reward`, line 15).

## Smaller differences

8. **The worst-attack critic is updated less often.** The code updates it only once every 100 updates (`dqn.py`, `WocaRDQNTrainer._maybe_update_worst_attack_critic`, line 246; default in `scripts/train.py`, `build_parser`, line 86). The paper updates it at every step.

9. **The update order is reversed.** The code updates $Q_r$ first and $\underline{Q}_\phi$ second (`dqn.py`, `WocaRDQNTrainer.update`, lines 160-163). The paper does it the other way round.

10. **The target network is updated differently.** The code copies the network every 2000 updates, a hard update (`dqn.py`, `WocaRDQNTrainer._sync_target`, line 265; default in `scripts/train.py`, `build_parser`, line 84). The paper uses a soft update with rate $\tau$.

11. **The TD loss function differs.** The code uses the Huber loss by default (`dqn.py`, `_build_trainer`, line 641). The paper uses the squared error.

12. **The agent ID is never perturbed (paper fix, not code).** The code only perturbs the observation, not the one-hot agent ID attached to it (`dqn.py`, `_build_trainer`, line 613). The paper's threat model doesn't mention this yet.

## What already matches

The hinge-form regularizer with margin $c = 1$, the IBP bounds, the shared replay buffer, parameter sharing, the one-hot agent ID, the episode ending when any controlled vehicle crashes, and the over-approximated adversarial action set.
