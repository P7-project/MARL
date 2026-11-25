# Attack Evaluation CLI

A single entry point for evaluating a trained WocaR-DQN model with
clean, random, or PGD observation attacks.

The evaluation mode is selected using `--mode`:

  -----------------------------------------------------------------------
  Value                               Description
  ----------------------------------- -----------------------------------
  `clean`                             Evaluates without observation
                                      attacks.

  `random`                            Adds bounded random noise to
                                      selected agent observations.

  `pgd`                               Applies projected-gradient-descent
                                      (PGD) perturbations to selected
                                      agent observations.
  -----------------------------------------------------------------------

## Examples

``` bash
# Clean evaluation
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode clean --episodes 100

# Randomly perturb Agent 0
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode random --target-mode fixed --target-agents 0 --epsilon 0.05 --episodes 100

# Randomly perturb Agents 0 and 2
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode random --target-mode fixed --target-agents 0 2 --epsilon 0.05 --episodes 100

# Random targeting: choose a non-empty subset of agents whenever an attack occurs
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode random --target-mode random --epsilon 0.05 --episodes 100

# Random targeting with a 50% attack probability at each timestep
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode random --target-mode random --epsilon 0.05 --attack-probability 0.5 --episodes 100

# PGD attack with default PGD settings
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode pgd --target-mode fixed --target-agents 0 --epsilon 0.05 --episodes 100

# PGD attack with custom iterations and step size
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode pgd --target-mode random --epsilon 0.05 --pgd-iters 10 --pgd-step-size 0.01 --episodes 100
```

## Arguments

### General evaluation

  ------------------------------------------------------------------------
  Argument                                   Default Meaning
  --------------------- ---------------------------- ---------------------
  `--checkpoint`                            Required Path to the trained
                                                     model checkpoint.

  `--agents`                                     `3` Number of agents;
                                                     must match the
                                                     trained model and
                                                     environment.

  `--mode`                                   `clean` Evaluation mode:
                                                     `clean`, `random`, or
                                                     `pgd`.

  `--episodes`                                  `10` Number of evaluation
                                                     episodes.

  `--seed`                                      `42` Random seed used for
                                                     evaluation and attack
                                                     randomness.
  ------------------------------------------------------------------------

### Observation attack

  ------------------------------------------------------------------------------
  Argument                                      Default Meaning
  ------------------------ ---------------------------- ------------------------
  `--epsilon`                                    `0.05` Maximum magnitude of
                                                        each observation
                                                        perturbation.

  `--attack-probability`                          `1.0` Probability of applying
                                                        an attack at each
                                                        timestep.

  `--target-mode`                               `fixed` Agent targeting
                                                        strategy: `fixed` or
                                                        `random`.

  `--target-agents`                              `None` Agent indices used in
                                                        fixed targeting;
                                                        required for attack
                                                        modes with
                                                        `--target-mode fixed`.
  ------------------------------------------------------------------------------

### PGD-specific

  -------------------------------------------------------------------------------
  Argument                                       Default Meaning
  ------------------------- ---------------------------- ------------------------
  `--pgd-iters`                                     `10` Number of PGD
                                                         iterations.

  `--pgd-step-size`                               `None` PGD update step size. If
                                                         omitted, the attacker
                                                         uses
                                                         `epsilon / pgd-iters`.

  `--pgd-random-start` /                          `True` Enable or disable random
  `--no-pgd-random-start`                                initialization within
                                                         the perturbation budget.
  -------------------------------------------------------------------------------

### Logging

  Argument           Default Meaning
  -------------- ----------- ------------------------------
  `--logdir`          `runs` TensorBoard root directory.
  `--run-name`     Timestamp Name of this evaluation run.

## Targeting

-   `fixed`: Attacks all agents specified with `--target-agents`
    whenever an attack is applied.
-   `random`: Randomly selects a non-empty subset of eligible agents
    whenever an attack is applied. Both the subset size and agent
    identities may change between attack timesteps.

If `--attack-probability` is less than `1.0`, an attack may be skipped
at a timestep. The selected agents are recorded for rendering when an
attack is applied.

## Rendering

The evaluator can display the environment during evaluation. Rendering
is enabled with `--render`.

  Argument               Default Meaning
  ------------------- ---------- ---------------------------------------------
  `--render`            Disabled Enables visual rendering during evaluation.
  `--render-agents`          `1` Number of agent camera views displayed.

The number of agents in the environment and the number of rendered
camera views are independent. For example, an environment can contain
three agents while displaying only two camera views.

Example:

``` bash
# Render two agent perspectives in an environment with three agents
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode random --episodes 5 --render --agents 3 --render-agents 2 --target-mode random --attack-probability 0.8
```

## Outputs

The evaluator prints the average episode reward and reward standard
deviation.

Episode rewards are logged to TensorBoard under:

``` text
runs/attack_evaluation/<run-name>
```

Start TensorBoard with:

``` bash
uv run tensorboard --logdir=runs
```

Open <http://localhost:6006> to compare evaluation runs.
