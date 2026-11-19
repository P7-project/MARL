# Paper context: WocaR-MARL (P7, Aalborg University)

Summary of the paper-writing discussions so far: what has been decided, why,
what is drafted, and what is still open. For the codebase itself, see the
separate code context file and docs/architecture.md.

## The paper
- Title: "WocaR-MARL: Worst-case-aware Robust Multi-Agent Reinforcement Learning"
- P7 group project, M.Sc. Computer Science, Aalborg University. Supervisor: Yushuai Li.
- Written in Typst, one column, classic AAU report look (custom `template.typ`:
  AAU front page with fields + boxed abstract, Computer Modern font, chapter-style
  level-1 headings, roman page numbers for front matter).
- Planned structure: Introduction, Preliminaries, Related Work, Worst-case-aware
  Robust RL (original method), **WocaR-MARL** (our extension), Experiments,
  Limitations and Future Work, Conclusion, Appendix.

## Core idea (in one sentence)
WocaR-MARL takes WocaR-RL's idea of training against a *calculated* worst case
(instead of a real attacker) and applies it to a team of cooperating agents.

## Environments (context for the writing)
- **Highway** (custom `TeamHighwayEnv`): 5 controlled cars (may scale up) plus
  30 cars driven by the environment. Shared team reward = average of per-car
  rewards; the episode ends when any controlled car crashes. Each car observes
  the 5 nearest vehicles x (presence, x, y, vx, vy), relative = 25 numbers.
  5 discrete actions (lane left, idle, lane right, faster, slower).
- **Intersection** (`intersection-multi-agent-v2`, default config): 3 speed-only
  actions; per-agent rewards and individual termination. Does not yet match the
  cooperative common-reward setup described in the paper -> check.
- A one-hot agent ID is appended to each observation; its length depends on the
  number of agents, which blocks "train with 5, test with 3 or 8".
- Environment description belongs in the Experiments section (subsection
  "Environment"), unless the environment is a contribution of its own.

## Decisions and why

### Threat model
- The adversary only perturbs the controlled cars' observations, within an
  epsilon-ball, at every time step. It cannot change the true state.
  Environment-controlled cars are part of the environment, not attacked.
- No attacker exists during training: WocaR considers every action a
  perturbation within epsilon could cause and assumes the worst. This
  corresponds to an attacker with full knowledge of the policy.
- **Open: which agents are attacked.** Two scenarios are discussed in the draft:
  - one car attacked (hacked car, faulty sensor); since one crash ends the
    episode for the team, this is already a serious case;
  - all cars attacked (system-wide attack; also covers e.g. bad weather, since
    random noise is usually less harmful than a worst-case perturbation).
  - Attacking all agents makes the joint adversarial set grow exponentially.

### Training paradigm
- WocaR-MARL (DQN) uses **independent learning with parameter sharing**, not CTDE:
  one shared set of networks, each car acts on its own observation + ID, all
  cars' transitions go into one shared replay buffer, worst case computed per car.
- Why: the per-agent worst case scales **linearly** with the number of agents
  (5+5+5+5+5), while a joint worst case is **exponential** (5^5 = 3125 joint
  actions with 5 cars, 5^8 = 390625 with 8). Parameter sharing has separate
  reasons: fewer parameters, more data per step.
- Limitation: a car's worst case ignores teammates being attacked at the same
  time. It fits the one-car scenario; for all-cars attacked it gives no guarantee
  (each car is still robust to its own perturbations). Plan: evaluate both
  scenarios to measure this, and discuss a centralized worst-case critic (CTDE)
  as future work.
- Joint critic (CTDE) trade-off: captures the joint worst case and matches the
  MAPPO baseline's setup, but exponential cost, fixed team size, harder to learn,
  and would need value factorization (VDN/QMIX) for decentralized execution.
- Exam defense: state it clearly, justify by scalability, name the limitation,
  test both scenarios, point to future work. Don't call it CTDE.

### Algorithms in the repo (for reference)
| Algorithm | Agents | Setup | Critic | Environment |
|---|---|---|---|---|
| WocaR-DQN (our WocaR-MARL) | multi | independent learning + parameter sharing | worst-case critic per agent | highway, intersection |
| MAPPO (baseline) | multi | CTDE, shared actor + centralized critic | normal centralized critic | highway |
| WocaR-PPO | single | single agent | worst-case critic + value | MuJoCo (Hopper) |

### Worst-attack value estimation (per agent)
- Adversarial set: actions car i can be pushed into by perturbing its own
  observation o_i; computed in practice with IBP bounds on Q_r (over-approximation,
  on the safe side).
- Operator: (T Q)(o_i, a_i) = E[ r + gamma * min over a'_i in A_adv(o'_i, pi) of Q(o'_i, a'_i) ].
- R and P are dropped because they are defined on the full state and joint
  action; car i only receives r and o'_i. Other agents are part of the
  environment (like background traffic) and affect r and o'_i implicitly.
- Caveat: teammates keep learning (non-stationarity), so WocaR-RL's convergence
  guarantee does not strictly carry over. Do not claim it does.
- The joint version (product of per-agent sets, min over joint actions) is
  written out for reference only, to justify the per-agent choice.

### Worst-case-aware policy optimization (DQN)
- DQN has no policy network; the policy is greedy w.r.t. a Q-network.
- Four networks, all shared by the agents: Q_v (vanilla) + target Q_v',
  Q_r (robust, acts, only network kept at execution), Q_phi (worst-attack critic).
- Target for Q_r (corrected, kappa on the worst case):
  y^r = r + gamma * max_a' [ (1 - kappa_wst) Q_v'(o'_i, a') + kappa_wst Q_phi(o'_i, a') ]
- Schedule (from the WocaR paper, App. D.1.2, and our code): kappa_wst = 0 for
  the first third of training, then increased gradually to 0.5.
- Worst-attack critic != attacker: the critic only estimates values; it never
  perturbs observations.

### Value-enhanced state regularization (draft)
- Weight from the vanilla network: w(o_i) = normalized (max_a Q_v - min_a Q_v).
- Regularizer on Q_r (hinge form): penalize perturbations under which another
  action would beat the action chosen on the true observation; margin c.
- Applied to all agents, since it is unknown which will be attacked.
- Cite WocaR (not SA-DQN by name).

## Notes on the original WocaR paper (Algorithm 2, WocaR-DQN)
- Reproduced in the appendix with the Typst `algorithmic` package (1.0.7),
  as published, with a note.
- Index typos: lines 12, 13, 17 (our numbering, Input line = 1) use t where the
  minibatch index i is meant.
- The two terms in y^r (line 18) are swapped: the kappa_wst schedule starting
  at 0 only makes sense if kappa_wst weights the worst case.
- The official code differs from the published algorithm: 3 networks instead of
  4, the worst-attack critic only selects the next action (value from the target
  network), an extra bound loss, w(s) not found, Q_phi updated every few steps.
  Our implementation follows the published algorithm.

## Notation conventions
- o_i = local observation of agent i (never s_i); s = global state only.
- A_i = action space of agent i (`Actions_i`); bold a = joint action;
  A in the MARL part means the joint action space.
- o'_i = next observation (Typst: `o'_i`, not `o_i '`); r = shared team reward.
- pi = the shared policy (no subscript).
- Typst macros in use: WstBellmanOpr, AdvActions, PerturbSet, policy, Actions,
  TransitionMod, RewardFunc, DiscountFac. Bib key for WocaR: `src:wocar-rl`.

## Plan / remaining work
1. Finish Value-enhanced State Regularization.
2. Combined training objective (may be merged into the regularization part,
   since in DQN it is just the full Q_r loss).
3. WocaR-MARL pseudocode at the end of the WocaR-MARL section (main text),
   showing the differences from Algorithm 2.
4. Experiments: environment, attacks (PGD, MinBest, PA-AD), baselines,
   both attack scenarios, implementation details.
5. Limitations: joint worst case, non-stationarity, centralized critic as future work.

## Open TODOs
- Check in dqn.py which term kappa_wst multiplies (must be Q_phi).
- Check which regularizer SARegularizer uses (hinge or cross-entropy, margin c).
- Check how w(o_i) is normalized in dqn.py and state it.
- Highway and intersection have different action spaces: revisit the
  "identical for all agents" wording when writing the environment section.
- Decide which agents are attacked (threat model).
- Labels/refs to add: sec:wocar-rl, eq:worst-attack-bellman, eq:wst-loss,
  sec:experiments, sec:limitations.

## How to work with me
- One step at a time; discuss before producing anything.
- Do not create things I did not ask for (e.g. whole-section drafts).
- I am still learning: explain the reasoning.
- The paper chat is for writing/formatting; code is only context.
