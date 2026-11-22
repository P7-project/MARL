# Paper context: WocaR-MARL (P7, Aalborg University)

Summary of the paper-writing discussions so far: what has been decided, why,
what is drafted, and what is still open. For where the code does not yet
match the paper, see docs/paper-vs-code-differences.md.

Last updated: 2026-11-22.

## The paper
- Title: "WocaR-MARL: Worst-case-aware Robust Multi-Agent Reinforcement Learning"
- P7 group project, M.Sc. Computer Science.
- Written in Typst, one column.
- Planned structure: Introduction, Preliminaries, Related Work, Worst-case-aware
  Robust RL (original method), **WocaR-MARL** (our extension), Experiments,
  Limitations and Future Work, Conclusion, Appendix.

## Core idea (in one sentence)
WocaR-MARL takes WocaR-RL's idea of training against a *calculated* worst case
(instead of a real attacker) and applies it to a team of cooperating agents.

## Status of the WocaR-MARL chapter
Drafted and agreed (section by section): Threat Model, Problem Setting,
Training Paradigm, Value Estimation, Policy Optimization, Regularization
(`<eq:regLoss>`, `<eq:marl-regLoss>`), Combined Objective (`<eq:combined-loss>`),
and the WocaR-MARL algorithm (`<alg:wocar-marl>`).
The user is now rewriting the paper from the beginning; open consistency
points are listed under "Open fix list" below.

## Environments (context for the writing)
- **Highway** (custom `TeamHighwayEnv`): 5 controlled cars (may scale up) plus
  30 cars driven by the environment. The episode ends when any controlled car
  crashes. Each car observes the 5 nearest vehicles x (presence, x, y, vx, vy),
  relative = 25 numbers. 5 discrete actions (lane left, idle, lane right,
  faster, slower).
  - Paper: each car gets its **own** reward r_i. Code: every car gets the team
    mean (`env.py`, `TeamHighwayEnv._reward`). Code to be changed later, then
    retrain (also check the MAPPO critic).
- **Intersection** (`intersection-multi-agent-v2`, default config): 3 speed-only
  actions; per-agent rewards and individual termination. Check that it matches
  the setup described in the paper.
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
- Motivation must come from real-world traffic, not from our environment design
  (reasoning backwards from the choice was rejected).
- Two scenarios: one car attacked (hacked car, faulty sensor) and all cars
  attacked (system-wide attack). Attacking all agents jointly makes the joint
  adversarial set grow exponentially.
- To add: one sentence saying the one-hot agent ID is never perturbed (the
  code only perturbs the observation part).

### Problem setting (reward model)
- Cooperative does not imply a shared reward. Decision: each car has its own
  reward r_i (each car wants its own reward high), but the episode ends for
  everyone when any controlled car crashes, so no car wants to cause a crash.
  This is motivated by the cars operating as a team.
- Formally a general-sum stochastic game (individual R_i(s, a, s')), not a
  common-reward game.
- Wording: "team", not "fleet".

### Training paradigm
- WocaR-MARL (DQN) uses **independent learning with parameter sharing**, not CTDE:
  one shared set of networks, each car acts on its own observation + ID, all
  cars' transitions go into one shared replay buffer, worst case computed per car.
- Why: the per-agent worst case scales **linearly** with the number of agents
  (5+5+5+5+5), while a joint worst case is **exponential** (5^5 = 3125 joint
  actions with 5 cars, 5^8 = 390625 with 8). Parameter sharing has separate
  reasons: fewer parameters, more data per step. In training the network is
  shared; at execution each car runs its own copy.
- Non-stationarity and credit assignment are **not** discussed here; they go to
  Limitations / Future Work (see below).
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
  observation o_i. Computed in practice with IBP bounds on Q_r, giving the
  over-approximation hat(A)_adv (hat = estimated superset, on the safe side).
- Operator: (T Q)(o_i, a_i) = E[ r_i + gamma * min over a'_i in A_adv(o'_i, pi) of Q(o'_i, a'_i) ].
- Other agents are part of the environment (like background traffic) and affect
  r_i and o'_i implicitly.
- The joint version (with Q_i, R_i(s, a, s') and AdvJointActions) is written
  out for reference only, to justify the per-agent choice.

### Worst-case-aware policy optimization (DQN)
- DQN has no policy network; the policy is greedy w.r.t. a Q-network.
- Four networks, all shared by the agents: Q_v (vanilla) + target Q_v',
  Q_r (robust, acts, only network kept at execution), underline(Q)_phi
  (worst-attack critic; underline = worst case, as in WocaR).
- Target for Q_r (kappa on the worst case):
  y^r = r_i + gamma * max_a' [ (1 - kappa_wst) Q_v'(o'_i, a') + kappa_wst underline(Q)_phi(o'_i, a') ]
- Schedule (WocaR App. D.1.2): kappa_wst = 0 at first, then increased
  gradually to 0.5.
- Worst-attack critic != attacker: the critic only estimates values; it never
  perturbs observations.

### Value-enhanced state regularization
- WocaR-DQN uses the SA-DQN regularizer (WocaR App. C.3; Alg. 2 line 17 in the
  PDF uses kappa_reg * w * L_reg). We write it in WocaR notation.
- Hinge form (SA-DQN Eq. 7), per minibatch sample j:
  max over perturbed o_hat in B(o_j) of max over a != a*_j of
  Q_r(o_hat, a) - Q_r(o_hat, a*_j), clipped below at -c. Margin c = 1.
  The 1/N averaging is inside L_reg, in both the single-agent and MARL versions.
- Weight from the vanilla network: w(o_j) = max_a Q_v - min_a Q_v, divided by
  the largest value in the minibatch. WocaR only says "by normalizing"; the
  max-normalization is our choice (keeps kappa_reg meaningful) and is stated
  as such.
- MARL adaptation: samples from the shared buffer are indexed by j (agent index
  i stays separate). Sampling is uniform with replacement, so a car can appear
  several times or not at all in a minibatch.
- Applied to all agents, since it is unknown which will be attacked.

### Combined objective and algorithm
- In DQN the combined objective is the full Q_r loss: TD loss to y^r plus
  kappa_reg * w * L_reg.
- The WocaR-MARL pseudocode is in the main text (`<alg:wocar-marl>`), with
  individual rewards r_{i,t}.

## Notes on the original WocaR paper (Algorithm 2, WocaR-DQN)
- Reproduced in the appendix with the Typst `algorithmic` package (1.0.7),
  as published, with a note.
- Our numbering counts the Input line as line 1 (one higher than the PDF).
- Index typos: lines 12, 13, 17 (our numbering) use t where the minibatch
  index is meant.
- The two terms in y^r (line 18) are swapped: the kappa_wst schedule starting
  at 0 only makes sense if kappa_wst weights the worst case.
- The official code (WocaR-DQN/SA-DQN/train.py) differs from the published
  algorithm: 3 networks instead of 4, the worst-attack critic only mixes into
  action selection through a fixed worst_ratio (value from the target
  network), kappa weights the TD loss against an extra bound loss, no w(s)
  computed, Q_phi updated every worst_rate steps. Hinge regularizer with c = 1.
  Our paper follows the published algorithm.

## Notation conventions
- o_i = local observation of agent i (never s_i); s = global state only.
- Indices: i = agent, j = minibatch sample, t = time; avoid k (WocaR's iteration).
  Minibatch size N, with explicit sum limits.
- A_i = action space of agent i (`Actions_i`); bold a = joint action;
  A in Preliminaries means the joint action space (clash with single-agent use,
  see fix list).
- o'_i = next observation (Typst: `o'_i`, not `o_i '`); r_i = individual reward.
- pi = the shared policy (no subscript). Replay buffer = cal(D) (cal(B) is the
  perturbation set).
- Always use the `preamble.typ` macros, never raw symbols. In use:
  WstBellmanOpr, AdvActions, AdvJointActions, PerturbSet, policy, Actions,
  TransitionMod, RewardFunc, DiscountFac, CriticParam (= phi),
  WorstWeight (= kappa_wst), RegWeight (= kappa_reg), RegLoss, StateWeight.
- Proposed, postponed: JointActions, EstAdvActions, WstCritic, WstValFunc,
  VanillaQ/TargetQ/RobustQ/ReplayBuffer, remove duplicate WstBellmanOpr.
- Bib keys: `src:wocar-rl`, `src:marl-fma`, SA-DQN (verify volume/pages).

## Open fix list (raise when the rewrite reaches each section)
1. Policy Optimization: the kappa_wst swap vs WocaR Alg. 2 is not explained;
   point to the appendix note.
2. WocaR-RL chapter text still says k_worst; use the kappa macros.
3. State regularization (WocaR-RL chapter): "normalized difference" but the
   equation shows no normalization.
4. Line references: "Algorithm 2, line 17" -> line 18 of `@alg:wocar-dqn`;
   "lines 9 and 15" -> lines 10 and 16.
5. Combined Objective: cite our own `@eq:combined_woca` instead of
   "Eq. 8 in @src:wocar-rl".
6. a in cal(A) used for a single agent's action, but Preliminaries define
   cal(A) as the joint action space (needs Preliminaries rewrite + JointActions).
7. WocaR-MARL algorithm says IBP, WocaR-RL chapter says "convex relaxations":
   add one linking sentence.
8. Every remaining "shared team reward r" must become r_i.
9. Worst-attack critic notation: use one form everywhere (Q_phi vs
   underline(Q)_phi^pi vs Q_(CriticParam)^policy).
10. Preliminaries: optimality operator must use max_a' Q*(s', a'); the
    expectation operator text says "state-value".
11. Appendix: add the numbering note, change the index paragraph i -> j, and
    list the extra official-code differences above.

## Plan / remaining work
1. Finish the rewrite pass from the beginning, applying the fix list.
2. Limitations / Future Work must cover: joint worst case not captured;
   non-stationarity (teammates change as the shared net learns, replay holds
   stale behaviour, WocaR's convergence guarantee does not strictly carry
   over); multi-agent credit assignment (episode ends at any crash, a car
   cannot tell if it caused it). Source: `@src:marl-fma`. Future work:
   penalize only the causing car, or a centralized (CTDE) critic.
3. Experiments: environment, attacks (PGD, MinBest, PA-AD), baselines,
   both attack scenarios, implementation details.
4. Code PR(s) to align the code with the paper (see
   docs/paper-vs-code-differences.md), then retrain.

## Open TODOs
- Unverified: the marl-fma book's definitions of cooperative / competitive /
  mixed; SA-DQN bib volume and pages.
- Highway and intersection have different action spaces: revisit the
  "identical for all agents" wording when writing the environment section.
- Labels/refs to add: sec:wocar-rl, eq:worst-attack-bellman, eq:wst-loss,
  sec:experiments, sec:limitations.

## How to work with me
- One step at a time; discuss before producing anything.
- Do not create things I did not ask for (e.g. whole-section drafts).
- I am still learning: explain the reasoning, in prose rather than lists. Only
  use lists when there are several separate items (e.g. causes, reasons,
  problems).
- Do not make things up: cite where each claim comes from.
- The paper chat is for writing and formatting; writing code is delegated to
  the coder subagent.
