
    

from typing import Protocol

from torch import Tensor, nn
import torch

"""
High level overview:
Math	                            Code
(T_worst^pi Q)(s,a)	                compute_targets
E_{s'~P(s,a)}	                    replay-buffer sample next_states
R(s,a)	                            rewards
gamma	                            _discount_factor
min_{a' in A_adv(s',pi)} Q(s',a')   _worst_case_values
A_adv(s,pi)	                        possible_actions_mask
B_eps(s)	                        PerturbationLpNorm(inf, eps) / _sample_ball
Q (worst-attack critic)	            target_q_network
"""

class ActionSetEstimator(Protocol):
    def possible_actions_mask(self, states: Tensor) -> Tensor:
        """Bool tensor [batch, num_actions]: True where a in A_adv(s, pi).

        Math: indicator of the set
            A_adv(s, pi) := {a in A : exists s~ in B_eps(s) s.t. pi(s~) = a}
        """
        ...

class WorstAttackBellmanOperator:
    """
        Sample-based estimate of the worst-attack Bellman operator (Definition 4.1).

        (T_worst^pi Q)(s, a) := E_{s' ~ P(s, a)} [ R(s, a) + gamma * min_{a' in A_adv(s', pi)} Q(s', a') ]

        Fixed point of T_worst^pi: the worst-attack value Q_worst^pi, i.e. the
        expected return of pi when an adversary may perturb every observation
        within B_eps and picks the perturbation that minimizes the future value.
        Learning Q_worst means regressing a network onto the targets below.
    """

    def __init__(
        self,
        target_q_network: nn.Module,
        action_set_estimator: ActionSetEstimator,
        discount_factor: float,
    ) -> None:
        # Q in the definition: target copy of the worst-attack critic Q_worst.
        self._target_q_network = target_q_network
        # Provides A_adv(s', pi); encapsulates both pi and B_eps.
        self._action_set_estimator = action_set_estimator
        # gamma in the definition.
        self._discount_factor = discount_factor

    @torch.no_grad()
    def compute_targets(
        self,
        rewards: Tensor,
        next_states: Tensor,
        terminated: Tensor,
    ) -> Tensor:
        """
            Single-sample estimate of (T_worst^pi Q)(s, a) for a batch of (s, a, r, s').

            Math: R(s, a) + gamma * min_{a' in A_adv(s', pi)} Q(s', a')

            - The expectation E_{s' ~ P(s, a)} is replaced by the one sampled s'
            from the replay buffer. Because the min sits inside the expectation,
            this single-sample target is an unbiased estimate of the operator.
            - R(s, a) does not depend on s' or a', so it stays outside the min.
            - `terminated` is 1.0 on true terminal transitions (value 0 afterwards)
            and must exclude time-limit truncation, which does not end the MDP. MDP time limit truncation refers to the practice of ending a reinforcement learning episode after a fixed number of timesteps, which breaks the standard Markov Decision Process (MDP) framework if not handled correctly.
        """
        worst_case_values = self._worst_case_values(next_states)
        not_terminated = 1.0 - terminated
        return rewards + self._discount_factor * not_terminated * worst_case_values

    def _worst_case_values(self, next_states: Tensor) -> Tensor:
        """
            Math: min_{a' in A_adv(s', pi)} Q(s', a'), for each s' in the batch.

            The set is encoded as a boolean mask. Actions outside the set get +inf,
            so they can never be the minimizer.
        """
        possible_actions = self._action_set_estimator.possible_actions_mask(next_states)
        q_values = self._target_q_network(next_states)
        impossible_actions = torch.logical_not(possible_actions)
        masked_q_values = q_values.masked_fill(impossible_actions, float("inf"))

        return masked_q_values.min(dim=1).values

class CertifiedActionSetEstimator:
    """
        Over-approximates A_adv(s, pi) with CROWN bounds on pairwise logit differences.

        Math: for each pair (i, j), compute a lower bound
            L_ij(s) <= min_{s~ in B_eps(s)} [ logit_j(s~) - logit_i(s~) ].
        If L_ij(s) > 0 for some j, then logit_j > logit_i everywhere in B_eps(s),
        so argmax_a logit_a(s~) != i for every s~ in B_eps(s), hence i not in A_adv(s, pi).

        Soundness: the returned set contains A_adv (it only removes actions that are
        provably unreachable), so min over it is <= min over the true A_adv.
        The target therefore lower-bounds the true worst-attack Bellman target:
        conservative, never optimistic.

        The policy is pi(s) = argmax_a logit_a(s), the greedy policy over the network's outputs (for DQN these are the Q-values).
    """

    def __init__(
        self,
        policy_network: nn.Module,
        epsilon: float,
        num_actions: int,
        example_state: Tensor,
    ) -> None:
        # Imported here so the sampling estimator works without auto_LiRPA.
        from auto_LiRPA.auto_LiRPA import BoundedModule
        
        # epsilon: radius of the L-inf ball B_eps(s).
        self._epsilon = epsilon
        self._num_actions = num_actions
        self._bounded_policy = BoundedModule(policy_network, torch.empty_like(example_state))

    @torch.no_grad()
    def possible_actions_mask(self, states: Tensor) -> Tensor:
        """
            Math: indicator of the over-approximation of A_adv(s, pi).

            Action i is excluded iff exists j with L_ij(s) > 0.
        """
        lower_bounds = self._pairwise_logit_difference_lower_bounds(states)
        provably_excluded = (lower_bounds > 0).any(dim=2)
        return torch.logical_not(provably_excluded)

    def _pairwise_logit_difference_lower_bounds(self, states: Tensor) -> Tensor:
        """
            Returns [batch, A, A]; entry (b, i, j) is L_ij(s_b), a lower bound on
            min_{s~ in B_eps(s_b)} (logit_j(s~) - logit_i(s~)).

            B_eps is the L-inf ball: {s~ : ||s~ - s||_inf <= epsilon}.
        """
        from auto_LiRPA.auto_LiRPA import BoundedTensor
        from auto_LiRPA.auto_LiRPA.perturbations import PerturbationLpNorm

        perturbation = PerturbationLpNorm(norm=float("inf"), eps=self._epsilon) # type: ignore[arg-type]
        bounded_states = BoundedTensor(states, perturbation)
        specification = self._build_specification(states.shape[0], states.device)
        lower, _ = self._bounded_policy.compute_bounds(
            x=(bounded_states,), C=specification, method="CROWN"
        )
        return lower.view(states.shape[0], self._num_actions, self._num_actions)

    def _build_specification(self, batch_size: int, device: torch.device) -> Tensor:
        """
            Linear specification C so that C @ logits = logit_j - logit_i for each (i, j).

            Row (i, j) = e_j - e_i. This asks the bound-propagation method for
            bounds on the difference directly, which is tighter than bounding
            each logit separately and subtracting.
        """
        identity = torch.eye(self._num_actions, device=device)
        # Row (i, j) = e_j - e_i, giving logit_j - logit_i.
        differences = identity.unsqueeze(0) - identity.unsqueeze(1)
        flattened = differences.view(self._num_actions**2, self._num_actions)
        return flattened.unsqueeze(0).expand(batch_size, -1, -1)


class SampledActionSetEstimator:
    """
        Under-approximates A_adv(s, pi) by sampling B_eps(s). Optimistic; sanity checks only.

        Math: returns {pi(s~_k)} for finitely many s~_k in B_eps(s), a subset of
        A_adv(s, pi). The min over a subset is >= the min over the full set, so
        the resulting target can exceed the true worst-attack target. It does not
        certify robustness and must not be used to train Q_worst.
    """

    def __init__(
        self,
        policy_network: nn.Module,
        epsilon: float,
        num_actions: int,
        num_samples: int,
    ) -> None:
        self._policy_network = policy_network
        self._epsilon = epsilon
        self._num_actions = num_actions
        self._num_samples = num_samples

    @torch.no_grad()
    def possible_actions_mask(self, states: Tensor) -> Tensor:
        """
            Math: indicator of {pi(s~_k) : k = 0..K}, with s~_0 = s (the clean state).
        """

        batch_size = states.shape[0]
        perturbed_states = self._sample_ball(states)
        flat_states = perturbed_states.reshape(-1, *states.shape[1:])
        # pi(s~) = argmax_a logit_a(s~)
        chosen_actions = self._policy_network(flat_states).argmax(dim=1)
        chosen_actions = chosen_actions.view(batch_size, self._num_samples + 1)
        one_hot = nn.functional.one_hot(chosen_actions, self._num_actions).bool()
        return one_hot.any(dim=1)

    def _sample_ball(self, states: Tensor) -> Tensor:
        """
            Returns [batch, num_samples + 1, *state_shape]: s itself plus uniform
            samples s~ with ||s~ - s||_inf <= epsilon, i.e. points in B_eps(s).
        """
        noise_shape = (states.shape[0], self._num_samples, *states.shape[1:])
        noise = (torch.rand(noise_shape, device=states.device) * 2 - 1) * self._epsilon
        expanded = states.unsqueeze(1)
        perturbed = expanded + noise
        return torch.cat([expanded, perturbed], dim=1)