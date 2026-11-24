from typing import Optional, List
import numpy as np

from marl.attacks.observation_attacks import ObservationAttacker, TargetMode, Observation


class RandomObservationAttacker(ObservationAttacker):
    """
    Applies random, bounded perturbations to agent observations.

    Parameters:
        epsilon: Maximum magnitude of the random noise added to each
            observation element.
        attack_probability: Probability of performing an attack at each
            timestep.
        target_agents: Agent indices eligible for attack. If None, all agents
            are eligible.
        target_mode: FIXED attacks all specified target agents; RANDOM selects
            a random non-empty subset of eligible agents.
        seed: Optional random seed for reproducible random noise and
            attack decisions.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        attack_probability: float = 1.0,
        target_agents: Optional[List[int]] = None,
        target_mode: TargetMode = TargetMode.FIXED,
        seed: Optional[int] = None,
    ) -> None:

        # Initialize the shared attack configuration.
        super().__init__(
            epsilon=epsilon,
            attack_probability=attack_probability,
            target_agents=target_agents,
            target_mode=target_mode,
            seed=seed,
        )


    def _perturb(self, observation: Observation, agent_idx: int,) -> Observation:
        """Apply random, bounded noise to one agent's observation."""

        # Generate random noise within the epsilon budget.
        noise = self.rng.uniform(
            -self.epsilon,
            self.epsilon,
            size=observation.shape,
        )

        # Add the noise to the original observation.
        return observation + noise
