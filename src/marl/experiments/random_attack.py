from typing import Optional, List
import numpy as np


FIXED_TARGET = "fixed"
RANDOM_TARGET = "random"


class RandomObservationAttacker:
    """
    Applies random, bounded perturbations to agent observations
    without modifying the environment state.

    Targeting modes:
    - "fixed": Attacks the agents specified in target_agents.
    - "random": Randomly selects a non-empty group of agents.

    Attack parameters:
    - epsilon: Maximum magnitude of observation noise (perturbation budget).
    - attack_probability: Probability of attacking at each timestep.
    - target_agents: Agents to attack in fixed mode.
    - target_mode: "fixed" or "random".
    - seed: Optional seed for reproducible random selection and noise.
    """

    def __init__(
        self,
        epsilon: float =0.05,
        attack_probability: float = 1.0,
        target_agents:  Optional[List[int]] =None,
        target_mode: str = FIXED_TARGET,
        seed: Optional[int] = None
    ) -> None:
        
        self.epsilon = epsilon
        self.attack_probability = attack_probability
        self.target_agents = target_agents
        self.target_mode = target_mode
  
        # Dedicated random number generator.
        self.rng = np.random.default_rng(seed)

        # Stores the agents attacked at the most recent timestep.
        self.last_attacked_agents: List[int] = []



        # Validate the attack configuration.
        if self.target_mode not in [FIXED_TARGET, RANDOM_TARGET]:
            raise ValueError(
                "target_mode must be 'fixed' or 'random'."
            )

        if self.epsilon < 0:
            raise ValueError("Attack strength cannot be negative.")

        if not 0.0 <= self.attack_probability <= 1.0:
            raise ValueError("Attack probability must be between 0 and 1.")
        
        # Fixed targeting requires a list of target agents.
        if self.target_mode == FIXED_TARGET:
            if self.target_agents is None or len(self.target_agents) == 0:
                raise ValueError(
                    "target_agents must be provided for fixed targeting."
                )



    def _select_agents(self, n_agents) -> List[int]:
        """Select the agents to attack based on the configured mode."""

        if self.target_mode == FIXED_TARGET:
            assert self.target_agents is not None
            agents = list(self.target_agents)

        else:
            # Randomly select a non-empty group of agents.
            num_agents = int(self.rng.integers(1, n_agents + 1))

            agents = self.rng.choice(
                n_agents,
                size=num_agents,
                replace=False
            ).tolist()

        # Check that all selected agents exist.
        if any(agent < 0 or agent >= n_agents for agent in agents):
            raise ValueError("A target agent index is invalid.")

        if len(set(agents)) != len(agents):
            raise ValueError("Duplicate target agents are not allowed.")

        return agents

    def attack(self, observations) -> np.ndarray:
        """
          Apply random noise to selected agents with a given probability.

        """

        # Create a copy so the original observations remain unchanged.
        attacked_observations = observations.copy()

        # Reset the list because each timestep gets a new attack decision.
        self.last_attacked_agents = []

        # Decide whether an attack happens at this timestep.
        # rng.random() generates values in [0.0, 1.0)
        if self.rng.random() >= self.attack_probability:
            return attacked_observations

        n_agents = observations.shape[0]


        # Select agents to attack at this timestep.
        selected_agents = self._select_agents(n_agents)

        # Remember which agents are being attacked for visualization.
        self.last_attacked_agents = selected_agents

        # Apply noise to each selected agent.
        for agent in selected_agents:
            noise = self.rng.uniform(
                -self.epsilon,
                self.epsilon,
                size=attacked_observations[agent].shape
            )

            attacked_observations[agent] += noise


        return attacked_observations
