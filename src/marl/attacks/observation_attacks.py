from typing import Optional, List, Union
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

Observation = Union[np.ndarray, torch.Tensor]

class TargetMode(Enum):
    """Available modes for selecting agents to attack."""
    FIXED = "fixed"
    RANDOM = "random"




class ObservationAttacker:
    """
    Base class for observation attacks with configurable agent targeting.

    Parameters:
        epsilon: Maximum allowed magnitude of the observation perturbation.
        attack_probability: Probability of performing an attack at each timestep.
        target_agents:  List of agent indices to attack. 
                        Defaults to [0] for fixed targeting. If None in random
                        targeting, all agents are eligible.
        target_mode: FIXED attacks all specified target agents, while RANDOM
            selects a random non-empty subset of eligible agents.
        seed: Optional random seed for reproducible random attack decisions.
    """

    def __init__(
        self,
        epsilon: float,
        attack_probability: float = 1.0,
        target_agents: Optional[List[int]] = None,
        target_mode: TargetMode = TargetMode.FIXED,
        seed: Optional[int] = None,
    ) -> None:

        self.epsilon = epsilon
        self.attack_probability = attack_probability
        self.target_mode = target_mode


        # Default to attacking agent 1 when using fixed targeting.
        if target_agents is None and target_mode == TargetMode.FIXED:
            self.target_agents = [0]
        else:
            self.target_agents = target_agents




        

        # Dedicated random number generator.
        self.rng = np.random.default_rng(seed)

        # Stores the agents attacked at the most recent timestep.
        self.last_attacked_agents: List[int] = []

        # Validate the shared attack configuration.
        if not isinstance(self.target_mode, TargetMode):
            raise ValueError("target_mode must be 'fixed' or 'random'.")

        if self.epsilon < 0:
            raise ValueError("Attack strength cannot be negative.")

        if not 0.0 <= self.attack_probability <= 1.0:
            raise ValueError("Attack probability must be between 0 and 1.")

        if self.target_mode == TargetMode.FIXED:
            if self.target_agents is None or len(self.target_agents) == 0:
                raise ValueError(
                    "target_agents must be provided for fixed targeting."
                )

    def _select_agents(self, n_agents: int) -> List[int]:
        """Select agents based on the configured targeting mode."""

        # Determine which agents are eligible for attack.
        if self.target_agents is None:
            candidate_agents = list(range(n_agents))
        else:
            candidate_agents = list(self.target_agents)

        # Check that all candidate agents exist.
        if any(agent < 0 or agent >= n_agents for agent in candidate_agents):
            raise ValueError("A target agent index is invalid.")

        if len(set(candidate_agents)) != len(candidate_agents):
            raise ValueError("Duplicate target agents are not allowed.")

        # Fixed targeting selects all specified agents.
        if self.target_mode == TargetMode.FIXED:
            return candidate_agents

        # Random targeting selects a non-empty subset of eligible agents.
        num_agents = int(
            self.rng.integers(1, len(candidate_agents) + 1)
        )

        selected_agents = self.rng.choice(
            candidate_agents,
            size=num_agents,
            replace=False,
        )

        return list(selected_agents)


    def attack(self, observations: np.ndarray) -> np.ndarray:
        """
        Apply perturbations to selected agents with a given probability.

        Args:
            observations: Observations of all agents.

        Returns:
            Observations with perturbations applied to selected agents.
        """

          # Create a copy so the original observations remain unchanged.
        if isinstance(observations, torch.Tensor):
            attacked_observations = observations.clone()
        else:
            attacked_observations = observations.copy()

        # Reset the list because each timestep gets a new attack decision.
        self.last_attacked_agents = []

        # Decide whether an attack happens at this timestep.
        if self.rng.random() >= self.attack_probability:
            return attacked_observations

        # Get the number of agents.
        n_agents = observations.shape[0]

        # Select agents to attack at this timestep.
        selected_agents = self._select_agents(n_agents)

        # Remember the agents being attacked for visualization.
        self.last_attacked_agents = selected_agents

        # Apply the specific perturbation to each selected agent.
        for agent in selected_agents:
            perturbed_observation = self._perturb(
                observations[agent], agent
            )

            # Convert PyTorch tensors back to NumPy when needed.
            if isinstance(attacked_observations, np.ndarray):
                if isinstance(perturbed_observation, torch.Tensor):
                    perturbed_observation = (
                        perturbed_observation.detach().cpu().numpy()
                    )

            attacked_observations[agent] = perturbed_observation


        return attacked_observations


    def _perturb(
        self,
        observation: Observation,
        agent_idx: int,
    ) -> Observation:
        """Apply an observation perturbation for one agent."""
        raise NotImplementedError





