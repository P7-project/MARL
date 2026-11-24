from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn


FIXED_TARGET = "fixed"
RANDOM_TARGET = "random"


class ObservationAttacker:
    """
    Base class for observation attacks with configurable agent targeting.
    """

    def __init__(
        self,
        epsilon: float,
        attack_probability: float = 1.0,
        target_agents: Optional[List[int]] = None,
        target_mode: str = FIXED_TARGET,
        seed: Optional[int] = None,
    ) -> None:

        self.epsilon = epsilon
        self.attack_probability = attack_probability
        self.target_agents = target_agents
        self.target_mode = target_mode

        # Dedicated random number generator.
        self.rng = np.random.default_rng(seed)

        # Stores the agents attacked at the most recent timestep.
        self.last_attacked_agents: List[int] = []

        # Validate the shared attack configuration.
        if self.target_mode not in [FIXED_TARGET, RANDOM_TARGET]:
            raise ValueError("target_mode must be 'fixed' or 'random'.")

        if self.epsilon < 0:
            raise ValueError("Attack strength cannot be negative.")

        if not 0.0 <= self.attack_probability <= 1.0:
            raise ValueError("Attack probability must be between 0 and 1.")

        if self.target_mode == FIXED_TARGET:
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
        if self.target_mode == FIXED_TARGET:
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



class RandomObservationAttacker(ObservationAttacker):
    """
    Applies random, bounded perturbations to agent observations.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        attack_probability: float = 1.0,
        target_agents: Optional[List[int]] = None,
        target_mode: str = FIXED_TARGET,
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


    def _perturb(self, observation: np.ndarray, agent_idx: int,) -> np.ndarray:
        """Apply random, bounded noise to one agent's observation."""

        # Generate random noise within the epsilon budget.
        noise = self.rng.uniform(
            -self.epsilon,
            self.epsilon,
            size=observation.shape,
        )

        # Add the noise to the original observation.
        return observation + noise




class PGDObservationAttacker(ObservationAttacker):
    """
    Applies gradient-based PGD perturbations to agent observations.
    """

    def __init__(
        self,
        model: nn.Module,
        n_agents: int,
        epsilon: float = 0.5,
        niters: int = 10,
        step_size: Optional[float] = None,
        random_start: bool = True,
        attack_probability: float = 1.0,
        target_agents: Optional[List[int]] = None,
        target_mode: str = FIXED_TARGET,
        seed: Optional[int] = None,
    ) -> None:

        # Validate PGD-specific parameters.
        if niters < 1:
            raise ValueError("niters must be at least 1.")

        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive.")

        if n_agents < 1:
            raise ValueError("n_agents must be at least 1.")

        # Initialize the shared attack configuration.
        super().__init__(
            epsilon=epsilon,
            attack_probability=attack_probability,
            target_agents=target_agents,
            target_mode=target_mode,
            seed=seed,
        )

        self.model = model
        self.n_agents = n_agents
        self.niters = niters
        self.step_size = step_size if step_size is not None else epsilon / niters
        self.random_start = random_start



    def _perturb(
        self,
        observation: torch.Tensor,
        agent_idx: int,
    ) -> torch.Tensor:
        """Apply PGD to one agent's observation."""

        # Convert the agent observation to a tensor 
        device = next(self.model.parameters()).device

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        )

        # Create the one-hot ID for this agent.
        agent_id = torch.zeros(
            self.n_agents,
            dtype=observation.dtype,
            device=observation.device,
        )
        agent_id[agent_idx] = 1.0

        # Append the agent ID to the observation.
        model_input = torch.cat([observation, agent_id])

        # Get the action selected by the model for the original observation.
        with torch.no_grad():
            original_q_values = self.model(model_input.unsqueeze(0))
            original_action = original_q_values.argmax(dim=1)

        # Define cross-entropy as the attack loss.
        loss_func = nn.CrossEntropyLoss()

        # Start from the original observation.
        X_adv = observation.detach().clone()

        # Optionally add random noise as the starting point.
        if self.random_start:
            noise = torch.empty_like(X_adv).uniform_(
                -self.epsilon,
                self.epsilon,
            )
            X_adv = X_adv + noise

        X_adv = torch.clamp(X_adv, -1.0, 1.0)
        # Enable gradient tracking for the adversarial observation.
        X_adv = X_adv.detach().requires_grad_(True)


        for i in range(self.niters):

            # Append the unchanged agent ID to the adversarial observation.
            model_input = torch.cat([X_adv, agent_id])

            # Get Q-values for the current adversarial observation.
            logits = self.model.forward(model_input.unsqueeze(0))

            # Calculate the attack loss using the original action as target.
            loss = loss_func(logits, original_action)

            # Clear previously accumulated model gradients.
            self.model.zero_grad()

            # Calculate gradients with respect to the adversarial observation.
            loss.backward()

            # Calculate the perturbation step using the observation gradient.
            eta = self.step_size * X_adv.grad.sign()

            # Update the adversarial observation.
            X_adv = (X_adv.detach() + eta).detach().requires_grad_(True)

            # Project the perturbation back into the allowed epsilon range.
            eta = torch.clamp(
                X_adv - observation,
                -self.epsilon,
                self.epsilon,
            )

            # Apply the projected perturbation to the original observation.
            X_adv = observation + eta

            # Keep observations within the normalized range [-1, 1].
            X_adv = torch.clamp(X_adv, -1.0, 1.0).detach()

            # Enable gradient tracking for the next iteration.
            X_adv.requires_grad_(True)


        return X_adv.detach()



            