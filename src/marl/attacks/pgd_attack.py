import numpy as np
import torch

import torch.nn as nn

from typing import Optional, List

FIXED_TARGET = "fixed"
RANDOM_TARGET = "random"

class PGDObservationAttacker:
    """
    Applies gradient-based adversarial perturbations to agent observations
    using Projected Gradient Descent (PGD).

    Targeting modes:
    - "fixed": Attacks the agents specified in target_agents.
    - "random": Randomly selects a group of agents to attack.

    Attack parameters:
    - model: Trained WocaR-DQN Q-network used to generate adversarial observations.
    - epsilon: Maximum magnitude of observation perturbation
    - niters: Number of PGD iterations per attack.
    - step_size: Size of each gradient update.
    - random_start: Whether to start PGD with random noise.
    - attack_probability: Probability of attacking at each timestep.
    - target_agents: Agents to attack in fixed mode.
    - target_mode: "fixed" or "random".
    - seed: Optional seed for reproducible random selection.
    """
       
    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 0.05,
        niters: int = 10,
        step_size: Optional[float] = None,
        random_start: bool = True,
        attack_probability: float = 1.0,
        target_agents: Optional[List[int]] = None,
        target_mode: str = FIXED_TARGET,
        seed: Optional[int] = None,
    ) -> None:
        
        self.model = model
        self.epsilon = epsilon
        self.niters = niters
        self.step_size = step_size if step_size is not None else epsilon / niters
        self.random_start = random_start
        self.attack_probability = attack_probability
        self.target_agents = target_agents
        self.target_mode = target_mode

        # Dedicated random number generator.
        self.rng = np.random.default_rng(seed)

        # Stores the agents attacked at the most recent timestep.
        self.last_attacked_agents = []


        # Validate the attack configuration.
        if self.target_mode not in [FIXED_TARGET, RANDOM_TARGET]:
            raise ValueError(
                "target_mode must be 'fixed' or 'random'."
            )

        if self.epsilon < 0:
            raise ValueError("Attack strength cannot be negative.")

        if self.niters <= 0:
            raise ValueError("Number of PGD iterations must be positive.")

        if self.step_size <= 0:
            raise ValueError("Step size must be positive.")

        if not 0.0 <= self.attack_probability <= 1.0:
            raise ValueError(
                "Attack probability must be between 0 and 1."
            )

        # Fixed targeting requires a list of target agents.
        if self.target_mode == FIXED_TARGET:
            if self.target_agents is None or len(self.target_agents) == 0:
                raise ValueError(
                    "target_agents must be provided for fixed targeting."
                )
    
    def _select_agents(self, n_agents: int) -> List[int]:
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

    def attack(self, observations: torch.Tensor) -> torch.Tensor:
        # Create a copy so the original observations remain unchanged.
        attacked_observations = observations.clone()

        # Reset the list because each timestep gets a new attack decision.
        self.last_attacked_agents = []


        # Decide whether an attack happens at this timestep.
        if self.rng.random() >= self.attack_probability:
            return attacked_observations


        # Get the number of agents.
        n_agents = observations.shape[0]

        # Select agents to attack at this timestep.
        selected_agents = self._select_agents(n_agents)

        # Remember which agents are being attacked for visualization.
        self.last_attacked_agents = selected_agents

        for agent in selected_agents:
            attacked_observations[agent] = pgd(
                model=self.model,
                X=observations[agent].unsqueeze(0),
                epsilon=self.epsilon,
                niters=self.niters,
                step_size=self.step_size,
                random_start=self.random_start,
            ).squeeze(0)

        return attacked_observations




def pgd(
    model, 
    X,
    y, 
    epsilon = 0.5,
    niters=10,
    step_size=None,
    random_start=True,
):
    """
    Generate an adversarial observation using PGD.

    Args:
        model: Trained WocaR-DQN Q-network.
        X: Original agent observation.
        y: Action selected from the original observation.
        epsilon: Maximum L-infinity perturbation.
        niters: Number of PGD iterations.
        step_size: Gradient step size; defaults to epsilon / niters.
        random_start: Whether to start with random noise.

    Returns:
        Perturbed observation within the epsilon budget.
    """


    if step_size is None:
        step_size = epsilon / niters

    # Define cross-entropy as the attack loss
    loss_func = nn.CrossEntropyLoss()

    # Ensure the target action has the correct format
    y = torch.as_tensor(y, dtype=torch.long, device=X.device)
    
    # Start from the original observation
    X_adv = X.detach().clone()

    if random_start: 

        # Generate random noise 
        noise = 2 * epsilon * torch.rand(X.size(), device=X.device) - epsilon

        # Add noise to the original observation
        X_adv = X_adv + noise

    # Enable gradient tracking for the adversarial observation
    X_adv = X_adv.detach().requires_grad_(True)


    for i in range(niters):

        # Get Q-values for the current adversarial observation
        logits = model.forward(X_adv)

        # Calculate the attack loss using the original action as the target
        loss = loss_func(logits, y)

        # Calculate gradients of the attack loss with respect to the adversarial observation
        loss.backward()

        # Calculate the perturbation step using the sign of the observation gradient
        eta = step_size * X_adv.grad.data.sign()

        # Update the adversarial observation using the calculated perturbation 
        # and enable gradient tracking for the next iteration
        X_adv = (X_adv.data + eta).detach().requires_grad_(True)

        # Project the perturbation back into the allowed epsilon range
        eta = torch.clamp(X_adv.data - X.data, -epsilon, epsilon)

        # Apply the projected perturbation to the original observation
        X_adv.data = X.data + eta

        # Keep adversarial observations within the normalized range [-1, 1]
        X_adv.data = torch.clamp(X_adv.data, -1.0, 1.0)

    # Return the final adversarial observation
    return X_adv.data




