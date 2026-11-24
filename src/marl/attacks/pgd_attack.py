from typing import Optional, List

import torch
from torch import nn


from marl.attacks.observation_attacks import ObservationAttacker, TargetMode, Observation



class PGDObservationAttacker(ObservationAttacker):
    """
    Applies gradient-based PGD perturbations to agent observations.

     Parameters:
        model: The Q-network used to calculate the adversarial perturbations.
        n_agents: Total number of agents, used to construct the one-hot agent ID.
        epsilon: Maximum allowed perturbation for each observation element.
        niters: Number of PGD iterations used to generate the perturbation.
        step_size: Size of each PGD update. Defaults to epsilon / niters.
        random_start: Whether to initialize PGD with random noise.
        attack_probability: Probability of performing an attack at each timestep.
        target_agents: Agent indices eligible for attack. If None, all agents
            are eligible.
        target_mode: FIXED attacks all specified target agents; RANDOM selects
            a random non-empty subset of eligible agents.
        seed: Optional random seed for reproducible attack decisions and
            random initialization.
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
        target_mode: TargetMode = TargetMode.FIXED,
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
        observation: Observation,
        agent_idx: int,
    ) -> Observation:
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

            # Get the gradient of the adversarial observation.
            grad = X_adv.grad

            # Ensure the gradient was calculated.
            if grad is None:
                raise RuntimeError("Failed to compute observation gradient.")

            # Calculate the perturbation step using the observation gradient.
            eta = self.step_size * grad.sign()

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



            