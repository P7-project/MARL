import numpy as np

class RandomObservationAttacker:
    """
    Applies random perturbations to the observations received by agents
    in a multi-agent reinforcement learning environment.

    The attacker modifies agents' observations before they are passed
    to their policies. It does not directly modify the environment state.

    Targeting modes:
    - "single": Attacks one specified agent.
    - "multiple": Attacks a specified group of agents.
    - "random": Randomly selects a group of agents to attack.

    Attack parameters:
    - attack_strength: Controls the maximum magnitude of the random
      noise added to observations.
    - target_agents: Specifies which agents are attacked in "single"
      or "multiple" mode. For example, [0] attacks Agent 0, while
      [0, 2] attacks Agents 0 and 2.
    - attack_mode: Determines how target agents are selected.
      Can be "single", "multiple", or "random".
    - attack_interval: Defines the number of environment steps
      between the starts of consecutive attack cycles.
    - attack_duration: Defines how many consecutive environment
      steps the attack remains active during each attack cycle.
    - seed: Optional random seed used to make agent selection and
      noise generation reproducible.

    """

    def __init__(
        self,
        attack_strength=0.05,
        target_agents=None,
        attack_mode="single",
        attack_interval=10,
        attack_duration=3,
        seed=None
    ):
        self.attack_strength = attack_strength
        self.target_agents = target_agents
        self.attack_mode = attack_mode
        self.attack_interval = attack_interval
        self.attack_duration = attack_duration

        # Dedicated random number generator.
        self.rng = np.random.default_rng(seed)


        # Keep track of the current environment step and selected targets.
        self.current_step = 0
        self.selected_agents = None

        # Validate the attack configuration.
        if self.attack_mode not in ["single", "multiple", "random"]:
            raise ValueError(
                "attack_mode must be 'single', 'multiple', or 'random'."
            )

        if self.attack_strength < 0:
            raise ValueError("Attack strength cannot be negative.")

        if self.attack_interval <= 0 or self.attack_duration <= 0:
            raise ValueError("Attack interval and duration must be positive.")

        if self.attack_duration > self.attack_interval:
            raise ValueError(
                "Attack duration cannot exceed attack interval."
            )

  
        # Fixed targeting modes require a list of target agents.
        if self.attack_mode in ["single", "multiple"]:
            if self.target_agents is None or len(self.target_agents) == 0:
                raise ValueError(
                    "target_agents must be provided for fixed targeting."
                )

            if self.attack_mode == "single" and len(self.target_agents) != 1:
                raise ValueError(
                    "Single mode requires exactly one target agent."
                )


    
    def reset(self):
        """Reset the attack schedule at the beginning of an episode."""
        self.current_step = 0
        self.selected_agents = None


    def _select_agents(self, n_agents):
        """Select the agents to attack based on the configured mode."""

        if self.attack_mode in ["single", "multiple"]:
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




    def attack(self, observations):
        """
        Apply random noise to selected agents at scheduled attack steps.

        observations: Array with shape (n_agents, obs_dim).
        """


        # Create a copy so the original observations remain unchanged.
        attacked_observations = observations.copy()

        # Check that the target agent exists.
        n_agents = observations.shape[0]

        # Check whether an attack should be active at this timestep.
        attack_active = (
            self.current_step % self.attack_interval
            < self.attack_duration
        )

        if attack_active:

            # Select targets at the beginning of each attack cycle.
            if self.current_step % self.attack_interval == 0:
                self.selected_agents = self._select_agents(n_agents)


            # Apply noise to each selected agent.
            for agent in self.selected_agents:
                noise = self.rng.uniform(
                    -self.attack_strength,
                    self.attack_strength,
                    size=attacked_observations[agent].shape
                )

                attacked_observations[agent] += noise


        self.current_step += 1

        return attacked_observations

