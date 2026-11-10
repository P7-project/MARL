from pathlib import Path
import sys

import torch
from torch import nn
import numpy as np

import random
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marl.env import make_env, flatten_obs

import importlib.util

# Locate the WocaR DQN implementation.
dqn_path = (
    Path(__file__).resolve().parents[1]
    / "woca-r-rl"
    / "dqn.py"
)

# Load the module from its file path.
spec = importlib.util.spec_from_file_location("wocar_dqn", dqn_path)
wocar_dqn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wocar_dqn)

# Import the WocaR components we need.
WorstAttackBellmanOperator = wocar_dqn.WorstAttackBellmanOperator
CertifiedActionSetEstimator = wocar_dqn.CertifiedActionSetEstimator



# --------------------------------------------------
# Q-Network
# --------------------------------------------------

class QNetwork(nn.Module):
    """
    Neural network that estimates Q-values for each possible action.

    Each agent receives its own observation and the network
    outputs one Q-value for each of the five driving actions.
    """

    def __init__(self,state_dim,action_dim):
        """
        Initialise the Q-network.

        Parameters:
            state_dim: Number of input features in the agent's observation (25).
            action_dim: Number of possible actions (5).
        """
        super().__init__()

        # Define the neural network architecture.
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), # Input layer: Takes the agent's observation
            nn.ReLU(),                # Activation function
            nn.Linear(64,64),         # Hidden layer
            nn.ReLU(),
            nn.Linear(64, action_dim) # Output layer: Produces one Q-value for each possible action.
        )

    def forward(self, x):
        """
        Pass an observation through the network
        and return the estimated Q-values.
        """
        return self.network(x)


# --------------------------------------------------
# Replay buffer
# --------------------------------------------------
class ReplayBuffer:
    """
    Stores experiences so DQN agents can reuse past interactions during training.
    Random sampling reduces correlations between consecutive experiences.
    """


    def __init__(self,capacity):
        # Store experiments up to the sepcifed capcity
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        # Add a new experince to the buffer
        self.buffer.append(transition)

    def sample(self, batch_size):
        # Randomly select a batch of experiences
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------
# Agent setup
# --------------------------------------------------

def create_agents(n_agents, state_dim, action_dim):
    """
    Create independent online and target Q-networks for each agent.

    Parameters:
        n_agents: Number of agents in the environment.
        state_dim: Number of observation features per agent.
        action_dim: Number of possible actions per agent.

    Returns:
        q_networks: List containing one online Q-network per agent.
        target_q_networks: List containing one target Q-network per agent.
    """

    # Create one independent online Q-network for each agent.
    q_networks = [
        QNetwork(state_dim, action_dim)
        for _ in range(n_agents)
    ]

    # Create one independent target Q-network for each agent.
    target_q_networks = [
        QNetwork(state_dim, action_dim)
        for _ in range(n_agents)
    ]


    # Initialise each target network with the weights
    for q_network, target_network in zip(q_networks, target_q_networks):

        # Copy the online network's parameters into the target network.
        target_network.load_state_dict(q_network.state_dict())

     

    return q_networks, target_q_networks



# --------------------------------------------------
# Update target network
# --------------------------------------------------

def update_target_networks(q_networks, target_q_networks):

    # Update each agent's target network with its online network's weights.
    for q_network, target_network in zip(
        q_networks, target_q_networks
    ):
        target_network.load_state_dict(
            q_network.state_dict()
        )



# --------------------------------------------------
# Select Action
# --------------------------------------------------

def select_action(q_network, state, epsilon, action_dim):
    # Explore: choose a random action.
    if random.random() < epsilon:
        return random.randrange(action_dim)

    # Exploit: choose the action with the highest Q-value.
    with torch.no_grad():
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_network(state)
        return q_values.argmax(dim=1).item()

    
# --------------------------------------------------
# Experience collection
# --------------------------------------------------

def collect_experiences(  env, q_networks, n_agents, num_steps, epsilon, action_dim):
    """
        Collect experiences for each agent by interacting with the environment
        using epsilon-greedy action selection.

        Parameters:
            env: The multi-agent highway environment.
            q_networks: Separate Q-network for each agent.
            n_agents: Number of controlled agents.
            num_steps: Number of environment interactions to perform.
            epsilon: Probability of selecting a random action (exploration).
            action_dim: Number of possible actions per agent.

        Returns:
            agent_transitions: Separate list of experiences for each agent.
            Each transition contains (state, action, reward, next_state, terminated).
    """

    # Reset the environment to start a new episode.
    obs, info = env.reset()

    # Create a separate list for storing experiences for each agent.
    agent_transitions = [[] for _ in range(n_agents)]

    for step in range(num_steps):
        # Flatten observations so each agent gets its own state vector.
        obs_flat = flatten_obs(obs)

        # Let each agent select an action using its own Q-network.
        actions = tuple(
            select_action(
                q_networks[i],
                obs_flat[i],
                epsilon,
                action_dim
            )
            for i in range(n_agents)
        )

        # Execute all agents' actions simultaneously in the environment.
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        # Convert each agent's current observation into a flat vector.
        next_obs_flat = flatten_obs(next_obs)

        # Store one transition for each agent.
        for i in range(n_agents):

            transition = (
                obs_flat[i],
                actions[i],
                rewards,
                next_obs_flat[i],
                terminated,
            )

            agent_transitions[i].append(transition)

         # Check whether the current episode has ended.
        if terminated or truncated:
            obs, info = env.reset()
        else:
            # If the episode is still running, use the next observations
            obs = next_obs

    return agent_transitions


# --------------------------------------------------
# Train agents
# --------------------------------------------------

def train_agents(
    q_networks,
    optimizers,
    replay_buffers,
    bellman_operators,
    batch_size
    ):


    losses = []

    # Train each agent independently.
    for i in range(len(q_networks)):
        # Wait until the replay buffer contains enough experiences.
        if len(replay_buffers[i]) < batch_size:
            continue


        # Randomly sample experiences for this agent.
        batch = replay_buffers[i].sample(batch_size)

        # Separate the experiences into their individual components.
        states, actions, rewards, next_states, terminated = zip(*batch)

        # Convert the sampled data into PyTorch tensors.
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.float32)

        # Get the Q-values predicted by the online network.
        all_q_values = q_networks[i](states)

        # Select only the Q-values for the actions actually taken.
        predicted_q_values = all_q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Calculate the WocaR worst-case Bellman targets.
        targets = bellman_operators[i].compute_targets(
            rewards,
            next_states,
            terminated,
        )

        # Measure the difference between predictions and targets.
        loss = nn.functional.mse_loss(predicted_q_values, targets)

        # Update the online Q-network using gradient descent.

        # Clear old gradients
        optimizers[i].zero_grad()

        # Calculate gradients
        loss.backward()

        # Adam updates the network weights
        optimizers[i].step()

        
        losses.append(loss.item())

    return losses

# --------------------------------------------------
# Create Bellman operator
# --------------------------------------------------

def create_bellman_operators(
    q_networks,
    target_q_networks,
    n_agents,
    state_dim,
    action_dim,
    gamma,
    epsilon,
):
    """
    Create one WocaR Bellman operator for each agent.

    Each operator uses the agent's online network to define its policy,
    and its target network to calculate worst-case future Q-values.
    """
    bellman_operators = []

    for i in range(n_agents):

        # Estimate which actions are reachable under adversarial perturbations.
        estimator = CertifiedActionSetEstimator(
            policy_network=q_networks[i],
            epsilon=epsilon,
            num_actions=action_dim,
            example_state=torch.zeros(1, state_dim),
        )

        # Create the WocaR Bellman operator for this agent.
        operator = WorstAttackBellmanOperator(
            target_q_network=target_q_networks[i],
            action_set_estimator=estimator,
            discount_factor=gamma,
        )

        bellman_operators.append(operator)

    return bellman_operators

# --------------------------------------------------
# Print summary
# --------------------------------------------------


def print_experiment_summary(agent_transitions):
    """
    Print statistics about the experiences collected by each agent.
    """

    print("\n========== EXPERIMENT SUMMARY ==========")

    for i, transitions in enumerate(agent_transitions):

        states, actions, rewards, next_states, terminated = zip(*transitions)

        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)

        print(f"\n========== AGENT {i + 1} ==========")

        # Number of experiences collected.
        print(f"Transitions collected: {len(transitions)}")

        # Distribution of selected actions.
        unique_actions, action_counts = np.unique(
            actions, return_counts=True
        )

        print("\nAction distribution:")
        for action, count in zip(unique_actions, action_counts):
            percentage = count / len(actions) * 100
            print(
                f"  Action {action}: {count} times "
                f"({percentage:.1f}%)"
            )

        # Reward statistics.
        print("\nReward statistics:")
        print(f"  Average reward: {np.mean(rewards):.3f}")
        print(f"  Minimum reward: {np.min(rewards):.3f}")
        print(f"  Maximum reward: {np.max(rewards):.3f}")
        print(f"  Total reward:   {np.sum(rewards):.3f}")

        # State statistics.
        print("\nState statistics:")
        print(f"  Average state value: {np.mean(states):.3f}")
        print(f"  Average state std:   {np.std(states):.3f}")

        # Episode termination statistics.
        terminal_count = sum(terminated)
        terminal_percentage = terminal_count / len(transitions) * 100

        print("\nEpisode statistics:")
        print(f"  Terminal transitions: {terminal_count}")
        print(f"  Terminal percentage:  {terminal_percentage:.2f}%")

    print("\n=========================================")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # Define the experiment configuration.
    render = True
    
    n_agents = 3
    state_dim = 25
    action_dim = 5


    num_steps = 100

    capacity = 1000

    gamma = 0.99
    epsilon = 0.01

    batch_size = 16
    training_iterations = 50

   
    # Create an online and target Q-network for each agent.
    q_networks, target_q_networks = create_agents(
        n_agents,
        state_dim,
        action_dim,
    )

    bellman_operators = create_bellman_operators(
        q_networks,
        target_q_networks,
        n_agents,
        state_dim,
        action_dim,
        gamma,
        epsilon,
    )

    # Create an indenpendt replay buffer for each agent
    replay_buffers = [
        ReplayBuffer(capacity=capacity)
        for _ in range(n_agents)
    ]

    # Create the multi-agent highway environment.
    env = make_env(
        n_agents=n_agents,
        render_mode="human" if render else None)

    try:
        # Let the agents interact with the environment and collect experiences.
        agent_transitions = collect_experiences(
            env,
            q_networks,
            n_agents,
            num_steps,
            epsilon,
            action_dim,
        )
    finally:
        # Ensure the environment is closed after collection.
        env.close()


    # Add collected experiences to each agent's replay buffer.
    for i in range(n_agents):
        for transition in agent_transitions[i]:
            replay_buffers[i].add(transition)

    optimizers = [
        torch.optim.Adam(network.parameters(), lr=1e-3)
        for network in q_networks
    ]

    # Train agents
    for step in range(training_iterations):

        losses = train_agents(
            q_networks,
            optimizers,
            replay_buffers,
            bellman_operators,
            batch_size
        )


        if (step + 1) % 50 == 0:
            print(f"Training step: {step + 1}")
            print(f"Agent losses: {losses}")

        # Update target networks every 100 iterations.
        if (step + 1) % 100 == 0:
            update_target_networks(q_networks, target_q_networks)

    print("Training losses:", losses)


    print_experiment_summary(agent_transitions)



if __name__ == "__main__":
    main()