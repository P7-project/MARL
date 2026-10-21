import argparse
import gymnasium as gym
import numpy as np
import highway_env  # noqa: F401  (registers the highway-* environments)
 
# Newer highway-env versions expose an explicit registration helper
if hasattr(gym, "register_envs"):
    gym.register_envs(highway_env)
 
N_AGENTS = 3
 
 
def make_env(n_agents: int = N_AGENTS, render_mode=None) -> gym.Env:
    config = {
        # --- multi-agent core -------------------------------------------
        "controlled_vehicles": n_agents,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteMetaAction"},
        },
        # --- scenario ---------------------------------------------------
        "lanes_count": 4,
        "vehicles_count": 1,
        "duration": 40,              # seconds
        "initial_lane_id": None,     # random lane per agent
        "policy_frequency": 1,
        "simulation_frequency": 15,
    }
    return gym.make("highway-v0", config=config, render_mode=render_mode)
 
class RandomPolicy:
    """Placeholde per-agent policy. Replace with your own (e.g. a neural net)."""
 
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space
 
    def act(self, obs: np.ndarray) -> int:
        return self.action_space.sample()
 
 
def main(render: bool = False, episodes: int = 3):
    env = make_env(render_mode="human" if render else None)
 
    # With MultiAgentAction, the action space is a Tuple with one space per agent
    policies = [RandomPolicy(space) for space in env.action_space.spaces]
 
    for ep in range(episodes):
        obs, info = env.reset()          # obs is a tuple: one array per agent
        done = truncated = False
        ep_return = 0.0
        steps = 0
 
        while not (done or truncated):
            # One action per agent -> tuple of actions
            actions = tuple(p.act(o) for p, o in zip(policies, obs))
 
            obs, reward, done, truncated, info = env.step(actions)
            ep_return += reward
            steps += 1
 
        print(
            f"Episode {ep + 1}: steps={steps}, return={ep_return:.2f}, "
            f"crashed={info.get('crashed')}, n_agents={len(obs)}"
        )
 
    env.close()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    main(render=args.render, episodes=args.episodes)
 
