"""Highway environment setup and observation helpers."""

import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
import pygame
from highway_env.vehicle.graphics import VehicleGraphics


from highway_env.envs.common.graphics import EnvViewer, ObservationGraphics
from highway_env.road.graphics import RoadGraphics

ATTACKED_ATTRIBUTE = "attacked"


class AttackedVehicleGraphics(VehicleGraphics):
    """Draw a red outline around vehicles receiving observation attacks."""


    @classmethod
    def display(
        cls,
        vehicle,
        surface,
        transparent=False,
        offscreen=False,
        label=False,
        draw_roof=False,
    ) -> None:


     # Draw the vehicle using the original renderer.
        super().display(
            vehicle,
            surface,
            transparent=transparent,
            offscreen=offscreen,
            label=label,
            draw_roof=draw_roof,
        )

     

        # Only highlight vehicles marked as attacked.
        if getattr(vehicle, ATTACKED_ATTRIBUTE, False):

            # Convert the vehicle dimensions from world coordinates to pixels.
            # Add a small margin so the outline sits just outside the vehicle.

            length = surface.pix(vehicle.LENGTH + 2)
            width = surface.pix(vehicle.WIDTH + 1)

            # Create a transparent square surface for drawing the vehicle outline.
            vehicle_surface = pygame.Surface(
                (length, length),
                flags=pygame.SRCALPHA,
            )

            # Draw a red rectangle around the vehicle.
            pygame.draw.rect(
                vehicle_surface,
                (255, 0, 0),
                (
                    surface.pix(1),
                    (length - width) // 2,
                    surface.pix(vehicle.LENGTH),
                    width,
                ),
                width=3,
            )

            # Rotate the outline to match the vehicle's heading and draw it
            cls.blit_rotate(
                surface,
                vehicle_surface,
                [*surface.pos2pix(vehicle.position[0], vehicle.position[1])],
                np.rad2deg(-vehicle.heading),
            )

            
class AttackedRoadGraphics(RoadGraphics):
    """Render road traffic using the custom attacked-vehicle renderer."""

    @staticmethod
    def display_traffic(
        road,
        surface,
        simulation_frequency=15,
        offscreen=False,
    ) -> None:

        # Draw each vehicle's history and then render the vehicle itself.
        for vehicle in road.vehicles:
            VehicleGraphics.display_history(
                vehicle, surface, simulation_frequency
            )
            # Use the custom renderer to highlight attacked vehicles.
            AttackedVehicleGraphics.display(
                vehicle, surface, offscreen=offscreen
            )

        

         

class AttackedEnvViewer(EnvViewer):
    """Render the environment with attacked vehicles highlighted."""


    def display(self) -> None:
        """Display the road and vehicles using custom graphics."""

        if not self.enabled:
            return

        # Position the simulation view according to the current observer position.
        self.sim_surface.move_display_window_to(self.window_position())


        # Draw the road, including its lanes and markings, onto the simulation surface.
        RoadGraphics.display(self.env.road, self.sim_surface)


        # Draw the vehicle trajectory, if one is available.
        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen,
            )


        # Draw road objects, such as obstacles, on the simulation surface.
        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen,
        )


        # Draw the optional agent visualization, if one has been configured.
        if EnvViewer.agent_display:

            # Render the agent visualization onto the separate agent surface.
            EnvViewer.agent_display(self.agent_surface, self.sim_surface)

            # Display the agent visualization only when rendering to the screen.
            if not self.offscreen:

                # For landscape screens, position the agent visualization below the simulation.
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(
                        self.agent_surface,
                        (0, self.config["screen_height"]),
                    )
                # For portrait screens, position it to the right of the simulation.
                else:
                    self.screen.blit(
                        self.agent_surface,
                        (self.config["screen_width"], 0),
                    )



        # Draw road traffic using the custom renderer.
        AttackedRoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen,
        )


        # Draw the current observation on the simulation surface.
        ObservationGraphics.display(
            self.env.observation_type,
            self.sim_surface,
        )


        # Update the screen when rendering to a visible display.
        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))

            # Maintain the configured simulation frequency during real-time rendering.
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])

            # Refresh the display to show the latest simulation frame.
            pygame.display.flip()

        # Save the current simulation frame if image saving is enabled.
        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(
                self.sim_surface,
                str(self.directory / f"highway-env_{self.frame}.png"),
            )

            # Increment the frame counter for the next saved image.
            self.frame += 1





class TeamHighwayEnv(HighwayEnv):
    """Highway driving with one reward and episode outcome shared by all controlled vehicles."""

    def set_attacked_agents(self, agent_indices: list[int]) -> None:
        """Set the agents whose vehicles should be highlighted during rendering."""
        self.attacked_agents = set(agent_indices)

         # Mark each controlled vehicle according to whether its agent is attacked.
        for agent_idx, vehicle in enumerate(self.controlled_vehicles):
            vehicle.attacked = agent_idx in self.attacked_agents
           


    def render(self):
        # Create the custom viewer the first time rendering is requested.
        if self.render_mode is not None and self.viewer is None:
            self.viewer = AttackedEnvViewer(self)

        # Continue with HighwayEnv's original rendering process.
        return super().render()
    
    def _reward(self, action) -> float:
        """Average the per-vehicle driving reward over all controlled vehicles."""
        return float(
            np.mean(
                [self._agent_reward(vehicle) for vehicle in self.controlled_vehicles]
            )
        )

    def _rewards(self, action) -> dict[str, float]:
        """Average each reward term over all controlled vehicles."""
        agents_rewards = [
            self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: float(
                np.mean([agent_rewards[name] for agent_rewards in agents_rewards])
            )
            for name in agents_rewards[0]
        }

    def _agent_reward(self, vehicle) -> float:
        """HighwayEnv._reward, applied to the given vehicle."""
        rewards = self._agent_rewards(vehicle)
        reward = sum(
            self.config.get(name, 0) * value for name, value in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        return reward * rewards["on_road_reward"]

    def _agent_rewards(self, vehicle) -> dict[str, float]:
        """HighwayEnv._rewards, applied to the given vehicle."""
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if any controlled vehicle crashed or, if configured, left the road."""
        return any(
            vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road)
            for vehicle in self.controlled_vehicles
        )

    def _info(self, obs, action=None) -> dict:
        """Report team values under the standard keys and per-vehicle values under agents_*."""
        vehicles = self.controlled_vehicles
        info = super()._info(obs, action)
        info["speed"] = float(np.mean([vehicle.speed for vehicle in vehicles]))
        info["crashed"] = any(vehicle.crashed for vehicle in vehicles)
        info["agents_speed"] = tuple(float(vehicle.speed) for vehicle in vehicles)
        info["agents_crashed"] = tuple(vehicle.crashed for vehicle in vehicles)
        info["agents_rewards"] = tuple(
            float(self._agent_reward(vehicle)) for vehicle in vehicles
        )
        return info


gym.register(id="team-highway-v0", entry_point=TeamHighwayEnv)

def make_env(n_agents: int = 3, render_mode=None, name: str = "highway",) -> gym.Env:
    if name == "intersection":
        return make_intersection_env(n_agents, render_mode)
    if name == "highway":
        return make_highway_env(n_agents, render_mode)
    raise ValueError(f"Unknown environment: {name}")


def make_intersection_env(n_agents: int = 3, render_mode=None) -> gym.Env:
    return gym.make(
        "intersection-multi-agent-v2",
        config={"controlled_vehicles": n_agents},
        render_mode=render_mode,
    )


def make_highway_env(n_agents: int = 3, render_mode=None) -> gym.Env:
    config = {
        "controlled_vehicles": n_agents,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteMetaAction"},
        },
        "lanes_count": 4,
        "vehicles_count": 30,
        "duration": 40,
        "initial_lane_id": None,
        "policy_frequency": 1,
        "simulation_frequency": 15,
    }
    return gym.make("team-highway-v0", config=config, render_mode=render_mode)


def flatten_obs(obs) -> np.ndarray:
    """Convert one observation per agent to a float32 (agents, features) array."""
    return np.stack(
        [np.asarray(agent_obs, dtype=np.float32).reshape(-1) for agent_obs in obs]
    )


def build_actor_obs(obs_flat: np.ndarray, n_agents: int) -> np.ndarray:
    """Append one-hot agent IDs to the per-agent observations."""
    agent_ids = np.eye(n_agents, dtype=np.float32)
    return np.concatenate([obs_flat, agent_ids], axis=-1)
