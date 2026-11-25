
import numpy as np
import pygame

from highway_env.envs.common.graphics import (
    EnvViewer,
    ObservationGraphics,
    WorldSurface,
)
from highway_env.road.graphics import RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics



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


         
class MultiAgentEnvViewer(EnvViewer):
    """Render multiple agent camera views in one window."""

    def __init__(self, env):
        super().__init__(env)

        self.render_agents = min(
        env.config["render_agents"],
        env.config["controlled_vehicles"],
        )

        self.render_surfaces = self.create_render_surface()

        
    def get_render_agents(self):
        # Return the first N controlled agents for rendering.
        return self.env.controlled_vehicles[:self.render_agents]

    def get_render_layout(self):
        # Calculate the number of rows and columns.
        columns = int(np.ceil(np.sqrt(self.render_agents)))
        rows = int(np.ceil(self.render_agents / columns))

        return rows, columns

    def get_render_view_size(self):
        # Divide the window into equally sized views.
        rows, columns = self.get_render_layout()
        width = self.config["screen_width"] // columns
        height = self.config["screen_height"] // rows

        return width, height

    def get_render_view_position(self, agent_idx):
        # Calculate the top-left position of an agent's view.
        rows, columns = self.get_render_layout()
        width, height = self.get_render_view_size()

        row = agent_idx // columns
        column = agent_idx % columns

        return column * width, row * height

    def create_render_surface(self):
        # Create a surface for each selected agent.
        width, height = self.get_render_view_size()

        return [
            WorldSurface(
                (width, height),
                0,
                pygame.Surface((width, height)),
            )
            for _ in self.get_render_agents()
        ]


    def get_render_camera_position(self, agent_idx):
        """Centre each camera on its own car, so the car and its outline stay in view."""
        vehicle = self.get_render_agents()[agent_idx]
        return np.array(vehicle.position)

    def get_render_view_rect(self, agent_idx):
        position = self.get_render_view_position(agent_idx)
        width, height = self.get_render_view_size()

        return (*position, width, height)

    def display_render_borders(self, screen):
        for agent_idx in range(self.render_agents):
            rect = self.get_render_view_rect(agent_idx)

            pygame.draw.rect(
                screen,
                (0,0,0),
                rect,
                2,
            )

    def update_render_cameras(self):
        # Center each camera on its assigned agent.
        for agent_idx, surface in enumerate(self.render_surfaces):
            surface.move_display_window_to(
                self.get_render_camera_position(agent_idx)
            )


    def display(self) -> None:
        """Display the road and vehicles using custom graphics."""

        if not self.enabled:
            return

        # Update camera positions for all selected agents.
        self.update_render_cameras()

        # Draw all elements on each agent's camera surface.
        for surface in self.render_surfaces:

            # Draw the road.
            RoadGraphics.display(self.env.road, surface)

            # Draw the vehicle trajectory, if one is available.
            if self.vehicle_trajectory:
                VehicleGraphics.display_trajectory(
                    self.vehicle_trajectory,
                    surface,
                    offscreen=self.offscreen,
                )

            # Draw road objects.
            RoadGraphics.display_road_objects(
                self.env.road,
                surface,
                offscreen=self.offscreen,
            )

            # Draw traffic, including attacked vehicle outlines.
            AttackedRoadGraphics.display_traffic(
                self.env.road,
                surface,
                simulation_frequency=self.env.config["simulation_frequency"],
                offscreen=self.offscreen,
            )

            # Draw observations.
            ObservationGraphics.display(
                self.env.observation_type,
                surface,
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
                # For portrait screens, position the agent visualization to the right.
                else:
                    self.screen.blit(
                        self.agent_surface,
                        (self.config["screen_width"], 0),
                    )

        # Update the screen when rendering to a visible display.
        if not self.offscreen:

            # Clear the window before drawing the camera views.
            self.screen.fill((0, 0, 0))

            # Display each agent's camera view in the main window.
            for agent_idx, surface in enumerate(self.render_surfaces):
                position = self.get_render_view_position(agent_idx)
                self.screen.blit(surface, position)

            # Draw borders around all camera views.
            self.display_render_borders(self.screen)

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
