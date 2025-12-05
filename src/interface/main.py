from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.orbits import KeplerianOrbit, OrbitType, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.time import AbsoluteDate, TimeScalesFactory

from dynamics.classes import Body
from dynamics.orekit.constants import EARTH_RADIUS, MU, ATTITUDE, INERTIAL_FRAME

from orekit import JArray_double
import numpy as np
import matplotlib.pyplot as plt
from math import radians
import sys
import pygame
from play3d.models import Grid, CircleChain, Plot
from interface.interface_models import Trail, Point, EarthSurface
from interface.pygame_utils import handle_camera_with_keys
from play3d.three_d import Device, Camera
import play3d.three_d as three_d
from play3d.matrix import Matrix

class Interface:

    def __init__(self, params: dict = {}, bodies: list[Body] = [], initial_epoch: AbsoluteDate = None):

        self.params = params
        self.bodies = bodies
        # self.initial_epoch = initial_epoch or AbsoluteDate()
        if initial_epoch:
            self.initial_epoch = AbsoluteDate(
                int(initial_epoch["year"]),
                int(initial_epoch["month"]),
                int(initial_epoch["day"]),
                int(initial_epoch["hour"]),
                int(initial_epoch["minute"]),
                float(initial_epoch["second"]),
                TimeScalesFactory.getUTC())
        else:
            # current date and time in UTC
            self.initial_epoch = AbsoluteDate()

        default_params = {
            "zoom": 5.0,
            "background": {
                "color": (20, 20, 20),
            },
            "earth": {
                "show": True,
                "color": (0, 0, 255),
                "resolution": 70,
            },
            "equator_grid": {
                "show": False,
                "color": (0, 204, 204),
                "resolution": 10,
            },
            "timestamp": {
                "show": True,
                "size": 12,
                "color": (255, 255, 255),
            },
            "bodies": {
                "show": True,
                "show_label": True,
                "show_velocity": False,
                "show_trail": False,
                "show_thrust": True,
                "trail_last_steps": 100,
                "color_body": (255, 255, 255),
                "color_label": (255, 255, 255),
                "color_velocity": (0, 255, 255),
                "color_trail": (255, 255, 255),
                "color_thrust": (0, 255, 0),
            }
        }

        for param in default_params:
            if param not in self.params:
                self.params[param] = default_params[param]

        scale = params['zoom']
        self.distance_scale = float(scale / EARTH_RADIUS)
        self.scale = scale

        Device.viewport(1024, 768)
        pygame.init()
        pygame.display.set_caption("OrbitZoo")
        pygame.display.set_icon(pygame.image.load(".\src\interface\logo_orbitzoo.png"))
        self.screen = pygame.display.set_mode(Device.get_resolution())

        if self.params['timestamp']['show']:
            self.font = pygame.font.SysFont("Consolas", self.params["timestamp"]["size"])

        x, y, z = 0, 1, 2
        line_adapter = lambda p1, p2, color: pygame.draw.line(self.screen, color, (p1[x], p1[y]), (p2[x], p2[y]), 1)
        put_pixel = lambda x, y, color: pygame.draw.circle(self.screen, color, (x, y), 1)
        Device.set_renderer(put_pixel, line_renderer=line_adapter)

        if self.params["earth"]["show"]:
            # self.earth = Earth(resolution = self.params["earth"]["resolution"], position=(0,0,0), color=self.params["earth"]["color"], scale=scale)
            self.earth = EarthSurface(n_points = 10000, position=(0,0,0), color=self.params["earth"]["color"], scale=scale)
            # self.earth = EarthContinents(position=(0,0,0), color=self.params["earth"]["color"], scale=scale)
        if self.params["equator_grid"]["show"]:
            self.grid = Grid(color=self.params["equator_grid"]["color"], dimensions=(6*int(scale), 6*int(scale)), dot_precision=self.params["equator_grid"]["resolution"])
        if "orbits" in params:
            self.orbits = []
            # propagate each orbit a full period to get the trail
            for orbit in self.params["orbits"]:
                current_date = AbsoluteDate()
                keplerian_orbit = KeplerianOrbit(orbit["a"] + EARTH_RADIUS, orbit["e"], radians(orbit["i"]), radians(orbit["pa"]), radians(orbit["raan"]), radians(0.0), PositionAngleType.MEAN, INERTIAL_FRAME, current_date, MU)
                state = SpacecraftState(keplerian_orbit, 1.0)
                tolerances = NumericalPropagator.tolerances(60.0, state.getOrbit(), state.getOrbit().getType())
                integrator = DormandPrince853Integrator(1e-3, 500.0, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
                integrator.setInitialStepSize(10.0)
                propagator = NumericalPropagator(integrator)
                propagator.setMu(MU)
                propagator.setOrbitType(OrbitType.KEPLERIAN)
                propagator.setAttitudeProvider(ATTITUDE)
                propagator.addForceModel(NewtonianAttraction(MU))
                propagator.setInitialState(state)
                trail = []
                step_seconds = 30.0
                for _ in range(int(state.getKeplerianPeriod() / step_seconds) + 2):
                    current_date = current_date.shiftedBy(step_seconds)
                    state = propagator.propagate(current_date)
                    pos = state.getPVCoordinates().getPosition()
                    pos = [pos.getX(), pos.getZ(), pos.getY()]
                    if len(trail) == 0:
                        trail.append(pos)
                    else:
                        relative_pos = np.array(pos) - np.array(trail[0])
                        trail.append(list(relative_pos))
                initial_pos = list(np.array(trail[0]) * self.distance_scale)
                self.orbits.append(Trail(trail = trail[1:], position=initial_pos, color=orbit["color"], scale=self.distance_scale))

        self.spheres = []
        for body in self.bodies:
            scaled_pos = body.position * self.distance_scale
            x, y, z = scaled_pos
            if "body" in self.params and body.name in self.params["body"] and "color_body" in self.params["body"][body.name]:
                body_color = self.params["body"][body.name]["color_body"]
            else:
                body_color = self.params["bodies"]["color_body"]
            # self.spheres.append(CircleChain(resolution= 1, position=(x, z, y), color=body_color, scale=scale/300))
            self.spheres.append(Point(resolution=1, position=(x, z, y), color=body_color, scale=scale/300))
            # pos = body.current_state.getPVCoordinates().getPosition()
            # pos = pos.scalarMultiply(float(self.distance_scale))
            # if isinstance(body, Satellite):
            #     self.spheres.append(CircleChain(position=(pos.getX(), pos.getZ(), pos.getY()), color=params["satellites"]["color_body"] if body.color is None else body.color, scale=scale/300))
            # else:
            #     self.spheres.append(CircleChain(position=(pos.getX(), pos.getZ(), pos.getY()), color=params["drifters"]["color_body"] if body.color is None else body.color, scale=scale/300))

        camera = Camera.get_instance()
        camera.move(y=0.5, z=5+scale)
        # camera.move(y=1, z=5+2*scale)
        # camera.move(y=70)
        # camera.rotate('x', 90)

        self.reset()

    def add_bodies(self, bodies = [], type = None):
        for body in bodies:
            scaled_pos = body.position * self.distance_scale
            x, y, z = scaled_pos
            # if "body" in self.params and body.name in self.params["body"] and "color_body" in self.params["body"][body.name]:
            #     body_color = self.params["body"][body.name]["color_body"]
            # else:
            #     body_color = self.params["bodies"]["color_body"]
            if type == 'debris':
                body_color = (255, 0, 0)
            else:
                body_color = (0, 0, 0)
            # self.spheres.append(CircleChain(resolution= 1, position=(x, z, y), color=body_color, scale=scale/300))
            self.spheres.append(Point(resolution=1, position=(x, z, y), color=body_color, scale=self.scale/300))
            self.bodies.append(body)

    def reset(self):
        self.initial_timestamp = self.initial_epoch.getDate().toString().rpartition('.')[0].replace('T', ' ')
        # self.initial_timestamp = self.initial_epoch.getDate().toString()[:4]
        if len(self.bodies) > 0:
            self.trails = [[] for _ in self.bodies]

    def save_screenshot(self, filename = 'screenshot.jpg'):
        pygame.image.save(self.screen, filename)

    def frame(self, epoch: AbsoluteDate, save_path = None, info = None):

        save_points = save_path != None
        projected_points = {}

        if pygame.event.get(pygame.QUIT):
            sys.exit(0)

        # background color
        self.screen.fill(self.params["background"]["color"])

        # draw earth
        if self.params["earth"]["show"]:
            points_model = self.earth.draw()
            if save_points:
                projected_points['earth'] = {'points': np.delete(points_model, 2, axis=1), 'color': self.earth.color}

        # draw equator grid
        if self.params["equator_grid"]["show"]:
            points_model = self.grid.draw()
            if save_points:
                projected_points['equator_grid'] = {'points': np.delete(points_model, 2, axis=1), 'color': self.grid.color}

        # draw orbits
        if "orbits" in self.params:
            counter = 1
            for orbit in self.orbits:
                points_model = orbit.draw()
                if save_points:
                    projected_points[f'orbit_{counter}'] = {'points': np.delete(points_model, 2, axis=1), 'color': orbit.color}
                counter += 1
                

        # Cache the view-projection matrix outside the loop
        VP = three_d.Camera.View_Projection_matrix()

        # Cache distance_scale once
        distance_scale = self.distance_scale

        # Cache font to avoid repeated attribute lookup
        if self.params["timestamp"]["show"]:
            font = self.font

        # Pre-render timestamp texts
        if len(self.bodies) > 0:

            if self.params["timestamp"]["show"]:
                # text_year = f'Year:     {epoch.getDate().toString()[:4]}'
                # label_year = font.render(text_year, True, self.params["timestamp"]["color"])
                # rect_year = label_year.get_rect()
                # rect_year.topleft = (10, 10)
                
                # text_payloads = f'Payloads: {info["total_payloads"]}'
                # label_payloads = font.render(text_payloads, True, self.params["timestamp"]["color"])
                # rect_payloads = label_payloads.get_rect(topleft=(10, rect_year.bottom + 5))

                # text_debris = f'Debris:   {info["total_debris"]}'
                # label_debris = font.render(text_debris, True, (255, 0, 0))
                # rect_debris = label_debris.get_rect(topleft=(10, rect_payloads.bottom + 5))

                text_start = f'Start:   {self.initial_timestamp}'
                label_start = font.render(text_start, True, (255, 255, 255))
                # rect_start = label_start.get_rect(center=(110, 20))
                rect_start = label_start.get_rect()
                rect_start.topleft = (10, 10)
                current_timestamp = epoch.getDate().toString().rpartition('.')[0].replace('T', ' ')
                text_current = f'Current: {current_timestamp}'
                label_current = font.render(text_current, True, (255, 255, 255))
                # rect_current = label_current.get_rect(center=(110, 35))
                rect_current = label_current.get_rect(topleft=(10, rect_start.bottom + 5))

            positions = np.array([body.position for body in self.bodies])  # shape (N,3)
            velocities = np.array([body.velocity for body in self.bodies]) # shape (N,3)
            scaled_positions = positions * distance_scale

            # normalize positions
            r_norm = np.linalg.norm(positions, axis=1)
            r = positions / r_norm[:, None]

            # w = normalized cross(r, v)
            w = np.cross(r, velocities)
            w /= np.linalg.norm(w, axis=1)[:, None]

            # s = normalized cross(w, r)
            s = np.cross(w, r)
            s /= np.linalg.norm(s, axis=1)[:, None]

            r[:, [1, 2]] = r[:, [2, 1]]
            w[:, [1, 2]] = w[:, [2, 1]]
            s[:, [1, 2]] = s[:, [2, 1]]

            #positions2 = positions
            # positions2[:, [1, 2]] = positions2[:, [2, 1]]

            counter = 1
            for i, body in enumerate(self.bodies):

                show_body = ("body" in self.params and body.name in self.params["body"] and "show" in self.params["body"][body.name] and self.params["body"][body.name]["show"]) or self.params["bodies"]["show"]

                # skip body if we do not want to show it
                if not show_body:
                    continue

                show_label = self.params["body"][body.name]["show_label"] if "body" in self.params and body.name in self.params["body"] and "show_label" in self.params["body"][body.name] else self.params["bodies"]["show_label"]
                show_velocity = self.params["body"][body.name]["show_velocity"] if "body" in self.params and body.name in self.params["body"] and "show_velocity" in self.params["body"][body.name] else self.params["bodies"]["show_velocity"]
                show_thrust = self.params["body"][body.name]["show_thrust"] if "body" in self.params and body.name in self.params["body"] and "show_thrust" in self.params["body"][body.name] else self.params["bodies"]["show_thrust"]
                show_trail = self.params["body"][body.name]["show_trail"] if "body" in self.params and body.name in self.params["body"] and "show_trail" in self.params["body"][body.name] else self.params["bodies"]["show_trail"]

                scaled_pos = scaled_positions[i]
                x, y, z = scaled_pos

                # draw body
                sphere = self.spheres[i]
                sphere.set_position(x, z, y)
                points_model = sphere.draw()
                if save_points:
                        points_model = Matrix([[0, 0, 0, 1]]) @ sphere.matrix @ VP
                        points_model = sphere._perspective_divide(points=points_model)
                        projected_points[f'body_{counter}'] = {'points': np.delete(points_model, 2, axis=1), 'color': sphere.color}

                # draw body velocity
                if show_velocity and np.any(body.velocity):
                    color_velocity = self.params["body"][body.name]["color_velocity"] if "body" in self.params and body.name in self.params["body"] and "color_velocity" in self.params["body"][body.name] else self.params["bodies"]["color_velocity"]
                    velocity_arrow = Plot(func=lambda t: self.arrow_func(t, s[i], scale=0.3), allrange=[0, 1], position=(x, z, y), color=color_velocity, interpolate=50)
                    points_model = velocity_arrow.draw()
                    if save_points:
                        projected_points[f'velocity_{counter}'] = {'points': np.delete(points_model, 2, axis=1), 'color': color_velocity}

                # draw body thrust
                if show_thrust and np.any(body.thrust):
                    color_thrust = self.params["body"][body.name]["color_thrust"] if "body" in self.params and body.name in self.params["body"] and "color_thrust" in self.params["body"][body.name] else self.params["bodies"]["color_thrust"]
                    thrust = body.thrust
                    thrust = r[i] * thrust[0] + s[i] * thrust[1] + w[i] * thrust[2]
                    thrust = thrust / np.linalg.norm(thrust)
                    thrust_arrow = Plot(func=lambda t: self.arrow_func(t, thrust, scale=0.2), allrange=[0, 1], position=(x, z, y), color=color_thrust, interpolate=50)
                    points_model = thrust_arrow.draw()
                    if save_points:
                        projected_points[f'thrust_{counter}'] = {'points': np.delete(points_model, 2, axis=1), 'color': color_thrust}

                # draw trail
                if show_trail:
                    trail_last_steps = self.params["body"][body.name]["trail_last_steps"] if "body" in self.params and body.name in self.params["body"] and "trail_last_steps" in self.params["body"][body.name] else self.params["bodies"]["trail_last_steps"]
                    color_trail = self.params["body"][body.name]["color_trail"] if "body" in self.params and body.name in self.params["body"] and "color_trail" in self.params["body"][body.name] else self.params["bodies"]["color_trail"]
                    body_trail = self.trails[i]
                    if len(body_trail) > trail_last_steps:
                        body_trail.pop(1)
                    pos = positions[i]
                    pos[[1, 2]] = pos[[2, 1]]
                    if len(body_trail) == 0:
                        body_trail.append(list(pos))
                    else:
                        relative_pos = pos - np.array(body_trail[0])
                        body_trail.append(list(relative_pos))
                    initial_pos = list(np.array(body_trail[0]) * self.distance_scale)
                    trail = Trail(trail = body_trail[1:], position=initial_pos, color=color_trail, scale=self.distance_scale)
                    points_model = trail.draw()
                    if save_points and np.any(points_model):
                            projected_points[f'trail_{counter}'] = {'points': np.delete(points_model, 2, axis=1), 'color': color_trail}

                # draw label if name exists
                if show_label and body.name:
                    color_label = self.params["body"][body.name]["color_label"] if "body" in self.params and body.name in self.params["body"] and "color_label" in self.params["body"][body.name] else self.params["bodies"]["color_label"]
                    
                    # Use a scaled position matrix multiply only once
                    data = Matrix([[x*1/40, y*1/40, z*1/40, 1]])
                    points = data @ sphere.matrix @ VP
                    center = sphere._perspective_divide(points=points)[0][:2]

                    # Cache label text per body name to avoid re-rendering every frame
                    if not hasattr(body, '_cached_label') or body._cached_label_text != body.name:
                        body._cached_label = font.render(body.name, True, color_label)
                        body._cached_label_text = body.name

                    label = body._cached_label
                    label_rect = label.get_rect(center=(center[0], center[1] - 20))
                    self.screen.blit(label, label_rect)

                    if save_points:
                        projected_points[f'body_{counter}']['name'] = body.name

                counter += 1

            # Blit timestamp labels
            if self.params["timestamp"]["show"]:
                self.screen.blit(label_start, rect_start)
                self.screen.blit(label_current, rect_current)
            # self.screen.blit(label_year, rect_year)
            # self.screen.blit(label_payloads, rect_payloads)
            # self.screen.blit(label_debris, rect_debris)

        handle_camera_with_keys()
        pygame.display.update()

        # camera = Camera.get_instance()
        # print(camera)
        # print(f'{camera['x']}, {camera['y']}, {camera['z']}')

        if save_points:
            # print(projected_points)

            # default_config = {
            #     "earth_point_size": 0.2,
            #     "body_point_size": 0.2,
            #     "body_thrust_point_size": 0.2,
            #     "body_trail_point_size": 0.2,
            # }

            fig, ax = plt.subplots(figsize=(8, 8))

            for key, data in projected_points.items():
                arr = data['points']
                color = tuple(c / 255 for c in data['color'])  # Normalize RGB
                x, y = arr[:, 0], arr[:, 1]

                # Use larger marker for single-point bodies
                size = 20 if len(arr) == 1 else 0.2
                # size = 40 if len(arr) == 1 else 0.2
                ax.scatter(x, y, s=size, color=color)

                # Add name slightly above the body (only for single points)
                if 'name' in data and len(arr) == 1:
                    name = data['name']
                    ax.text(
                        x[0], y[0] - 10, name,
                        color=(0,0,0),
                        fontsize=8,
                        # fontsize=30,
                        ha='center', va='bottom',
                        fontweight='bold'
                    )

            # Clean look
            ax.set_axis_off()
            ax.set_aspect('equal')
            ax.invert_yaxis()

            plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()



    def frame_old(self, epoch: AbsoluteDate):

        if pygame.event.get(pygame.QUIT):
            sys.exit(0)

        self.screen.fill((20, 20, 20))
        # self.screen.fill((255, 255, 255))

        # draw grid
        if self.params["equator_grid"]["show"]:
            self.grid.draw()

        # draw Earth
        if self.params["earth"]["show"]:
            self.earth.draw()

        # draw orbits
        # if "orbits" in self.params:
        #     for orbit in self.orbits:
        #         orbit.draw()

        # draw bodies
        for i, body in enumerate(self.bodies):

            # body_type = "satellites" if body.has_thrust else "drifters"

            pos = np.array(body.position)
            vel = np.array(body.velocity)

            r = self.normalize(pos)          # normalized position vector
            w = self.normalize(np.cross(r, vel))  # normalized cross product of r and v
            s = self.normalize(np.cross(w, r))  # normalized cross product of w and r

            scaled_pos = pos * float(self.distance_scale)
            x, y, z = scaled_pos

            # draw body
            # if self.params[body_type]["show"]:
            self.spheres[i].set_position(x, y, z)
            self.spheres[i].draw()

            # draw body velocity
            # if self.params[body_type]["show_velocity"]:
            #     vel_vector = [s.getX(), s.getZ(), s.getY()]
            #     velocity_arrow = Plot(func=lambda t: self.arrow_func(t, vel_vector, scale=0.3), allrange=[0, 1], position=(x, y, z), color=self.params["satellites"]["color_velocity"], interpolate=50)
            #     velocity_arrow.draw()

            # draw body thrust
            # if body_type == "satellites" and self.params[body_type]["show_thrust"] and body.thrust:
            #     thrust = Vector3D.cast_(Vector.cast_(r.scalarMultiply(float(body.thrust[0])).add(s.scalarMultiply(float(body.thrust[1]))).add(w.scalarMultiply(float(body.thrust[2])))).normalize())
            #     thrust_vector = [thrust.getX(), thrust.getZ(), thrust.getY()]
            #     thrust_arrow = Plot(func=lambda t: self.arrow_func(t, thrust_vector, scale=0.2), allrange=[0, 1], position=(x, y, z), color=self.params["satellites"]["color_thrust"], interpolate=50)
            #     thrust_arrow.draw()

            # scaled_pos = pos.scalarMultiply(float(self.distance_scale))
            # x, y, z = scaled_pos.getX(), scaled_pos.getZ(), scaled_pos.getY()

            # r = Vector3D.cast_(Vector.cast_(pos).normalize())
            # v = Vector3D.cast_(Vector.cast_(vel))
            # w = Vector3D.cast_(Vector.cast_(r.crossProduct(v)).normalize())
            # s = Vector3D.cast_(Vector.cast_(w.crossProduct(r)).normalize())

            # draw trail
            # if self.params[body_type]["show_trail"]:
            #     if len(self.trails[body.name]) > self.params[body_type]["trail_last_steps"]:
            #         self.trails[body.name].pop(1)
            #     pos = [pos.getX(), pos.getZ(), pos.getY()]
            #     if len(self.trails[body.name]) == 0:
            #         self.trails[body.name].append(pos)
            #     else:
            #         relative_pos = np.array(pos) - np.array(self.trails[body.name][0])
            #         self.trails[body.name].append(list(relative_pos))
            #     initial_pos = list(np.array(self.trails[body.name][0]) * self.distance_scale)
            #     trail = Trail(trail = self.trails[body.name][1:], position=initial_pos, color=self.params[body_type]["color_trail"] if body.color is None else body.color, scale=self.distance_scale)
            #     trail.draw()

            # draw body label
            if body.name is not None:
            # if self.params[body_type]["show_label"]:
                VP = three_d.Camera.View_Projection_matrix()
                # data = Matrix([[x, y, z, 1]])
                data = Matrix([[x*1/40, y*1/40, z*1/40, 1]])
                points = data @ self.spheres[i].matrix @ VP
                center = self.spheres[i]._perspective_divide(points=points)[0][:2]
                label = self.font.render(body.name, True, (255, 255, 255))
                # label = self.font.render(body.name, True, self.params[body_type]["color_label"])
                label_rect = label.get_rect(center=(center[0], center[1] - 20))
                self.screen.blit(label, label_rect)

        # draw timestamp
        if True:
            # initial timestamp
            text_timestamp = f'Start:   {self.initial_timestamp}'
            label_timestamp = self.font.render(text_timestamp, True, (255, 255, 255))
            # label_timestamp = self.font.render(text_timestamp, True, (0, 0, 0))
            label_rect_timestamp = label_timestamp.get_rect(center=(110, 20))
            self.screen.blit(label_timestamp, label_rect_timestamp)

            # current timestamp
            current_timestamp = epoch.getDate().toString().rpartition('.')[0].replace('T', ' ')
            text_timestamp = f'Current: {current_timestamp}'
            label_timestamp = self.font.render(text_timestamp, True, (255, 255, 255))
            label_rect_timestamp = label_timestamp.get_rect(center=(110, 35))
            self.screen.blit(label_timestamp, label_rect_timestamp)

        handle_camera_with_keys()
        pygame.display.update()
        # pygame.time.delay(self.params["delay_ms"])

    def arrow_func(self, t, vector, scale=1):
        return [t * component * scale for component in vector]
    