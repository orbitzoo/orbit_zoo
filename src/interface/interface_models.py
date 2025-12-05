from play3d.models import Model
from play3d.matrix import Matrix
import math
import numpy as np
import random
import trimesh

class Earth(Model):

    @classmethod
    def _fn(cls, phi, theta, flattening_factor=0.9966472):

        return [
            math.sin(phi * math.pi / 180) * math.cos(theta * math.pi / 180),
            math.sin(theta * math.pi / 180) * math.sin(phi * math.pi / 180) * flattening_factor,
            math.cos(phi * math.pi / 180)
        ]
    
    def __init__(self, resolution = 40, **kwargs):
        super(Earth, self).__init__(**kwargs)

        data = []
        for x in np.linspace(0, 360, resolution):
            for y in np.linspace(0, 360, resolution):
                data.append(Earth._fn(x, y) + [1])

        self.data = Matrix(data)

class EarthSurface(Model):

    @classmethod
    def fibonacci_sphere(cls, samples, flattening_factor=0.9966472):
        points = []
        phi_golden = (1 + 5 ** 0.5) / 2  # golden ratio

        for i in range(samples):
            # evenly spaced z
            z = 1 - 2 * i / (samples - 1)
            radius = math.sqrt(1 - z * z)

            theta = 2 * math.pi * i / phi_golden

            x = radius * math.cos(theta)
            y = radius * math.sin(theta) * flattening_factor
            points.append([x, y, z, 1])

        return points

    def __init__(self, n_points=1000, **kwargs):
        super().__init__(**kwargs)

        # Generate evenly distributed points on the sphere
        self.data = Matrix(self.fibonacci_sphere(n_points))
      
class Trail(Model):

    def __init__(self, trail = [], **kwargs):
        super(Trail, self).__init__(**kwargs)
        self.data = [point + [1] for point in trail]
        self.data = Matrix(self.data)

class Point(Model):
    
    def __init__(self, **kwargs):
        super(Point, self).__init__(**kwargs)
        self.data = [[0, 0, 0, 1]]
        self.data = Matrix(self.data)