import math
import numpy as np

def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm != 0:
        return vector / norm
    return vector

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction
    
    def position(self, time: float) -> np.ndarray:
        return self.origin + self.direction * time
    
def hit_sphere(center: np.ndarray, radius: float, ray: Ray) -> float | None:
    o_minus_c = ray.origin - center
    a = np.dot(ray.direction, ray.direction)
    b = 2.0 * np.dot(o_minus_c, ray.direction)
    c = np.dot(o_minus_c, o_minus_c) - radius * radius

    b_squared_minus_4ac = b * b - 4 * a * c
    if b_squared_minus_4ac < 0:
        return None
    
    solution = (-b - math.sqrt(b_squared_minus_4ac)) / (2 * a)
    if solution > 0:
        return solution
    solution = (-b + math.sqrt(b_squared_minus_4ac)) / (2 * a)
    if solution > 0:
        return solution
    return None

def ray_to_sky_color(ray: Ray) -> np.ndarray:

    random_sphere_center = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    random_sphere_radius = 0.5

    time_hit_sphere = hit_sphere(random_sphere_center, random_sphere_radius, ray)

    if time_hit_sphere:
        position_hit_sphere = ray.position(time_hit_sphere)
        normal = normalize(position_hit_sphere - random_sphere_center)
        return 0.5 * (normal + 1.0)

    unit_direction = normalize(ray.direction)
    blueness = .5 * (unit_direction[1] + 1.0)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    blue  = np.array([0.5, 0.7, 1.0], dtype=np.float32)
    return blueness * blue + (1 - blueness) * white