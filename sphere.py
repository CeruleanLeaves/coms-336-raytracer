import numpy as np
import math
from hit_record import HitRecord
from ray import Ray, normalize

class Sphere:
    def __init__(self, center: np.ndarray, radius: float, base_color: np.ndarray, is_mirror: bool = False):
        self.center = center
        self.radius = radius
        self.base_color = base_color
        self.is_mirror = is_mirror
    
    def hit(self, ray: Ray, time_min: float, time_max: float) -> HitRecord | None:
        o_minus_c = ray.origin - self.center

        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(ray.direction, o_minus_c)
        c = np.dot(o_minus_c, o_minus_c) - self.radius * self.radius

        b_squared_minus_4ac = b * b - 4 * a * c
        if b_squared_minus_4ac < 0:
            return None
        
        hit_time = (-b - math.sqrt(b_squared_minus_4ac)) / (2 * a)
        if not (time_min <= hit_time <= time_max):
            hit_time = (-b + math.sqrt(b_squared_minus_4ac)) / (2 * a)
            if not (time_min <= hit_time <= time_max):
                return None
        
        hit_point = ray.position(hit_time)
        normal = normalize(hit_point - self.center)

        return HitRecord(
            time=hit_time,
            point=hit_point,
            normal=normal,
            base_color=self.base_color,
            is_mirror=self.is_mirror
        )