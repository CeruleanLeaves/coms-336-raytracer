import math
import numpy as np

from aabb import Axis_Aligned_Bounding_Box
from hit_record import HitRecord
from materials import Material
from ray import Ray, normalize


class MovingSphere:
    def __init__(
    self,
    center0: np.ndarray,
    center1: np.ndarray,
    time0: float,
    time1: float,
    radius: float,
    material: Material,
    ):
        self.center0 = center0
        self.center1 = center1
        self.time0 = time0
        self.time1 = time1
        self.radius = radius
        self.material = material

    def current_center(self, time: float) -> np.ndarray:
        if self.time0 == self.time1:
            return self.center0
        relative_time = (time - self.time0) / (self.time1 - self.time0)
        return self.center0 + relative_time * (self.center1 - self.center0)
    
    def hit(self, ray: Ray, time_min, time_max: float):
        center = self.current_center(ray.time)
        o_minus_c = ray.origin - center

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
        outward_normal = normalize(hit_point - center)

        front_face = np.dot(ray.direction, outward_normal) < 0.0
        normal = outward_normal if front_face else -outward_normal

        return HitRecord(
            time=hit_time,
            point=hit_point,
            normal=normal,
            material=self.material,
            front_face=front_face,
        )
    
    def bounding_box(self) -> Axis_Aligned_Bounding_Box:
        radius_vector = np.array([self.radius, self.radius, self.radius], dtype=np.float32)
        box0 = Axis_Aligned_Bounding_Box(self.center0 - radius_vector, self.center0 + radius_vector)
        box1 = Axis_Aligned_Bounding_Box(self.center1 - radius_vector, self.center1 + radius_vector)
        return Axis_Aligned_Bounding_Box.surrounding_box(box0, box1)