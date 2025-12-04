import numpy as np
import math
from aabb import Axis_Aligned_Bounding_Box
from hit_record import HitRecord
from materials import Material
from ray import Ray, normalize

class Sphere:
    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 material: Material):
        self.center = center
        self.radius = radius
        self.material = material
    
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
        outward_normal = normalize(hit_point - self.center)

        front_face = np.dot(ray.direction, outward_normal) < 0.0
        normal = outward_normal if front_face else -outward_normal

        local_point = (hit_point - self.center) / self.radius
        texture_coordinates = self.sphere_texture_coordinates(local_point)

        return HitRecord(
            time=hit_time,
            point=hit_point,
            normal=normal,
            material=self.material,
            front_face=front_face,
            texture_coordinates=texture_coordinates
        )
    
    def bounding_box(self) -> Axis_Aligned_Bounding_Box:
        radius_vector = np.array([self.radius, self.radius, self.radius], dtype=np.float32)
        minimum_vertice = self.center - radius_vector
        maximum_vertice = self.center + radius_vector
        return Axis_Aligned_Bounding_Box(minimum_vertice, maximum_vertice)
    
    @staticmethod
    def sphere_texture_coordinates(local_point: np.ndarray) -> np.ndarray:
        x, y, z = local_point[0], local_point[1], local_point[2]
        theta = math.acos(-y)
        phi = math.atan2(-z, x) + math.pi
        texture_x = phi / (2.0 * math.pi)
        texture_y = theta / math.pi
        return np.array([texture_x, texture_y], dtype=np.float32)
