import numpy as np

from aabb import Axis_Aligned_Bounding_Box
from hit_record import HitRecord
from materials import Material
from ray import Ray, normalize


class Triangle:
    def __init__(self,
                 v0: np.ndarray,
                 v1: np.ndarray,
                 v2: np.ndarray,
                 material: Material):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.e1 = v1 - v0
        self.e2 = v2 - v0
        self.material = material
        self.normal = np.cross(v1 - v0, v2 - v0)
    
    def hit(self, ray: Ray, time_min: float, time_max: float) -> HitRecord | None:
        #triple product scaler
        ray_is_parallel = abs(np.dot(-ray.direction, np.cross(self.e1, self.e2))) < 1e-6
        if ray_is_parallel:
            return None
        
        #cramers rule
        #solving uE1 + vE2 - time * direction = (ray_origin - v0)
        v0_to_ray_origin = ray.origin - self.v0
        cramer_denominator = np.dot(-ray.direction, np.cross(self.e1, self.e2))

        u_cramer_numerator = np.dot(v0_to_ray_origin, np.cross(self.e2, -ray.direction))
        u = u_cramer_numerator / cramer_denominator
        if not (0.0 <= u <= 1.0):
            return None
        
        v_cramer_numerator = np.dot(self.e1, np.cross(v0_to_ray_origin, -ray.direction))
        v = v_cramer_numerator / cramer_denominator
        if not (0.0 <= v <= 1.0):
            return None
        
        if u + v > 1.0:
            return None
        
        t_cramer_numerator = np.dot(self.e1, np.cross(self.e2, v0_to_ray_origin))
        time = t_cramer_numerator / cramer_denominator
        if not (time_min <= time <= time_max):
            return None
        
        point = ray.position(time)

        outward_normal = normalize(self.normal)
        front_face = np.dot(ray.direction, outward_normal) < 0.0
        normal = outward_normal if front_face else -outward_normal

        return HitRecord(
            time=time,
            point=point,
            normal=normal,
            material=self.material,
            front_face=front_face
        )
    
    def bounding_box(self) -> Axis_Aligned_Bounding_Box:
        minimum_vertice = np.minimum(np.minimum(self.v0, self.v1), self.v2)
        maximum_vertice = np.maximum(np.maximum(self.v0, self.v1), self.v2)
        #padding if one of the x y or z of the triangle vertices are all the same
        padding = 1e-3
        minimum_vertice -= padding
        maximum_vertice += padding
        return Axis_Aligned_Bounding_Box(minimum_vertice, maximum_vertice)