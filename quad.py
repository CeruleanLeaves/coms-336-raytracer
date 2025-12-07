import numpy as np

from aabb import Axis_Aligned_Bounding_Box
from materials import Material
from ray import Ray
from triangle import Triangle


class Quad:
    def __init__(
        self,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        material: Material,
        texture_xy0: np.ndarray | None = None,
        texture_xy1: np.ndarray | None = None,
        texture_xy2: np.ndarray | None = None,
        texture_xy3: np.ndarray | None = None,
    ):
        self.triangle1 = Triangle(v0, v1, v2, material, texture_xy0, texture_xy1, texture_xy2)
        self.triangle2 = Triangle(v0, v2, v3, material, texture_xy0, texture_xy2, texture_xy3)
        self.material = material

    def hit(self, ray: Ray, time_min: float, time_max: float):
        hit1 = self.triangle1.hit(ray, time_min, time_max)
        if hit1 is not None:
            time_max = hit1.time
        hit2 = self.triangle2.hit(ray, time_min, time_max)
        if hit2 is not None and (hit1 is None or hit2.time < hit1.time):
            return hit2
        return hit1
    
    def bounding_box(self) -> Axis_Aligned_Bounding_Box:
        box1 = self.triangle1.bounding_box()
        box2 = self.triangle2.bounding_box()
        return  Axis_Aligned_Bounding_Box.surrounding_box(box1, box2)