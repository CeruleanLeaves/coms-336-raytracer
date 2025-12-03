import numpy as np
from materials import Material
from ray import Ray
from triangle import Triangle


class Mesh:
    def __init__(self, triangles: list[Triangle]):
        self.triangles = triangles

    @classmethod
    def from_vertices_indices(cls, vertices: np.ndarray, indices: list[tuple[int, int, int]], material: Material):
        triangles = []
        for i0, i1, i2, in indices:
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
            #make sure front face vertices are counter clockwise
            triangles.append(Triangle(v0, v1, v2, material)) 
        return cls(triangles)

    def hit(self, ray: Ray, time_min: float, time_max: float):
        closest_time = time_max
        hit_record = None

        for triangle in self.triangles:
            hit = triangle.hit(ray, time_min, time_max)
            if hit and hit.time < closest_time:
                closest_time = hit.time
                hit_record = hit

        return hit_record