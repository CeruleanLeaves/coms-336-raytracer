import numpy as np
import math

def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm != 0:
        return vector / norm
    return vector

def reflect(unit_vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return unit_vector - 2.0 * np.dot(unit_vector, normal) * normal 

def refract(unit_vector: np.ndarray, normal: np.ndarray, refraction_index_ratio: float) -> np.ndarray:
    cos_theta = min(np.dot(-unit_vector, normal), 1.0)
    res_vector_perpendicular = (unit_vector + cos_theta * normal) + refraction_index_ratio
    res_vector_parallel = normal * -math.sqrt(max(0.0, 1.0 - np.dot(res_vector_perpendicular, res_vector_perpendicular)))
    return res_vector_parallel + res_vector_perpendicular

def schlick(cosine: float, refraction_index: float) -> float:
    r0 = (1.0 - refraction_index) / (1.0 + refraction_index) ** 2
    return r0 + (1.0 - r0) * ((1.0 - cosine) ** 5)

def random_point_in_unit_sphere() -> np.ndarray:
    while True:
        point = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
        if np.dot(point, point) < 1.0:
            return point

def random_unit_vector() -> np.ndarray:
    return normalize(random_point_in_unit_sphere())
    
class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction
    
    def position(self, time: float) -> np.ndarray:
        return self.origin + self.direction * time

