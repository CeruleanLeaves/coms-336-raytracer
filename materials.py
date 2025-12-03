import math
import numpy as np

from hit_record import HitRecord
from ray import Ray, normalize, random_point_in_unit_sphere, random_unit_vector, reflect, refract, schlick


class Material:
    def scatter(self, incoming_ray: Ray, hit_record: HitRecord):
        raise NotImplementedError
    
    def emitted(self):
        return np.zeros(3, dtype=np.float32)
    
class Lambertian(Material):
    def __init__(self, base_color: np.ndarray):
        self.base_color = base_color
    
    def scatter(self, incoming_ray: Ray, hit_record: HitRecord):
        scatter_direction = hit_record.normal + random_unit_vector()
        if np.allclose(scatter_direction, 0.0):
            scatter_direction = hit_record.normal
        
        scatter_ray = Ray(hit_record.point + hit_record.normal * 1e-3, scatter_direction)
        attenuation = self.base_color
        return attenuation, scatter_ray

class Metal(Material):
    def __init__(self, base_color: np.ndarray, fuzz: float = 0.0):
        self.base_color = base_color
        self.fuzz = min(fuzz, 1.0)

    def scatter(self, incoming_ray: Ray, hit_record: HitRecord):
        unit_incoming_direction = normalize(incoming_ray.direction)
        reflected_direction = reflect(unit_incoming_direction, hit_record.normal)
        fuzzy_reflected_direction = reflected_direction + self.fuzz * random_point_in_unit_sphere()
        scatter_ray = Ray(hit_record.point + hit_record.normal * 1e-3, fuzzy_reflected_direction)
        
        if np.dot(scatter_ray.direction, hit_record.normal) <= 0:
            return None, None
        
        attenuation = self.base_color
        return attenuation, scatter_ray
    
class Dielectric(Material):
    def __init__(self, refraction_index: float):
        self.refraction_index = refraction_index

    def scatter(self, incoming_ray: Ray, hit_record: HitRecord) -> tuple[np.ndarray, Ray]:
        unit_incoming_direction = normalize(incoming_ray.direction)
        
        if hit_record.front_face:
            refraction_index_ratio = 1.0 / self.refraction_index
        else:
            refraction_index_ratio = self.refraction_index
        
        cos_theta = min(np.dot(-unit_incoming_direction, hit_record.normal), 1.0)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta ** 2))

        do_not_refract = refraction_index_ratio * sin_theta > 1.0
        reflect_probability = schlick(cos_theta, self.refraction_index)

        use_reflect = do_not_refract or np.random.rand() < reflect_probability
        if use_reflect:
            direction = reflect(unit_incoming_direction, hit_record.normal)
        else:
            direction = refract(unit_incoming_direction, hit_record.normal, refraction_index_ratio)
        
        scatter_ray = Ray(hit_record.point + hit_record.normal * 1e-3, direction)
        attenuation = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return attenuation, scatter_ray
    
class Emissive(Material):
    def __init__(self, emit_color: np.ndarray):
        self.emit_color = emit_color

    def scatter(self, incoming_ray: Ray, hit_record: HitRecord) -> tuple[None, None]:
        return None, None
    
    def emitted(self):
        return self.emit_color