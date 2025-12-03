import math
import numpy as np
from PIL import Image

from camera import Camera
from materials import Dielectric, Emissive, Lambertian, Metal
from ray import Ray, normalize, reflect, refract, schlick
from sphere import Sphere
from triangle import Triangle

MAX_DEPTH = 10

def ray_color(ray: Ray, world: list[Sphere], depth: int) -> np.ndarray:


    shortest_hit_time = float('inf')
    hit_record = None

    for object in world:
        hit = object.hit(ray, 1e-3, shortest_hit_time)
        if hit:
            shortest_hit_time = hit.time
            hit_record = hit
    
    if hit_record:

        emitted = hit_record.material.emitted()
        if depth >= MAX_DEPTH:
            return emitted
        
        attenuation, scatter_ray = hit_record.material.scatter(ray, hit_record)
        if attenuation is None or scatter_ray is None:
            return emitted
        return emitted + attenuation * ray_color(scatter_ray, world, depth + 1)

    unit_direction = normalize(ray.direction)
    blueness = .5 * (unit_direction[1] + 1.0)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    blue  = np.array([0.5, 0.7, 1.0], dtype=np.float32)
    return blueness * blue + (1 - blueness) * white

def main():

    width = 400
    height = 225
    samples_per_pixel = 20

    image = np.zeros(shape=(height, width, 3), dtype=np.float32)

    aspect_ratio = width / height
    camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    camera_look_at_point = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vertical_fov_degrees = 90.0

    camera = Camera(camera_position, camera_look_at_point, up_direction, vertical_fov_degrees, aspect_ratio)

    material_ground = Lambertian(np.array([0.8, 0.8, 0.0], dtype=np.float32))
    material_center = Dielectric(1.5)  # glass center
    material_left   = Lambertian(np.array([0.8, 0.3, 0.3], dtype=np.float32))
    material_right  = Metal(np.array([0.9, 0.9, 0.9], dtype=np.float32), fuzz=0.0)
    material_light = Emissive(np.array([4.0, 4.0, 4.0], dtype=np.float32))
    material_triangle = Lambertian(np.array([0.3, 0.5, 0.8], dtype=np.float32))
    world = [
        Sphere(
            center=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            radius=0.2,
            material=material_center
        ),
        Sphere(
            center=np.array([-1.0, 0.0, -2.0], dtype=np.float32),
            radius=0.4,
            material=material_left
        ),
        Sphere(
            center=np.array([1.0, 0.0, -2.0], dtype=np.float32),
            radius=0.5,
            material=material_right
        ),
        Sphere(
            center=np.array([0.0, -100.5, -1.0], dtype=np.float32),
            radius=100.0,
            material=material_ground
        ),
        Sphere(
            center=np.array([0.0, 3.0, -0.9], dtype=np.float32),
            radius=0.5,
            material=material_light,
        ),
        Triangle(
            v0=np.array([-3.0, -1.0, -3.0], dtype=np.float32),
            v1=np.array([ 3.0, -1.0, -3.0], dtype=np.float32),
            v2=np.array([ 0.0,  3.0, -3.0], dtype=np.float32),
            material=material_triangle,
        ),
    ]

    for row in range(height):
        for col in range(width):
            pixel_color = np.array([0, 0, 0], dtype=np.float32)
            for _ in range(samples_per_pixel):
                horizontal_percentage = (col + np.random.rand()) / (width - 1)
                vertical_percentage = (height - 1 - row + np.random.rand()) / (height - 1)

                ray = camera.get_ray(horizontal_percentage, vertical_percentage)
                pixel_color += ray_color(ray, world, 0)

            pixel_color /= samples_per_pixel    
            image[row, col, :] = pixel_color

    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    rendered_image = Image.fromarray(image_uint8, mode='RGB')
    rendered_image.save('output.png')
    print('yayy')

if __name__ == '__main__':
    main()