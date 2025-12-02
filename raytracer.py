import numpy as np
from PIL import Image

from camera import Camera
from ray import Ray, normalize, reflect
from sphere import Sphere

MAX_DEPTH = 10

def ray_color(ray: Ray, world: list[Sphere], depth: int) -> np.ndarray:

    if depth >= MAX_DEPTH:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    shortest_hit_time = float('inf')
    hit_record = None

    for sphere in world:
        hit = sphere.hit(ray, 1e-3, shortest_hit_time)
        if hit:
            shortest_hit_time = hit.time
            hit_record = hit
    
    if hit_record:
        normal = hit_record.normal
        material_base_color = hit_record.base_color
        
        if hit_record.is_mirror:
            incoming_direction = normalize(ray.direction)
            reflected_direction = reflect(incoming_direction, normal)
            reflected_origin = hit_record.point + normal * 1e-3
            reflected_ray = Ray(reflected_origin, reflected_direction)
            
            reflected_color = ray_color(reflected_ray, world, depth + 1)
            return np.clip(material_base_color * reflected_color, 0.0, 1.0)
        else:
            light_direction = normalize(np.array([1.0, 1.0, -0.5], dtype=np.float32))
            diffuse_intensity = max(np.dot(normal, light_direction), 0.0)
            ambient = 0.3
            color = ambient * material_base_color + diffuse_intensity * material_base_color
            color = np.clip(color, 0.0, 1.0)
            return color

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

    world = [
        Sphere(
            center=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            radius=0.2,
            base_color=np.array([0.8, 0.3, 0.3], dtype=np.float32),
            is_mirror=True
        ),
        Sphere(
            center=np.array([-1.0, 0.0, -2.0], dtype=np.float32),
            radius=0.4,
            base_color=np.array([0.3, 0.8, 0.3], dtype=np.float32),
        ),
        Sphere(
            center=np.array([1.0, 0.0, -2.0], dtype=np.float32),
            radius=0.5,
            base_color=np.array([0.3, 0.3, 0.8], dtype=np.float32),
        ),
        Sphere(
            center=np.array([0.0, -100.5, -1.0], dtype=np.float32),
            radius=100.0,
            base_color=np.array([0.8, 0.8, 0.0], dtype=np.float32),
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