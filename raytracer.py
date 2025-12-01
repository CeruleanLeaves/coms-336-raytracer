import numpy as np
from PIL import Image

from camera import Camera
from ray import Ray, ray_to_sky_color

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


    for row in range(height):
        for col in range(width):
            pixel_color = np.array([0, 0, 0], dtype=np.float32)
            for _ in range(samples_per_pixel):
                horizontal_percentage = (col + np.random.rand()) / (width - 1)
                vertical_percentage = (height - 1 - row + np.random.rand()) / (height - 1)

                ray = camera.get_ray(horizontal_percentage, vertical_percentage)
                pixel_color += ray_to_sky_color(ray)

            pixel_color /= samples_per_pixel    
            image[row, col, :] = pixel_color

    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    rendered_image = Image.fromarray(image_uint8, mode='RGB')
    rendered_image.save('output.png')
    print('yayy')

if __name__ == '__main__':
    main()