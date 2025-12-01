import numpy as np
from ray import Ray, normalize

class Camera:
    def __init__(self, position: np.ndarray, look_at_point: np.ndarray,
                 up_direction: np.ndarray, vertical_fov_degrees: float, aspect_ratio: float):
        vertical_fov_radians = np.deg2rad(vertical_fov_degrees)
        viewport_height = np.tan(vertical_fov_radians / 2) * 2
        viewport_width = aspect_ratio * viewport_height

        camera_backward = normalize(position - look_at_point)
        camera_right = normalize(np.cross(up_direction, camera_backward))
        camera_up = normalize(np.cross(camera_backward, camera_right))

        self.origin = position
        focal_length = 1.0
        self.horizontal = viewport_width * camera_right * focal_length
        self.vertical = viewport_height * camera_up * focal_length
        self.lower_left_vertice = (self.origin - self.horizontal / 2 - self.vertical / 2 - camera_backward * focal_length)

    def get_ray(self, horizontal_percentage: float, vertical_percentage: float) -> Ray:
        direction = (
            self.lower_left_vertice
            + horizontal_percentage * self.horizontal
            + vertical_percentage * self.vertical
            - self.origin
        )
        return Ray(self.origin, direction)
