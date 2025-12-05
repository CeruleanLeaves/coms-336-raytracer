import numpy as np
from ray import Ray, normalize

class Camera:
    def __init__(self,
                 position: np.ndarray,
                 look_at_point: np.ndarray,
                 up_direction: np.ndarray,
                 vertical_fov_degrees: float,
                 aspect_ratio: float,
                 aperture: float = 0.0,
                 focus_distance: float | None = None,
                 shutter_open_time: float = 0.0,
                 shutter_close_time: float = 0.0):
        
        self.shutter_open_time = shutter_open_time
        self.shutter_close_time = shutter_close_time

        vertical_fov_radians = np.deg2rad(vertical_fov_degrees)
        viewport_height = np.tan(vertical_fov_radians / 2) * 2
        viewport_width = aspect_ratio * viewport_height

        self.camera_backward = normalize(position - look_at_point)
        self.camera_right = normalize(np.cross(up_direction, self.camera_backward))
        self.camera_up = normalize(np.cross(self.camera_backward, self.camera_right))

        if focus_distance is None:
            focus_distance = float(np.linalg.norm(position - look_at_point))

        self.origin = position
        self.horizontal = viewport_width * self.camera_right * focus_distance
        self.vertical = viewport_height * self.camera_up * focus_distance
        self.lower_left_vertice = (self.origin - self.horizontal / 2 - self.vertical / 2 - self.camera_backward * focus_distance)
        self.lens_radius = aperture / 2.0

    def get_ray(self, horizontal_percentage: float, vertical_percentage: float) -> Ray:

        time = self.shutter_open_time + np.random.rand() * (self.shutter_close_time - self.shutter_open_time)

        if self.lens_radius > 0.0:
            random_radius = self.lens_radius * self.random_unit_disk_point()
            offset = self.camera_right * random_radius[0] + self.camera_up * random_radius[1]
            ray_origin = self.origin + offset
        else:
            ray_origin = self.origin

        direction = (
            self.lower_left_vertice
            + horizontal_percentage * self.horizontal
            + vertical_percentage * self.vertical
            - ray_origin
        )
        return Ray(ray_origin, direction, time=time)
    
    @staticmethod
    def random_unit_disk_point() -> np.ndarray:
        while True:
            point = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), 0.0], dtype=np.float32)
            if np.dot(point, point) < 1.0:
                return point
