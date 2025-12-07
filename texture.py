import math
from PIL import Image
import numpy as np

class Texture:
    def sample(self, x: float, y: float, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ImageTexture(Texture):
    def __init__(self, filename: str):
        image = Image.open(filename).convert('RGB')
        self.width, self.height = image.size
        self.data = np.asarray(image, dtype=np.float32) / 255.0

    def sample(self, x: float, y: float, point: np.ndarray) -> np.ndarray:
        x = max(0.0, min(1.0, x))
        y = 1.0 - max(0.0, min(1.0, y))
        x = int(x * (self.width - 1))
        y = int(y * (self.height - 1))
        return self.data[y, x, :]
    
class Perlin:
    def __init__(self, point_count: int = 256):
        self.point_count = point_count
        self.random_floats = np.random.rand(point_count).astype(np.float32)
        permutation = np.arange(point_count, dtype=np.int32)
        np.random.shuffle(permutation)
        self.permutation_x = permutation.copy()
        np.random.shuffle(permutation)
        self.permutation_y = permutation.copy()
        np.random.shuffle(permutation)
        self.permutation_z = permutation.copy()

    def noise(self, point: np.ndarray) -> float:
        x = int(math.floor(point[0])) & self.point_count - 1
        y = int(math.floor(point[1])) & self.point_count - 1
        z = int(math.floor(point[2])) & self.point_count - 1
        random_float_index = self.permutation_x[x] ^ self.permutation_y[y] ^ self.permutation_z[z]
        return float(self.random_floats[random_float_index])
    
    def turbulence(self, point: np.ndarray, depth: int = 6) -> float:
        accumulator = 0.0
        temp_point = point.copy()
        weight = 1.0
        for _ in range(depth):
            accumulator += weight * self.noise(temp_point)
            weight *= 0.5
            temp_point *= 2.0
        return abs(accumulator)
    
class PerlinNoiseTexture(Texture):
    def __init__(self, scale: float = 1.0, base_color: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)):
        self.scale = scale
        self.noise = Perlin()
        self.base_color = base_color

    def sample(self, x: float, y: float, point: np.ndarray) -> np.ndarray:
        t = 0.5 * (1.0 + math.sin(self.scale * point[2] + 10.0 * self.noise.turbulence(point)))
        return t * self.base_color