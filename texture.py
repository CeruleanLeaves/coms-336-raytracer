from PIL import Image
import numpy as np

class Texture:
    def sample(self, x: float, y: float) -> np.ndarray:
        raise NotImplementedError

class ImageTexture(Texture):
    def __init__(self, filename: str):
        image = Image.open(filename).convert('RGB')
        self.width, self.height = image.size
        self.data = np.asarray(image, dtype=np.float32) / 255.0

    def sample(self, x: float, y: float) -> np.ndarray:
        x = max(0.0, min(1.0, x))
        y = 1.0 - max(0.0, min(1.0, y))
        x = int(x * (self.width - 1))
        y = int(y * (self.height - 1))
        return self.data[y, x, :]