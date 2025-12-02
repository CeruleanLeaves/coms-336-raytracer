import numpy as np

def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm != 0:
        return vector / norm
    return vector

def reflect(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vector - 2.0 * np.dot(vector, normal) * normal #Vector has to be normalized

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction
    
    def position(self, time: float) -> np.ndarray:
        return self.origin + self.direction * time

