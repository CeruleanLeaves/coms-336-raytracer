import numpy as np
from dataclasses import dataclass


@dataclass
class HitRecord:
    time: float
    point: np.ndarray
    normal: np.ndarray
    base_color: np.ndarray
    is_mirror: bool