from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from materials import Material


@dataclass
class HitRecord:
    time: float
    point: np.ndarray
    normal: np.ndarray
    material: Material
    front_face: bool
    texture_coordinates: np.ndarray | None = None