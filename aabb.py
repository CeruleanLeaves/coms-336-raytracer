from typing import Self
import numpy as np
from ray import Ray


class Axis_Aligned_Bounding_Box:
    def __init__(self, minimum_vertice, maximum_vertice):
        self.minimum_vertice = minimum_vertice
        self.maximum_vertice = maximum_vertice

    @classmethod
    def surrounding_box(cls, box_a: Self, box_b: Self) -> Self:
        minimum_vertice = np.minimum(box_a.minimum_vertice, box_b.minimum_vertice)
        maximum_vertice = np.maximum(box_a.maximum_vertice, box_b.maximum_vertice)
        return cls(minimum_vertice, maximum_vertice)
    
    def hit(self, ray: Ray, time_min: float, time_max: float) -> bool:
        #slab test
        for axis in range(3):
            time_reached = (self.minimum_vertice[axis] - ray.origin[axis]) / ray.direction[axis]
            time_exited = (self.maximum_vertice[axis] - ray.origin[axis]) / ray.direction[axis]
            #reverse if negative direction
            if ray.direction[axis] < 0.0:
                time_reached, time_exited = time_exited, time_reached
            time_min = max(time_min, time_reached)
            time_max = min(time_max, time_exited)
            if not (time_min < time_max):
                return False
        return True
    
    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.minimum_vertice + self.maximum_vertice)
    
