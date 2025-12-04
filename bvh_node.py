import random

from aabb import Axis_Aligned_Bounding_Box
from ray import Ray

class BVHNode:
    def __init__(self, objects: list):
        if len(objects) == 0:
            raise ValueError('Empty objects list for constructing BVHNode')
        
        if len(objects) == 1:
            self.left = self.right = objects[0]
            self.box = objects[0].bounding_box()
            return
        
        axis = random.randint(0, 2)
        objects_sorted = sorted(objects, key=lambda object: object.bounding_box().center[axis])
        middle_index = len(objects_sorted) // 2

        self.left = BVHNode(objects_sorted[:middle_index])
        self.right = BVHNode(objects_sorted[middle_index:])
        
        self.box = Axis_Aligned_Bounding_Box.surrounding_box(self.left.box, self.right.box)
    
    def hit(self, ray: Ray, time_min: float, time_max: float):
        if not self.box.hit(ray, time_min, time_max):
            return None
        
        hit_left = self.left.hit(ray, time_min, time_max)
        if hit_left:
            time_max = hit_left.time
        
        hit_right = self.right.hit(ray, time_min, time_max)
        if hit_right and (hit_left is None or hit_right.time < hit_left.time):
            return hit_right
        return hit_left
