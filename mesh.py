import numpy as np
from materials import Material
from ray import Ray
from triangle import Triangle


class Mesh:
    def __init__(self, triangles: list[Triangle]):
        self.triangles = triangles

    @classmethod
    def from_vertices_indices(cls, vertices: np.ndarray, indices: list[tuple[int, int, int]], material: Material):
        triangles = []
        for i0, i1, i2, in indices:
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
            #make sure front face vertices are counter clockwise
            triangles.append(Triangle(v0, v1, v2, material)) 
        return cls(triangles)
    
    @classmethod
    def load_from_file(cls, file_path: str, material: Material):
        vertices = []
        indices = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                #format for vertice: v x y z
                if line.startswith('v '):
                    tokens = line.split()
                    x = float(tokens[1])
                    y = float(tokens[2])
                    z = float(tokens[3])
                    vertices.append([x, y, z])
                #format for face: f i j k or f i/j/k i/j/k i/j/k
                elif line.startswith('f '):
                    tokens = line.split()[1:]
                    triangle_indices = []
                    for token in tokens:
                        indice = int(token.split('/')[0]) - 1
                        triangle_indices.append(indice)
                    
                    if len(triangle_indices) < 3:
                        continue
                    elif len(triangle_indices) == 3:
                        indices.append((triangle_indices[0], triangle_indices[1], triangle_indices[2]))
                    else:
                        #handling quad and higher faces
                        #(v0, v1, v2), (v0, v2, v3), ...
                        v0 = triangle_indices[0]
                        for i in range(1, len(triangle_indices) - 1):
                            v1 = triangle_indices[i]
                            v2 = triangle_indices[i+1]
                            indices.append((v0, v1, v2))
        
        if not vertices or not indices:
            raise ValueError(f'File {file_path} has error')
        
        vertices_np = np.array(vertices, dtype=np.float32)
        return cls.from_vertices_indices(vertices_np, indices, material)

    def hit(self, ray: Ray, time_min: float, time_max: float):
        closest_time = time_max
        hit_record = None

        for triangle in self.triangles:
            hit = triangle.hit(ray, time_min, time_max)
            if hit and hit.time < closest_time:
                closest_time = hit.time
                hit_record = hit

        return hit_record