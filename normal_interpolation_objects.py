import numpy as np

from materials import Lambertian
from triangle import Triangle

# Octahedron vertices (centered at origin)
v_top    = np.array([ 0.0,  1.0,  -2.0], dtype=np.float32)
v_bottom = np.array([ 0.0, -1.0,  -2.0], dtype=np.float32)
v_right  = np.array([ 1.0,  0.0,  -2.0], dtype=np.float32)
v_left   = np.array([-1.0,  0.0,  -2.0], dtype=np.float32)
v_front  = np.array([ 0.0,  0.0,  -1.0], dtype=np.float32)
v_back   = np.array([ 0.0,  0.0, -3.0], dtype=np.float32)

# For a sphere-like shading, vertex normals = normalized vertex positions
def norm(p: np.ndarray) -> np.ndarray:
    return p / np.linalg.norm(p)

n_top    = norm(v_top)
n_bottom = norm(v_bottom)
n_right  = norm(v_right)
n_left   = norm(v_left)
n_front  = norm(v_front)
n_back   = norm(v_back)

# Some material you already defined
mat_red_sphere = Lambertian(np.array([0.8, 0.2, 0.2], dtype=np.float32))

flat_sphere_tris = [
    # Top four faces
    Triangle(v_top,    v_right, v_front, mat_red_sphere),
    Triangle(v_top,    v_front, v_left,  mat_red_sphere),
    Triangle(v_top,    v_left,  v_back,  mat_red_sphere),
    Triangle(v_top,    v_back,  v_right, mat_red_sphere),

    # Bottom four faces
    Triangle(v_bottom, v_front, v_right, mat_red_sphere),
    Triangle(v_bottom, v_left,  v_front, mat_red_sphere),
    Triangle(v_bottom, v_back,  v_left,  mat_red_sphere),
    Triangle(v_bottom, v_right, v_back,  mat_red_sphere),
]

smooth_sphere_tris = [
    # Top four faces
    Triangle(v_top,    v_right, v_front, mat_red_sphere,
             normal0=n_top, normal1=n_right, normal2=n_front),

    Triangle(v_top,    v_front, v_left,  mat_red_sphere,
             normal0=n_top, normal1=n_front, normal2=n_left),

    Triangle(v_top,    v_left,  v_back,  mat_red_sphere,
             normal0=n_top, normal1=n_left, normal2=n_back),

    Triangle(v_top,    v_back,  v_right, mat_red_sphere,
             normal0=n_top, normal1=n_back, normal2=n_right),

    # Bottom four faces
    Triangle(v_bottom, v_front, v_right, mat_red_sphere,
             normal0=n_bottom, normal1=n_front, normal2=n_right),

    Triangle(v_bottom, v_left,  v_front, mat_red_sphere,
             normal0=n_bottom, normal1=n_left, normal2=n_front),

    Triangle(v_bottom, v_back,  v_left,  mat_red_sphere,
             normal0=n_bottom, normal1=n_back, normal2=n_left),

    Triangle(v_bottom, v_right, v_back,  mat_red_sphere,
             normal0=n_bottom, normal1=n_right, normal2=n_back),
]
