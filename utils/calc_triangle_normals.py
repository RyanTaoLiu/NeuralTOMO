import torch
import numpy as np
def compute_normals(v, f):
    # Fetch the vertices for each face
    v1 = v[f[:, 0]]
    v2 = v[f[:, 1]]
    v3 = v[f[:, 2]]

    # Compute the vectors for two sides of the triangle
    edge1 = v2 - v1
    edge2 = v3 - v1

    # Compute the cross product (normal to the plane of the triangle)
    normals = torch.cross(edge1, edge2, dim=1)

    # Normalize the normals to unit length
    # normals = torch.nn.functional.normalize(normals, p=2, dim=1)
    # DO NOT normalize to make sure the length is the area of triangle
    return normals


def compute_normals_np(v, f):
    # Fetch the vertices for each face
    v1 = v[f[:, 0]]
    v2 = v[f[:, 1]]
    v3 = v[f[:, 2]]

    # Compute the vectors for two sides of the triangle
    edge1 = v2 - v1
    edge2 = v3 - v1

    # Compute the cross product (normal to the plane of the triangle)
    normals = np.cross(edge1, edge2, axis=1)

    # Normalize the normals to unit length
    # normals = torch.nn.functional.normalize(normals, p=2, dim=1)
    # DO NOT normalize to make sure the length is the area of triangle
    return normals