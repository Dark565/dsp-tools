# Copyright (c) 2026 Grzegorz Kociołek
# SPDX-License-Identifier: MIT

import numpy as np

def get_circle_intersection(pos1: np.ndarray, pos2: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    dist_vec = pos1 - pos2  # (..., 2)
    dist_sq = np.einsum("...i,...i->...", dist_vec, dist_vec)
    dist_len = np.sqrt(dist_sq)

    r1_rs = r1.squeeze(axis=-1)
    r2_rs = r2.squeeze(axis=-1)

    invalid_mask = (dist_sq > (r1_rs + r2_rs)**2) | (dist_sq <= (r1_rs - r2_rs)**2)

    r1_2 = r1_rs**2
    r2_2 = r2_rs**2

    a = np.divide(r1_2 - r2_2 + dist_sq, 2 * dist_len, 
                  out=np.zeros_like(dist_len), where=~invalid_mask)
    
    h_sq = r1_2 - a**2
    h = np.sqrt(np.maximum(h_sq, 0))

    unit_v = np.divide(-dist_vec, dist_len[..., np.newaxis], 
                       out=np.zeros_like(dist_vec), where=dist_len[..., np.newaxis] != 0)
    
    p2 = pos1 + a[..., np.newaxis] * unit_v

    ortho_v = np.stack([-unit_v[..., 1], unit_v[..., 0]], axis=-1)

    res1 = p2 + h[..., np.newaxis] * ortho_v
    res2 = p2 - h[..., np.newaxis] * ortho_v

    res1[invalid_mask] = np.nan
    res2[invalid_mask] = np.nan

    return np.stack([res1, res2], axis=-2)


def trilaterate_2d(vertices: tuple[np.ndarray, np.ndarray, np.ndarray],
                   values: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Calculate the position of a point given distances at 3 points.
    Return an array where the last axis is the vector of coordinates.
    """

    # TODO: Vectorize take_along_axis into single invocations.

    if np.array(vertices).shape[1:] != vertices[0].shape:
        raise Exception("All 3 vertex arrays must have the same shape")

    if vertices[0].shape[-1] != 2:
        raise Exception("The last axis in the vertex array must have size 2")

    if np.array(values).shape[1:] != values[0].shape:
        raise Exception("All 3 value arrays must have the same shape")

    if values[0].shape[-1] != 1:
        raise Exception("The last axis in the value arrays must be a scalar")

    # We pack 2 circles into uniform tensors as an optimization 

    vertex_tensor_1 = np.stack((vertices[0], vertices[0]), axis=0)
    vertex_tensor_2 = np.stack((vertices[1], vertices[2]), axis=0)
    values_tensor_1 = np.stack((values[0], values[0]), axis=0)
    values_tensor_2 = np.stack((values[1], values[2]), axis=0)

    intersections = get_circle_intersection(vertex_tensor_1, vertex_tensor_2,
                                            values_tensor_1, values_tensor_2)

    dists = intersections[0, ..., :, np.newaxis, :] - intersections[1, ..., np.newaxis, :, :]
    dists_sq = np.einsum("...ijk,...ijk->...ij", dists, dists)
    dists_sq = dists_sq.reshape(*dists_sq.shape[:-2], 4)
    best_index = np.argmin(dists_sq, axis=-1)

    best_i = (best_index // 2)[..., np.newaxis, np.newaxis]
    best_j = (best_index  % 2)[..., np.newaxis, np.newaxis]

    best_ab = np.take_along_axis(intersections[0], indices=best_i, axis=-2)
    best_ac = np.take_along_axis(intersections[1], indices=best_j, axis=-2)

    # vector average
    src_point = (np.squeeze(best_ab, axis=-2) + np.squeeze(best_ac, axis=-2)) / 2
    return src_point









    
