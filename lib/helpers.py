# Copyright (c) 2026 Grzegorz Kociołek
# SPDX-License-Identifier: MIT

import numpy as np

def complex_to_vec(c):
  return np.array([np.imag(c), np.real(c)])

def sign_no_zero(x):
  return np.where(x < 0.0, -1.0, 1.0)

def trangular_dist_rand(shape: tuple, min=0.0, max=1.0):
  range = max - min
  return (np.random.rand(*shape) + np.random.rand(*shape)) / 2 * range - min

BT709_LUMINANCE_WEIGHTS = [0.2126, 0.7152, 0.0722]

def rgb_to_luminance(rgb: np.array, axis=-1):
  return np.take(rgb, indices=0, axis=axis) * BT709_LUMINANCE_WEIGHTS[0] \
    + np.take(rgb, indices=1, axis=axis) * BT709_LUMINANCE_WEIGHTS[1] \
    + np.take(rgb, indices=2, axis=axis) * BT709_LUMINANCE_WEIGHTS[2]