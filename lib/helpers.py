# Copyright (c) 2026 Grzegorz Kociołek
# SPDX-License-Identifier: MIT

import numpy as np

def complex_to_vec(c):
  return np.array([np.imag(c), np.real(c)])

def sign_no_zero(x):
  return np.where(x < 0.0, -1.0, 1.0)

def add_epsilon_move_sign(x):
  return x + np.finfo(float).eps * sign_no_zero(x)

def add_epsilon(x):
  return x + np.finfo(float).eps

def trangular_dist_rand(shape: tuple, min=0.0, max=1.0):
  range = max - min
  return (np.random.rand(*shape) + np.random.rand(*shape)) / 2 * range - min

BT709_LUMINANCE_WEIGHTS = [0.2126, 0.7152, 0.0722]

def rgb_to_luminance(rgb: np.array, axis=-1):
  return np.take(rgb, indices=0, axis=axis) * BT709_LUMINANCE_WEIGHTS[0] \
    + np.take(rgb, indices=1, axis=axis) * BT709_LUMINANCE_WEIGHTS[1] \
    + np.take(rgb, indices=2, axis=axis) * BT709_LUMINANCE_WEIGHTS[2]

def alpha_blend(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
  """
  Perform linear blending of an array foreground onto an array background.
  The foreground array must have one more color channel than the background array.

  The input arrays are expected to be normalized into the <0, 1> real range.

  The result is in the same shape as the background array.
  """
  print(background.shape)
  print(foreground.shape)

  if len(background.shape) < 2:
    raise Exception("The background array must have at least 2 dimensions with the last dimension being the color channels")
  
  if foreground.shape[:-1] != background.shape[:-1] or foreground.shape[-1] != background.shape[-1] + 1:
    raise Exception("The foreground array must be of the same shape as the background array but with 1 more color channel (alpha)")

  alpha_broadcast = np.broadcast_to(foreground[..., -1][..., np.newaxis], background.shape)
  blend_img = foreground[..., :-1] * alpha_broadcast + (1.0 - alpha_broadcast) * background
  return blend_img