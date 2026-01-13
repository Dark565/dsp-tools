# Copyright (c) 2026 Grzegorz Kociołek
# SPDX-License-Identifier: MIT

import numpy as np
import cv2

import lib.helpers as helpers

def get_fft2_freq_matrix(height: int, width: int) -> np.ndarray:
  """
  Get a tuple matrix of FFT 2D frequency indices
  """
  d_h = 1.0/height
  d_w = 1.0/width
  y_freqs = np.fft.fftfreq(height, d_h)
  x_freqs = np.fft.rfftfreq(width, d_w)
  return np.stack(np.meshgrid(y_freqs, x_freqs, indexing='ij'), axis=-1)


def move_image_subpixel(img: np.ndarray, move_vector: np.array) -> np.ndarray:
  """
  Move the image's content in the direction of a continuous 2D move_vector (in pixel units).

  Returns a shifted image of shape (w,h,3) with subpixel accuracy.
  The function expects the input image to be in the linear space (not sRGB). 
  """

  if len(img.shape) != 3 and img.shape[2] < 3:
    raise Exception("Invalid image tensor")

  if move_vector.shape != (2,):
    raise Exception("Invalid move vector")

  # We adjust by an epsilon to avoid division by 0 errors
  adj_move_vector = move_vector + np.finfo(float).eps * helpers.sign_no_zero(move_vector)
  
  move_amplitude_px = np.linalg.norm(adj_move_vector)
  move_direction = adj_move_vector / move_amplitude_px

  img_fft = np.fft.rfft2(img, axes=(0,1))

  rad_per_unit_cycle = np.pi * 2 / np.array(img.shape[:2])
  rad_per_unit_move = np.linalg.norm(rad_per_unit_cycle * adj_move_vector)

  freqs_matrix = get_fft2_freq_matrix(img.shape[0], img.shape[1])

  freq_cosines = np.dot(freqs_matrix, move_direction)

  phase_shift_vec_matrix = np.exp(-rad_per_unit_move * freq_cosines * 1j)

  old_dc = np.copy(img_fft[0,0])
  img_fft *= phase_shift_vec_matrix[:,:,None]
  img_fft[0,0] = old_dc

  res_img = np.fft.irfft2(img_fft, axes=(0,1))
  return res_img

def load_image(path: str) -> np.ndarray:
  return cv2.imread(path, flags=cv2.IMREAD_COLOR)

def save_image(img: np.ndarray, output: str):
  cv2.imwrite(output, (np.clip(img,0,255)))

def srgb_to_linear(img: np.ndarray) -> np.ndarray: 
  return np.where(
    img <= 0.04045,
    img / 12.92,
    np.power((img + 0.055) / 1.055, 2.4)
  )

def linear_to_srgb(img: np.ndarray) -> np.ndarray:
  return np.where(
    img <= 0.0031308,
    12.92 * img,
    1.055 * np.power(img, (1.0 / 2.4)) - 0.055
  )
