# Copyright (c) 2026 Grzegorz Kociołek
# SPDX-License-Identifier: MIT

import numpy as np
import cv2
from dataclasses import dataclass
import scipy # for the Tukey window

import lib.helpers as helpers

def move_image_subpixel_fir(img: np.ndarray, move_vector: np.array, kernel_size = 6,
                            window = scipy.signal.windows.hann,
                            move_jitter: float = 0.0) -> np.ndarray:
  """
  Move the image's content in the direction of a continuous 2D move_vector (in pixel units) by
  performing a spatial domain convolution with a finite sinc kernel (hanned sinc interpolation).
  """

  hann = window(kernel_size)

  int_move = np.round(move_vector).astype(int)
  frac_move = move_vector - int_move

  # We roll the image by the integer part
  img_rolled = np.roll(img, shift=(int_move[0], int_move[1]), axis=(0, 1)) 
  img_extended = np.pad(img_rolled, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)), mode='wrap')

  sinc_kern_y = np.sinc(np.arange(kernel_size) - (kernel_size - 1)/2 + frac_move[0]) * hann
  sinc_kern_x = np.sinc(np.arange(kernel_size) - (kernel_size - 1)/2 + frac_move[1]) * hann

  sinc_kern_y /= np.sum(sinc_kern_y)
  sinc_kern_x /= np.sum(sinc_kern_x)

  # The sinc kernel is separable so we can optimize by separable to
  # horizontal and vertical convolution

  patches_y = np.lib.stride_tricks.sliding_window_view(img_extended, (kernel_size, 1), axis=(0,1))
  filtered_y = np.einsum("...jk,j->...", patches_y, sinc_kern_y)

  patches_x = np.lib.stride_tricks.sliding_window_view(filtered_y, (1, kernel_size), axis=(0,1))
  filtered_yx = np.einsum("...jk,k->...", patches_x, sinc_kern_x)

  return filtered_yx[:img.shape[0], :img.shape[1]]

def get_fft2_freq_matrix(height: int, width: int) -> np.ndarray:
  """
  Get a tuple matrix of FFT 2D frequency indices
  """
  d_h = 1.0/height
  d_w = 1.0/width
  y_freqs = np.fft.fftfreq(height, d_h)
  x_freqs = np.fft.rfftfreq(width, d_w)
  return np.stack(np.meshgrid(y_freqs, x_freqs, indexing='ij'), axis=-1)

def find_oversampling_resolution(img_shape: np.array, factor: np.array) -> np.array:
  # For each dimension we find the power of 2 closest to the dimension times its factor. 
  # Reason: FFT is fastest for dimensions being the power of 2.
  target_res = np.int32(2 ** np.ceil(np.log2(img_shape * factor)))
  return target_res

def move_image_subpixel_fft(img: np.ndarray, move_vector: np.array, oversampling_res: np.array = None) -> np.ndarray:
  """
  Move the image's content in the direction of a continuous 2D move_vector (in pixel units) via
  linear phase shift in the frequency domain.
  The image can be oversampled to avoid ringing artifacts due to too low spectral density and
  the spectral leakage.
  
  'oversampling_res' describe the oversampling resolution.
  `oversampling_res` should be higher than the input image resolution and at least 2x higher for correct wrapping.
  If it is None oversampling is not performed.

  Returns a shifted image of shape (w,h,3) with subpixel accuracy.
  The function expects the input image to be in the linear space (not sRGB). 
  """

  if len(img.shape) != 3 and img.shape[2] < 3:
    raise Exception("Invalid image tensor")

  if move_vector.shape != (2,):
    raise Exception("Invalid move vector")

  if oversampling_res is not None and not all(oversampling_res > img.shape[:2]):
    raise Exception("oversampling_res must be higher than the image resolution")
  
  if oversampling_res is not None:
    pad_res = (oversampling_res - np.array(img.shape[:2]))//2
    work_img = np.pad(img, ((pad_res[0], pad_res[0]), (pad_res[1], pad_res[1]), (0, 0)), mode='wrap')
    # The window must have constant amplitude in the visible part
    alpha_tukey =  0.5 * (pad_res * 2) / work_img.shape[:2]
    tukey_win_y = scipy.signal.windows.tukey(work_img.shape[0], alpha_tukey[0])
    tukey_win_x = scipy.signal.windows.tukey(work_img.shape[1], alpha_tukey[1])
    tukey_win = np.outer(tukey_win_y, tukey_win_x)
    visible_pos = pad_res
    work_img *= tukey_win[:,:,None]
  else:
    work_img = img
    visible_pos = np.array([0,0])

  # We adjust by an epsilon to avoid division by 0 errors
  adj_move_vector = move_vector + np.finfo(float).eps * helpers.sign_no_zero(move_vector)
  
  move_amplitude_px = np.linalg.norm(adj_move_vector)
  move_direction = adj_move_vector / move_amplitude_px

  img_fft = np.fft.rfft2(work_img, axes=(0,1))

  rad_per_unit_cycle = np.pi * 2 / np.array(work_img.shape[:2])
  rad_per_unit_move = np.linalg.norm(rad_per_unit_cycle * adj_move_vector)

  freqs_matrix = get_fft2_freq_matrix(work_img.shape[0], work_img.shape[1])

  # We calculate the influence of the move direction on a wave front
  freq_cosines = np.dot(freqs_matrix, move_direction)

  phase_shift_vec_matrix = np.exp(-rad_per_unit_move * freq_cosines * 1j)

  old_dc = np.copy(img_fft[0,0])
  img_fft *= phase_shift_vec_matrix[:,:,None]
  img_fft[0,0] = old_dc

  res_img = np.fft.irfft2(img_fft, axes=(0,1))
  return res_img[visible_pos[0]:visible_pos[0]+img.shape[0], visible_pos[1]:visible_pos[1]+img.shape[1]]

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
