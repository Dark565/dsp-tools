""" 
Copyright (c) 2026 Grzegorz Kociołek
SPDX-License-Identifier: MIT

This demo shows subpixel group delay gradient visualized on top of an image using 
trilateration of wave field values probed at 3 equilateral triangle vertices using wavelet transform.
"""

import numpy as np
import argparse
import pygame
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable

import lib.helpers as helpers
import lib.image_processing as image_processing

def get_vector_field_image(
    arr: np.array,
    color='teal',
    dpi=100
):
    weights = np.array([0.2126, 0.7152, 0.0722])

    combined_vectors = np.tensordot(arr, weights, axes=([2], [0]))

    U = combined_vectors[:, :, 0]
    V = combined_vectors[:, :, 1]
    
    H, W = U.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor='none')

    ax.quiver(X, Y, U, V, color=color, angles='xy', scale_units='xy', scale=1, pivot='middle')

    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    fig.canvas.draw()
    rgba_buffer = np.array(fig.canvas.buffer_rgba())

    # TODO: This is a dirty workaround. This padding should not be needed

    pad_width = (
      (np.max((0, arr.shape[0] - rgba_buffer.shape[0])), 0),
      (np.max((0, arr.shape[1] - rgba_buffer.shape[1])), 0),
      (0, 0)
    )
    rgba_buffer = np.pad(rgba_buffer, pad_width)

    plt.close(fig)
    return rgba_buffer


@dataclass
class AppConfig:
  framerate: int
  mouse_sensitivity: float
  kernel_size: int
  window_function: Callable

class App:
  def __init__(self, image: np.ndarray, start_displacement: np.ndarray, config: AppConfig):
    """ 
      :param image: Image to move of shape (H, W, 3) with values in range [0, 255].
                    The image's size is expected to be even, if not, the odd part is cut.
    """
    cut_img_shape = image.shape[:2] - np.mod(image.shape[:2], 2)
    cut_img = image[:cut_img_shape[0], :cut_img_shape[1]]

    # Pygame expects (width, height)
    self.ref_image = App._convert_image_for_proc(cut_img.transpose(1, 0, 2))
    self.image = cut_img.transpose(1, 0, 2)
    self.config = config
    self._init_ui()

  def _init_ui(self):
    pygame.init()
    self.w_width = np.max([self.image.shape[0], self.image.shape[1], 1000])
    self.w_height = self.w_width

    self.image_x = (self.w_width - self.ref_image.shape[0]) // 2
    self.image_y = (self.w_height - self.ref_image.shape[1]) // 2

    self.screen = pygame.display.set_mode((self.w_width, self.w_height))
    pygame.display.set_caption("Subpixel moving")

    self.surface = pygame.Surface((self.image.shape[0], self.image.shape[1]))

    self.clock = pygame.time.Clock()
    self.running = False

  def loop(self):
    self.running = True

    pygame.event.set_grab(True)

    self.need_update = True

    while self.running:

      for event in pygame.event.get():
        match event.type:
          case pygame.QUIT:
            self.running = False

      if not self.running:
        break

      if self.need_update:
        self.need_update = False
        self._update_image()
        self.clock.tick(self.config.framerate)
        pygame.surfarray.blit_array(self.surface, self.image)
        self.screen.blit(self.surface, (self.image_x, self.image_y))
        pygame.display.flip()

  def _update_image(self):
    gradient_vector_array = image_processing.calculate_gdg_2d(
      self.ref_image,
      self.config.kernel_size,
      self.config.window_function)

    # in <0,255> sRGB
    vector_field_img = get_vector_field_image(gradient_vector_array)
    vector_field_img_proc = App._convert_image_for_proc(vector_field_img[...,:-1])

    # TODO: Workaround, alpha should be set in get_vector_field_image
    vector_field_img_proc = np.concatenate((vector_field_img_proc,
                                            np.full(shape=(*vector_field_img_proc.shape[:-1], 1),
                                                    fill_value=0.6, dtype=vector_field_img_proc.dtype)), 
                                            axis=-1)
    
    image_blend = helpers.alpha_blend(self.ref_image, vector_field_img_proc)

    self.image = App._convert_image_for_view(image_blend)

  @staticmethod
  def _convert_image_for_view(img: np.ndarray):

    srgb_img = image_processing.linear_to_srgb(np.clip(img, 0.0, 1.0))
    return np.astype(np.clip(srgb_img, 0.0, 1.0) * 255.0, np.uint8)

  @staticmethod
  def _convert_image_for_proc(img: np.ndarray):
    # pygame expects color axis in the flipped order than what cv2 provides
    img_colors_flipped = np.flip(img, axis=2)
    # Image should be converted to the linear space for correct processing

    return image_processing.srgb_to_linear(np.clip(img_colors_flipped, 0, 255).astype(np.float32) / 255.0)
  
  @staticmethod
  def _vector_to_internal_metric(v: np.ndarray):
    return np.flip(v, axis=0)
  
  @staticmethod
  def _vector_to_external_metric(v: np.ndarray):
    return np.flip(v, axis=0)
  

def main():
  parser = argparse.ArgumentParser(
    description="Draw a vector field of group delay gradients at each pixel position"
  )

  parser.add_argument("input_image", type=str, help="Path to input file")
  parser.add_argument("--framerate", type=int, default=60, help="The framerate of the UI")
  parser.add_argument("--mouse-sensitivity", type=float, default=1.0, help="The sensitivity of mouse displacement" )
  parser.add_argument("--kernel-size", type=int, default=8, help="Kernel size for convolution")
  parser.add_argument("--window-function", choices=['hann', 'lanczos', 'none'], default='hann')

  args = parser.parse_args()
  input_img = image_processing.load_image(args.input_image)

  match args.window_function:
    case 'hann':
      window_function = scipy.signal.windows.hann
    case 'lanczos':
      window_function = scipy.signal.windows.lanczos
    case 'none':
      window_function = lambda shape: np.ones(shape)
    case _:
      raise Exception("Invalid window function")

  app = App(input_img, np.array([0,0]), AppConfig(framerate=args.framerate,
                                                  mouse_sensitivity=args.mouse_sensitivity,
                                                  kernel_size=args.kernel_size,
                                                  window_function=window_function))
  app.loop()

if __name__ == '__main__':
  main()
