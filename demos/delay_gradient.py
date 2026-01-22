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
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Callable

import lib.helpers as helpers
import lib.image_processing as image_processing

def get_quiver_vector_field_image(
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
    return image_processing.srgb_to_linear(rgba_buffer / 255.0).astype(np.uint8)

def get_color_plane_image(
    vec_field: np.ndarray,
    normal: np.array,
    start_vec: np.array
) -> np.ndarray:
  
  # TODO: Fix color scaling for certain normals

  if len(vec_field.shape) < 4 or vec_field.shape[-2:] != (3, 2):
    raise Exception("Vector field is expected to be of shape (H, W, 3, 2)")

  u,v = helpers.get_orthonormal_vector_3d_starting_on_vec(normal, start_vec, True)
  bt709_weights = np.array([0.2126, 0.7152, 0.0722])

  # We calculate the center of rotation
  color_box = np.ones(u.shape, dtype=u.dtype)
  u_box_fit = helpers.distance_to_face(u, color_box)
  v_box_fit = helpers.distance_to_face(v, color_box)

  # This should have shape (1,)
  max_fit_amplitude = np.min((u_box_fit, v_box_fit), axis=-1) / 2

  gray_vf = np.einsum("...ij,i->...j", vec_field, bt709_weights)
  magnitudes = np.linalg.norm(gray_vf, axis=-1)
  norms = gray_vf / magnitudes[..., np.newaxis]

  # This should have shape (H, W, 3)
  color_vecs = (norms[..., 0, np.newaxis] * u[np.newaxis,np.newaxis] \
                + norms[..., 1, np.newaxis] * v[np.newaxis,np.newaxis]) \
                  * magnitudes[..., np.newaxis] * max_fit_amplitude

  return color_vecs + [0.5, 0.5, 0.5]


class VectorFieldVisualizationStrategy(ABC):
  @abstractmethod
  def get_visualization_array(self, img: np.ndarray) -> np.ndarray:
    """
    Takes the vector field array of shape (H, W, 3, 2) and returns
    visualization in shape (H, W, 4). An additional channel for alpha.

    The input and output are in the linear color space.
    """
    pass

class VectorFieldVisualizationQuivers(VectorFieldVisualizationStrategy):
  def __init__(self, color: str, dpi: int):
    super()
    self.color = color
    self.dpi = dpi

  def get_visualization_array(self, img):
    quiver_image = get_quiver_vector_field_image(img, self.color, self.dpi)

    # TODO: Workaround, alpha should be set in get_quiver_vector_field_image
    res_img = np.concatenate((quiver_image,
                              np.full(shape=(*quiver_image.shape[:-1], 1),
                                      fill_value=0.5, dtype=quiver_image.dtype)), 
                              axis=-1)
    
    return np.flip(res_img, axis=2) # Flip for compliance with the pygame order


class VectorFieldVisualizationColorPlane(VectorFieldVisualizationStrategy):
  """
  This strategy treats the vector angle as the angle on a color plane and
  the vector magnitude as the color intensity.
  """
  def __init__(self, color_normal: np.array, starting_color: np.array, alpha: float):
    super()
    self.color_normal = color_normal / np.linalg.norm(color_normal, axis=-1, keepdims=True)
    self.starting_color = starting_color
    self.alpha = alpha

  def get_visualization_array(self, img):
    vis_img = get_color_plane_image(img, self.color_normal, self.starting_color)

    res_img = np.concatenate((vis_img,
                              np.full(shape=(*vis_img.shape[:-1], 1),
                                      fill_value=self.alpha, dtype=vis_img.dtype)), 
                              axis=-1)
    
    return res_img

@dataclass
class AppConfig:
  framerate: int
  mouse_sensitivity: float
  kernel_size: int
  window_function: Callable
  visualization_strategy: VectorFieldVisualizationStrategy
  

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

    # TODO: Check if axis rotation is not required for some visualization strategies
    # in <0,1> linear
    vector_field_img = self.config.visualization_strategy.get_visualization_array(gradient_vector_array)

    image_blend = helpers.alpha_blend(self.ref_image, vector_field_img)
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
  parser.add_argument("--visualization-method", choices=['quivers', 'colorplane'], default='colorplane', help="The visualization method for gradient vectors")
  parser.add_argument("--dpi", type=int, default=100, help="DPI setting for certain visualization methods")
  parser.add_argument("--quiver-color", type=str, default='cyan')
  parser.add_argument("--color-normal", type=str, default="#ff0000", help="The rotation axis color for the colorplane visualization")
  parser.add_argument("--color-start", type=str, default="#0000ff", help="Start color for the colorplane visualization")


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

  color_normal = np.array(matplotlib.colors.to_rgb(args.color_normal))
  color_start  = np.array(matplotlib.colors.to_rgb(args.color_start))

  match args.visualization_method:
    case 'quivers':
      # TODO: This color should probably be parsed by colors.to_rgb too
      vis_method = VectorFieldVisualizationQuivers(args.quiver_color, args.dpi)
    case 'colorplane':
      # XXX: Alpha is currently hardcoded to 0.5. It should probably be selectable as a
      #      command line argument
      vis_method = VectorFieldVisualizationColorPlane(color_normal, color_start, 1.0)
    case _:
      raise Exception("Invalid visualization method selected")

  app = App(input_img, np.array([0,0]), AppConfig(framerate=args.framerate,
                                                  mouse_sensitivity=args.mouse_sensitivity,
                                                  kernel_size=args.kernel_size,
                                                  window_function=window_function,
                                                  visualization_strategy=vis_method))
  app.loop()

if __name__ == '__main__':
  main()
