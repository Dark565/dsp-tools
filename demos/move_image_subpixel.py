""" 
Copyright (c) 2026 Grzegorz Kociołek
SPDX-License-Identifier: MIT

This demo is a presentation of moving an image with the subpixel precision using
different interpolation methods like FIR or FFT.
The visible result should be image moving by constant displacement smoothly each frame.
"""

import numpy as np
import argparse
import time
import pygame
import scipy
from dataclasses import dataclass
from abc import ABC, abstractmethod

import lib.helpers as helpers
import lib.image_processing as image_processing
import lib.dither as dither

class ImageTransformAlgorithm(ABC):
  @abstractmethod
  def move(self, img: np.ndarray) -> np.ndarray:
    pass

class ImageTransformMoveAlgorithm(ImageTransformAlgorithm):
  def __init__(self, direction: np.array):
    self.direction = direction

  def set_direction(self, new_dir: np.array):
    self.direction = new_dir

class ImageTransformMoveFFT(ImageTransformMoveAlgorithm):
  def __init__(self, direction: np.array, oversampling_res: np.array):
    super()
    self.direction = direction
    self.oversampling_res = oversampling_res

  def move(self, img: np.ndarray):
    return image_processing.move_image_subpixel_fft(img, self.direction, self.oversampling_res)

class ImageTransformMoveFIR(ImageTransformMoveAlgorithm):
  def __init__(self, direction: np.array, kernel_size: int, window_function, move_jitter: float):
    self.direction = direction
    self.kernel_size = kernel_size
    self.window_function = window_function
    self.move_jitter = move_jitter

  def move(self, img: np.ndarray):
    return image_processing.move_image_subpixel_fir(img, self.direction, self.kernel_size,
                                                    self.window_function, self.move_jitter)

@dataclass
class AppConfig:
  framerate: int
  mouse_sensitivity: float
  dither_engine: dither.DitherEngine
  move_alg: ImageTransformMoveAlgorithm

class App:
  def __init__(self, image: np.ndarray, start_displacement: np.ndarray, config: AppConfig):
    """ 
      :param image: Image to move of shape (H, W, 3) with values in range [0, 255].
                    The image's size is expected to be even, if not the odd part is cut.
    """
    cut_img_shape = image.shape[:2] - np.mod(image.shape[:2], 2)
    cut_img = image[:cut_img_shape[0], :cut_img_shape[1]]

    # Pygame expects (width, height)
    self.ref_image = App._convert_image_for_proc(cut_img.transpose(1, 0, 2))
    self.image = cut_img.transpose(1, 0, 2)
    self.cur_image_pos = np.array([0.0, 0.0])
    self.displacement = App._vector_to_internal_metric(start_displacement.copy())
    self.config = config
    self._init_ui()

  def _init_ui(self):
    pygame.init()
    self.w_width = np.max([self.ref_image.shape[0], self.ref_image.shape[1], 1000])
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

    while self.running:

      for event in pygame.event.get():
        match event.type:
          case pygame.MOUSEMOTION:
            d_pos = (np.array(event.pos) - np.array([self.w_width//2, self.w_height//2])).astype(np.float32)
            d_pos *= self.config.mouse_sensitivity
            self.displacement = d_pos
          case pygame.QUIT:
            self.running = False

      if not self.running:
        break


      time_start = time.monotonic()
      self._displace_and_update_image()
      time_end = time.monotonic()

      pygame.surfarray.blit_array(self.surface, self.image)

      self.screen.blit(self.surface, (self.image_x, self.image_y))
      pygame.display.flip()
      self.clock.tick(self.config.framerate)

  def _displace_and_update_image(self):
    self.cur_image_pos += self.displacement

    self.config.move_alg.set_direction(self.cur_image_pos)
    moved_image = self.config.move_alg.move(self.ref_image)

    self.image = App._convert_image_for_view(moved_image, self.config.dither_engine)

  @staticmethod
  def _convert_image_for_view(img: np.ndarray, dither_engine: dither.DitherEngine):

    srgb_img = image_processing.linear_to_srgb(np.clip(img, 0.0, 1.0))
    dither_engine.apply(srgb_img) # dithering is applied in sRGB for correcting rounding support
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
    description="Draw a realtime animation of moving a given image in the direction of cursor displacement with a continuous subpixel precision"
  )

  parser.add_argument("input_image", type=str, help="Path to input file")
  parser.add_argument("--framerate", type=int, default=60, help="The framerate of the UI")
  parser.add_argument("--mouse-sensitivity", type=float, default=1.0, help="The sensitivity of mouse displacement" )
  parser.add_argument("--interpolation-algorithm", default='fir', choices=['fir', 'fft'])
  parser.add_argument("--kernel-size", type=int, default=8, help="Kernel size for the FIR interpolation")
  parser.add_argument("--window-function", choices=['hann', 'lanczos', 'none'], default='hann')
  parser.add_argument("--move_jitter", type=float, default=0.0, help="Jitter of a move vector for each pixel (for FIR)")
  parser.add_argument("--dither-amp", type=float, default=0.004, help="The amplitude of dithering")
  parser.add_argument("--dither-algorithm", default='random', choices=['random', 'selective-random'], help="The dither algorithm to use")
  parser.add_argument("--dither-block-size", type=int, default=32, help="Size of the dither block (for selective-random)")
  parser.add_argument("--no-oversampling", action='store_true', default=False, help="Disable oversampling (for FFT method)")
  parser.add_argument("--oversampling-factor", type=int, default=2, help="The factor of oversampling during processing (for FFT method)")

  args = parser.parse_args()
  input_img = image_processing.load_image(args.input_image)

  if args.dither_amp != 0.0:
    match args.dither_algorithm:
      case 'random':
        dither_inst = dither.DitherRandom(args.dither_amp)
      case 'selective-random':
        dither_inst = dither.DitherSelectiveRandom(args.dither_amp, args.dither_block_size)
  else:
    dither_inst = dither.DitherNone()

  oversampling_factor = args.oversampling_factor if not args.no_oversampling else None

  match args.window_function:
    case 'hann':
      window_function = scipy.signal.windows.hann
    case 'lanczos':
      window_function = scipy.signal.windows.lanczos
    case 'none':
      window_function = lambda shape: np.ones(shape)
    case _:
      raise Exception("Invalid window function")

  match args.interpolation_algorithm:
    case 'fir':
      interpolation_alg = ImageTransformMoveFIR(np.array([0,0]), args.kernel_size,
                                                window_function, args.move_jitter)
    case 'fft':
      oversampling_res = image_processing.find_oversampling_resolution(\
          input_img.shape[:2],
          np.array([args.oversampling_factor, args.oversampling_factor]))
      
      interpolation_alg = ImageTransformMoveFFT(np.array([0,0]), oversampling_res)
    case _:
      raise Exception("Invalid interpolation algorithm")

  app = App(input_img, np.array([0,0]), AppConfig(framerate=args.framerate,
                                                  mouse_sensitivity=args.mouse_sensitivity,
                                                  dither_engine=dither_inst,
                                                  move_alg=interpolation_alg))
  app.loop()

if __name__ == '__main__':
  main()
