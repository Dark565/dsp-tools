""" 
Copyright (c) 2026 Grzegorz Kociołek
SPDX-License-Identifier: MIT

This demo is a representation of moving an image with the subpixel precision
by first converting an image from the spatial domain to the spatial frequency domain and
applying wavefront phase shift, then performing sampling of the continuous wave field using
inverse discrete fourier transform.

The visible result should be image moving by constant displacement smoothly each frame.
"""

import numpy as np
import argparse
import time
import pygame
from dataclasses import dataclass

import lib.helpers as helpers
import lib.image_processing as image_processing

@dataclass
class AppConfig:
  framerate: int  

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
    moved_image = image_processing.move_image_subpixel(self.ref_image, self.cur_image_pos)
    self.image = App._convert_image_for_view(moved_image)

  @staticmethod
  def _convert_image_for_view(img: np.ndarray):
    return np.astype(image_processing.linear_to_srgb(np.clip(img, 0.0, 1.0)) * 255.0, np.uint8)

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

  args = parser.parse_args()
  input_img = image_processing.load_image(args.input_image)

  app = App(input_img, np.array([0,0]), AppConfig(args.framerate))
  app.loop()

if __name__ == '__main__':
  main()
