# Copyright (c) 2026 Grzegorz Kociołek
# SPDX-License-Identifier: MIT

import numpy as np
from abc import ABC, abstractmethod

import lib.helpers as helpers

class DitherEngine(ABC):
    """ 
    The abstract class for dither implementations. 
    """
    def __init__(self, amplitude: float):
        self.amplitude = amplitude

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply dithering to a given image.

        This function modifies the given image.
        Returns the array.
        """
        pass

class DitherNone(DitherEngine):
    def __init__(self):
        super().__init__(0.0)

    def apply(self, img):
        return img
        
class DitherRandom(DitherEngine):
    def apply(self, img):
        noise = helpers.trangular_dist_rand(img.shape, 0.0, 1.0) * self.amplitude
        img += noise

class DitherSelectiveRandom(DitherEngine):
    """
    An algorithm of selective dithering.
    Works by splitting an image into fixed-sized blocks, checking the sum of
    gradient magnitudes (the cumulative gradient pressure) in each and
    applying dithering to those that fit the cumulative pressure range.

    If image dimensions are not an integer multiply of the block size,
    the image is padded with 0s (for calculations) to fit.
    """
    
    # Minimum gradient magnitude threshold coefficient for block selection
    T_MIN = 0.016
    # Maximum gradient magnitude threshold coefficient for block selection
    T_MAX = 0.18
    
    def __init__(self, amplitude: float, block_size: int):
        super().__init__(amplitude)
        self.block_size = block_size
        self.t_min = DitherSelectiveRandom.T_MIN * block_size**2
        self.t_max = DitherSelectiveRandom.T_MAX * block_size**2

    def apply(self, img):
        img_dim = np.array(img.shape[:2])
        img_dim = (img_dim - 1) - (img_dim - 1) % self.block_size + self.block_size
    
        if any(img_dim != np.array(img.shape[:2])):
            work_img = np.zeros(shape=(*img_dim, img.shape[2]))
            work_img[:img.shape[0], :img.shape[1]] = img
        else:
            work_img = img

        gray_img = helpers.rgb_to_luminance(work_img, axis=2)

        n_blocks = img_dim // self.block_size
        img_blocks = gray_img\
            .reshape((n_blocks[0], self.block_size, n_blocks[1], self.block_size))\
            .transpose(0, 2, 1, 3)
        
        # For the first row and column we calculate gradients along only one axis
        cum_grad = np.zeros(shape=n_blocks, dtype=np.float32)

        dy = np.diff(img_blocks, axis=2)
        dx = np.diff(img_blocks, axis=3)

        cum_grad += np.sum(np.abs(dy[:,:,:,0]), axis=2)
        cum_grad += np.sum(np.abs(dx[:,:,0,:]), axis=2)

        inner_dx = dx[:,:,1:]
        inner_dy = dy[:,:,:,1:]

        grad_mgn = np.linalg.norm(np.stack((inner_dy, inner_dx), axis=-1), axis=-1)

        cum_grad += np.sum(grad_mgn, axis=(2,3))

        dither_apply_mask = (cum_grad >= self.t_min) & (cum_grad < self.t_max)
        expanded_dither_mask = np.broadcast_to(dither_apply_mask[:,:,np.newaxis,np.newaxis,np.newaxis],
                                                (n_blocks[0], n_blocks[1], self.block_size, self.block_size, img.shape[2]))

        img_mask = expanded_dither_mask.transpose(0, 2, 1, 3, 4).reshape(work_img.shape)[:img.shape[0], :img.shape[1]]
        dither_noise = helpers.trangular_dist_rand(shape=img.shape, min=0.0, max=1.0) * self.amplitude

        masked_noise = np.where(img_mask, dither_noise, 0.0)

        img += masked_noise
        return img











