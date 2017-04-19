# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  num_images = len(ims)
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob


def lidar_list_to_blob(lidars):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  top_shape = np.array([lidar[0].shape for lidar in lidars]).max(axis=0)
  front_shape = np.array([lidar[1].shape for lidar in lidars]).max(axis=0)
  num_lidars = len(lidars)
  top_blob = np.zeros((num_lidars, top_shape[0], top_shape[1], top_shape[2]),
                  dtype=np.float32)
  front_blob = np.zeros((num_lidars, front_shape[0], front_shape[1], front_shape[2]),
                  dtype=np.float32)
  for i in range(num_lidars):
    lidar = lidars[i][0]
    top_blob[i, 0:lidar.shape[0], 0:lidar.shape[1], 0:lidar.shape[2]] = lidar
    lidar = lidars[i][1]
    front_blob[i, 0:lidar.shape[0], 0:lidar.shape[1], 0:lidar.shape[2]] = lidar

  return (top_blob, front_blob)


def prep_im_for_blob(im, pixel_means, target_size, max_size):
  """Mean subtract and scale an image for use in a blob."""
  im = im.astype(np.float32, copy=False)
  im -= pixel_means
  im_shape = im.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  im_scale = float(target_size) / float(im_size_min)
  # Prevent the biggest axis from being more than MAX_SIZE
  if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
  im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)

  return im, im_scale

