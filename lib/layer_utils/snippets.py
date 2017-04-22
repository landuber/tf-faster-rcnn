# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from layer_utils.generate_anchors import generate_anchors
from model.bbox_transform import bbox_transform_inv, clip_boxes
from utils.cython_bbox import bbox_overlaps


def generate_anchors_pre(height, width, feat_stride, anchor_scales):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(scales=np.array(anchor_scales))
  A = anchors.shape[0]
  print("A")
  print(A)
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  shifts = np.insert(shifts, 2, 0, axis=1)
  shifts = np.insert(shifts, 5, 0, axis=1)

  K = shifts.shape[0]
  print('height')
  print(height)
  print('width')
  print(width)
  print('feat_stride')
  print(feat_stride)
  print("shifts")
  print(shifts[1:5, :])
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 6)) + shifts.reshape((1, K, 6)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 6)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length
