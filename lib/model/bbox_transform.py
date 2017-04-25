# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def bbox_transform(ex_rois, gt_rois):
  ex_widths = ex_rois[:, 3] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 4] - ex_rois[:, 1] + 1.0
  ex_depths = ex_rois[:, 5] - ex_rois[:, 2] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
  ex_ctr_z = ex_rois[:, 2] + 0.5 * ex_depths

  gt_widths = gt_rois[:, 3] - gt_rois[:, 0] + 1.0
  gt_heights = gt_rois[:, 4] - gt_rois[:, 1] + 1.0
  gt_depths = gt_rois[:, 5] - gt_rois[:, 2] + 1.0
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
  gt_ctr_z = gt_rois[:, 2] + 0.5 * gt_depths

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dz = (gt_ctr_z - ex_ctr_z) / ex_depths
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)
  targets_dd = np.log(gt_depths / ex_depths)

  targets = np.vstack(
    (targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dd)).transpose()
  return targets


def corner_transform(rois_corners, gt_corners):
    deltas = rois_corners[:,3,:] -  rois_corners[:,0,:]
    diagonals = np.hypot(np.hypot(deltas[:,0], deltas[:,1]), deltas[:,2])
    diagonals = diagonals[:, np.newaxis]
    targets = (gt_corners - rois_corners)
    targets = targets.reshape((-1, 24))
    targets = targets / diagonals
    return targets.reshape((-1, 8, 3))




def bbox_transform_inv(boxes, deltas):
  if boxes.shape[0] == 0:
    return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

  boxes = boxes.astype(deltas.dtype, copy=False)
  widths = boxes[:, 3] - boxes[:, 0] + 1.0
  heights = boxes[:, 4] - boxes[:, 1] + 1.0
  depths = boxes[:, 5] - boxes[:, 2] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights
  ctr_z = boxes[:, 2] + 0.5 * depths

  dx = deltas[:, 0::6]
  dy = deltas[:, 1::6]
  dz = deltas[:, 2::6]
  dw = deltas[:, 3::6]
  dh = deltas[:, 4::6]
  dd = deltas[:, 5::6]
  
  pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
  pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
  pred_ctr_z = dz * depths[:, np.newaxis] + ctr_z[:, np.newaxis]
  pred_w = np.exp(dw) * widths[:, np.newaxis]
  pred_h = np.exp(dh) * heights[:, np.newaxis]
  pred_d = np.exp(dd) * depths[:, np.newaxis]

  pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
  # x1
  pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
  # y1
  pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_h
  # z1
  pred_boxes[:, 2::6] = pred_ctr_z - 0.5 * pred_d
  # x2
  pred_boxes[:, 3::6] = pred_ctr_x + 0.5 * pred_w
  # y2
  pred_boxes[:, 4::6] = pred_ctr_y + 0.5 * pred_h
  # z2
  pred_boxes[:, 5::6] = pred_ctr_z + 0.5 * pred_d

  return pred_boxes


def clip_boxes(boxes, im_shape):
  """
  Clip boxes to image boundaries.
  """

  # x1 >= 0
  boxes[:, 0::6] = np.maximum(np.minimum(boxes[:, 0::6], im_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::6] = np.maximum(np.minimum(boxes[:, 1::6], im_shape[0] - 1), 0)
  # z1 >= 0
  boxes[:, 2::6] = np.maximum(np.minimum(boxes[:, 2::6], im_shape[2] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 3::6] = np.maximum(np.minimum(boxes[:, 3::6], im_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 4::6] = np.maximum(np.minimum(boxes[:, 4::6], im_shape[0] - 1), 0)
  # z2 >= 0
  boxes[:, 5::6] = np.maximum(np.minimum(boxes[:, 5::6], im_shape[2] - 1), 0)
  return boxes
