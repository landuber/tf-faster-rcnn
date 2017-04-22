# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.common import *

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

#todo: determine the appropriate base_size from the paper
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """

  base_anchor = np.array([1, 1, 1, base_size, base_size, ANCHOR_DEPTH]) - 1
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])])
  return anchors


def _whctrs(anchor):
  """
  Return width, height, depth, x center, y center and z center for an anchor (window).
  """

  w = anchor[3] - anchor[0] + 1
  h = anchor[4] - anchor[1] + 1
  d = anchor[5] - anchor[2] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  z_ctr = anchor[2] + 0.5 * (d - 1)
  return w, h, d, x_ctr, y_ctr, z_ctr


def _mkanchors(ws, hs, ds, x_ctr, y_ctr, z_ctr):
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  ds = ds[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       z_ctr - 0.5 * (ds - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1),
                       z_ctr + 0.5 * (ds -1)))
  return anchors



def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """

  w, h, d, x_ctr, y_ctr, z_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  ds = np.ones((ws.shape[0]), dtype=np.float32) * d
  anchors = _mkanchors(ws, hs, ds, x_ctr, y_ctr, z_ctr)
  return anchors


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.
  """

  w, h, d, x_ctr, y_ctr, z_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  ds = np.ones((ws.shape[0]), dtype=np.float32) * d
  anchors = _mkanchors(ws, hs, ds, x_ctr, y_ctr, z_ctr)
  return anchors


if __name__ == '__main__':
  import time

  t = time.time()
  a = generate_anchors()
  print(time.time() - t)
  print(a)
  from IPython import embed;

  embed()
