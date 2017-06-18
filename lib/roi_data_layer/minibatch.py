# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import math
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob, lidar_list_to_blob
from model.common import *


def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  top_lidar_blob, front_lidar_blob = _get_lidar_blob(roidb)

  blobs = {'top_lidar': top_lidar_blob}

  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 7), dtype=np.float32)
  gt_boxes[:, 0:6] = roidb[0]['top_boxes'][gt_inds, :]
  gt_boxes[:, 6] = roidb[0]['gt_classes'][gt_inds]

  gt_corners = np.empty((len(gt_inds), 8, 3), dtype=np.float32)
  gt_corners[:, :, :] = roidb[0]['lidar_boxes'][gt_inds, :]
  blobs['roidb_id'] = roidb[0]['roidb_id']
  blobs['gt_boxes'] = gt_boxes
  blobs['gt_corners'] = gt_corners
  blobs['top_lidar_info'] = np.array(
    [[top_lidar_blob.shape[1], top_lidar_blob.shape[2], top_lidar_blob.shape[3]]],
    dtype=np.float32)

  return blobs

def _get_lidar_blob(roidb):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_lidars = len(roidb)
  processed_lidars = []
  for i in range(num_lidars):
    lidar = np.fromfile(roidb[i]['lidar'], dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    keep = np.where((lidar[:, 0] > 1.8) | (lidar[:, 0] < -1.)
                      &(lidar[:, 1] > 1.) | (lidar[:, 1] < -1.))[0]
    lidar = lidar[keep, :]
    front_lidar = np.empty_like(lidar)
    front_lidar[:] = lidar
    top_lidar = lidar_to_top_tensor(lidar)
    front_lidar = lidar_to_front_tensor(front_lidar)
    processed_lidars.append((top_lidar, front_lidar))

  # Create a blob to hold the input images
  return lidar_list_to_blob(processed_lidars)




def lidar_to_top_tensor(lidar):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    qxs=((pxs-TOP_X_MIN)/TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)/TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)/TOP_Z_DIVISION).astype(np.int32)

    q_lidar = np.vstack((qxs, qys, qzs, pzs, prs)).T
    # constrain it to the box
    indices = np.where((q_lidar[:, 0] < Xn) & (q_lidar[:, 0] >= X0) 
                     & (q_lidar[:, 1] < Yn) & (q_lidar[:, 1] >= Y0) 
                     & (q_lidar[:, 2] < Zn) & (q_lidar[:, 2] >= Z0))[0]
    q_lidar = q_lidar[indices, :]
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    indices = q_lidar.astype(np.int)
    indices[:, 0] *= -1
    indices[:, 1] *= -1
    top[indices[:, 0], indices[:,1], indices[:,2]] = -sys.float_info.max

    for l in q_lidar:
        yy,xx,zz = -int(l[0]-X0),-int(l[1]-Y0),int(l[2]-Z0)
	# height in meters of the lidar point
        height = l[3]
        top[yy,xx,Zn+1] = top[yy,xx,Zn+1] + 1
        if top[yy, xx, zz] < height:
            top[yy,xx,zz] = height
	    top[yy,xx,Zn] = l[4]

    top[:,:,Zn+1] = np.minimum(1., np.log(top[:,:,Zn+1]+1)/np.log(32))

    return top


def lidar_to_front_tensor(lidar):
    THETA0,THETAn = 0, int((HORIZONTAL_MAX-HORIZONTAL_MIN)/HORIZONTAL_RESOLUTION)
    PHI0, PHIn = 0, int((VERTICAL_MAX-VERTICAL_MIN)/VERTICAL_RESOLUTION)
    #indices = np.where((lidar[:, 0] > 0.0))[0]

    width = THETAn - THETA0
    height = PHIn - PHI0

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    cs = ((np.absolute(np.arctan2(pxs, -pys)) - HORIZONTAL_MIN) / HORIZONTAL_RESOLUTION).astype(np.int32)
    rs = ((np.arctan2(pzs, np.hypot(pxs, pys)) - VERTICAL_MIN) / VERTICAL_RESOLUTION).astype(np.int32)
    ds = np.hypot(pxs, pys)


    rcs = np.vstack((rs, cs, pzs, ds, prs)).T
    indices = np.where((rcs[:,0] < PHIn) & (rcs[:,0] >= PHI0) & (rcs[:, 1] < THETAn) & (rcs[:, 1] >= THETA0))[0]
    rcs = rcs[indices, :]
    front = np.zeros(shape=(height,width,3), dtype=np.float32) 

    indices = rcs.astype(np.int)
    indices[:, 0] *= -1
    indices[:, 1] *= -1
    top[indices[:, 0], indices[:,1], :] = -sys.float_info.max

    for rc in rcs:
        yy, xx = -int(rc[0] - PHI0), -int(rc[1] - THETA0) 
        # rc[2] => height
        if front[yy,xx,0] < rc[2]:
            front[yy,xx, 0] = rc[2]
        # rc[3] => distance
        if front[yy,xx,1] < rc[3]:
            front[yy,xx,1] = rc[3]
        # rc[4] => intensity
        if front[yy,xx,2] < rc[4]:
            front[yy,xx,2] = rc[4]

    return front

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
