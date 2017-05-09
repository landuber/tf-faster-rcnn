# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
import glob

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import prep_im_for_blob, im_list_to_blob, lidar_list_to_blob

from model.common import *
from model.config import cfg, get_output_dir
from model.bbox_transform import corner_transform_inv
from model.boxes3d import *


def _get_lidar_blob(lidar_path):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  processed_lidars = []
  lidar = np.fromfile(lidar_path, dtype=np.float32)
  lidar = lidar.reshape((-1, 4))
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
    indices = np.where((q_lidar[:,0] < Xn) & (q_lidar[:,0] >= X0) & (q_lidar[:, 1] < Yn) & (q_lidar[:, 1] >= Y0) & (q_lidar[:,2] < Zn) & (q_lidar[:,2] >= Z0))[0]
    q_lidar = q_lidar[indices, :]
    #print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    for l in q_lidar:
        yy,xx,zz = -int(l[0]-X0),-int(l[1]-Y0),int(l[2]-Z0)
        height = max(0,l[3]-TOP_Z_MIN)
        top[yy,xx,Zn+1] = top[yy,xx,Zn+1] + 1
        if top[yy, xx, zz] < height:
            top[yy,xx,zz] = height
        if top[yy, xx, Zn] < l[4]:
            top[yy,xx,Zn] = l[4]

    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)


    return top

def lidar_to_front_tensor(lidar):
    THETA0,THETAn = 0, int((HORIZONTAL_MAX-HORIZONTAL_MIN)/HORIZONTAL_RESOLUTION)
    PHI0, PHIn = 0, int((VERTICAL_MAX-VERTICAL_MIN)/VERTICAL_RESOLUTION)
    indices = np.where((lidar[:, 0] > 0.0))[0]

    width = THETAn - THETA0
    height = PHIn - PHI0

    pxs=lidar[indices,0]
    pys=lidar[indices,1]
    pzs=lidar[indices,2]
    prs=lidar[indices,3]

    cs = ((np.arctan2(pxs, -pys) - HORIZONTAL_MIN) / HORIZONTAL_RESOLUTION).astype(np.int32)
    rs = ((np.arctan2(pzs, np.hypot(pxs, pys)) - VERTICAL_MIN) / VERTICAL_RESOLUTION).astype(np.int32)
    ds = np.hypot(pxs, pys)


    rcs = np.vstack((rs, cs, pzs, ds, prs)).T
    indices = np.where((rcs[:,0] < PHIn) & (rcs[:,0] >= PHI0) & (rcs[:, 1] < THETAn) & (rcs[:, 1] >= THETA0))[0]
    rcs = rcs[indices, :]
    front = np.zeros(shape=(height,width,3), dtype=np.float32)
    front[:, 0] = -1.73

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

    front[:, :, 0] = front[:, :, 0]-np.min(front[:, :, 0])
    front[:, :, 0] = (front[:, :, 0]/np.max(front[:, :, 0])*255).astype(np.uint8)
    front[:, :, 1] = front[:, :, 1]-np.min(front[:, :, 1])
    front[:, :, 1] = (front[:, :, 1]/np.max(front[:, :, 1])*255).astype(np.uint8)
    front[:, :, 2] = front[:, :, 2]-np.min(front[:, :, 2])
    front[:, :, 2] = (front[:, :, 2]/np.max(front[:, :, 2])*255).astype(np.uint8)
    front = np.dstack((front[:,:, 0], front[:,:, 1], front[:,:, 2])).astype(np.uint8)
        
    return front

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scale_factors


def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im, lidar_path):
  im_blob, im_scales = _get_image_blob(im)
  top_lidar_blob, front_lidar_blob = _get_lidar_blob(lidar_path)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  blobs = {'image': im_blob, 'top_lidar': top_lidar_blob, 'front_lidar': front_lidar_blob}
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)
  blobs['top_lidar_info'] = np.array(
    [[top_lidar_blob.shape[1], top_lidar_blob.shape[2], top_lidar_blob.shape[3]]],
    dtype=np.float32)
  blobs['front_lidar_info'] = np.array(
    [[front_lidar_blob.shape[1], front_lidar_blob.shape[2], front_lidar_blob.shape[3]]],
    dtype=np.float32)

  _, scores, corner_pred, rois = net.test(sess, blobs)
  
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  corners = top_box_to_lidar_box(rois[:, 1:7])
  scores = np.reshape(scores, [scores.shape[0], -1])
  corner_pred = np.reshape(corner_pred, [corner_pred.shape[0], -1])
  # Apply bounding-box regression deltas
  corner_deltas = corner_pred
  _, pred_corners = corner_transform_inv(corners, box_deltas)

  return scores, pred_corners

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, test_path, weights_filename, max_per_image=100, thresh=0.05):
  
  _imagedir = os.path.join(test_path, 'JPEGImages')
  _lidardir = os.path.join(test_path, 'Lidar')

  img_files = glob.glob(os.path.join(_imagedir, '*.jpg'))
  lidar_files = glob.glob(os.path.join(_lidardir, '*.bin'))  

  img_files.sort()
  lidar_files.sort()
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(img_files)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 25 array of detections in
  #  (x1, y1, z1, x2, y2, z2,....., score)
  all_corners = [[[] for _ in range(num_images)]
         for _ in range(num_classes)]

  output_dir = get_output_tracklets_dir('TRACKLET_TEST', weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i, file in enumerate(img_files):
    path, basename = os.path.splite(file)
    stem, ext = os.path.splitext(basename)
    im = cv2.imread(imdb.image_path_at(file))

    _t['im_detect'].tic()
    scores, corners = im_detect(sess, net, im, lidar_files[i])
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    #corners = np.delete(corners, np.arange(corners.shape[0])[::num_classes], axis=0)
    corners = corners.reshape((scores.shape[0], num_classes, 8, 3))
    for j in range(1, num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_corners = corners[inds, j, :].reshape((-1, 24))
      cls_dets = np.hstack((cls_corners, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      #todo: add 3D NMS
      #keep = nms(cls_dets, cfg.TEST.NMS)
      #cls_dets = cls_dets[keep, :]
      all_corners[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_corners[j][i][:, -1]
                    for j in range(1, num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, num_classes):
          keep = np.where(all_corners[j][i][:, -1] >= image_thresh)[0]
          all_corners[j][i] = all_corners[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_corners, f, pickle.HIGHEST_PROTOCOL)

  print('Generating  tracklets')
  #todo: generate tracklets

