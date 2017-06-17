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
import pandas as pd
import functools
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
import glob

from utils.timer import Timer
from utils.nms import nms_3d
from utils.boxes_grid import get_boxes_grid
from utils.blob import prep_im_for_blob, im_list_to_blob, lidar_list_to_blob

from roi_data_layer.minibatch import *
from model.common import *
from model.config import cfg, get_output_tracklets_dir
from model.bbox_transform import corner_transform_inv
from model.boxes3d import *
from model.bbox_transform import corner_transform_inv
from model.generate_tracklet import *

def _get_lidar_blob(lidar_path):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  processed_lidars = []
  lidar = np.fromfile(lidar_path, dtype=np.float32)
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

def _draw_on_image(im, corners):
  for i in range(corners.shape[0]):
    assert corners[i, :].shape[0] == 8
    points = []
    for k in range(8):
      point_x, point_y = lidar_to_top_coords(corners[i, k, 0],
				       corners[i, k, 1],
				       corners[i, k, 2])
      points.append([point_x, point_y])
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(im, [pts], True, (255, 0, 0))
  return im

def _get_poses(dets):
    delta = dets.max(axis=1) - dets.min(axis=1)
    loc = (dets.min(axis=1) + (delta / 2))
    rot = np.zeros_like(loc, dtype=np.float32)
    poses = np.hstack((loc, rot))
    poses = pd.DataFrame(poses, columns=['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
    return poses.to_dict(orient='records')
  


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

def im_detect(sess, net, lidar_path, num_classes):
  top_lidar_blob, front_lidar_blob = _get_lidar_blob(lidar_path)

  blobs = {'top_lidar': top_lidar_blob}

  blobs['top_lidar_info'] = np.array(
    [[top_lidar_blob.shape[1], top_lidar_blob.shape[2], top_lidar_blob.shape[3]]],
    dtype=np.float32)

  _, scores, corner_pred, rois = net.test(sess, blobs)
  
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  corners = top_box_to_lidar_box(rois[:, 1:7])
  scores = np.reshape(scores, [scores.shape[0], -1])
  corner_pred = np.reshape(corner_pred, [corner_pred.shape[0], -1])
  # Apply bounding-box regression deltas
  corner_deltas = corner_pred
  _, pred_corners = corner_transform_inv(corners, corner_deltas, num_classes)

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

def interpolate_lidar_to_camera(img_files, lidar_files):
  img_df = pd.DataFrame(map(lambda f: os.path.splitext(os.path.split(f)[1])[0], 
			    img_files)
			, columns = ['timestamp'])
  lidar_df = pd.DataFrame(map(lambda f: (os.path.splitext(os.path.split(f)[1])[0], f), 
			    lidar_files)
			, columns = ['timestamp', 'path'])
  img_df['timestamp'] = pd.to_datetime(img_df['timestamp'].astype(int))
  img_df.set_index(['timestamp'], inplace=True)
  img_df.index.rename('index', inplace=True)

  lidar_df['timestamp'] = pd.to_datetime(lidar_df['timestamp'].astype(int))
  lidar_df.set_index(['timestamp'], inplace=True)
  lidar_df.index.rename('index', inplace=True)

  merged = functools.reduce(lambda left, right: pd.merge(left, right, how='outer', 
							 left_index=True, right_index=True),
			   [img_df] + [lidar_df])
  arr, cat = merged['path'].factorize()
  paths = pd.Series(arr).replace(-1, np.nan).interpolate(method='nearest').ffill().bfill().astype('category').cat.rename_categories(cat).astype('str')
  paths = paths.to_frame()
  paths.set_index(merged.index, inplace=True)
  paths = paths.loc[img_df.index]
  return paths.values.T[0]

def top_img(lidar):
  top_tensor = lidar_to_top_tensor(lidar)
  img = np.sum(np.absolute(top_tensor), axis=2)
  img = img - np.min(img)
  img = (img/np.max(img) * 255)
  img = np.dstack((img, img, img)).astype(np.uint8)

  return img

def test_net(sess, net, num_classes, test_path, weights_filename, max_per_image=1, thresh=0.00):
  
  print(test_path)
  _imagedir = os.path.join(test_path, 'JPEGImages')
  _lidardir = os.path.join(test_path, 'Lidar')

  img_files = glob.glob(os.path.join(_imagedir, '*.png'))  
  lidar_files = glob.glob(os.path.join(_lidardir, '*.bin'))  

  img_files.sort()
  lidar_files.sort()
  files = interpolate_lidar_to_camera(img_files, lidar_files)
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_lidars = len(files)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 25 array of detections in
  #  (x1, y1, z1, x2, y2, z2,....., score)
  all_corners = [[[] for _ in range(num_lidars)]
         for _ in range(num_classes)]

  #output_dir = get_output_tracklets_dir('TRACKLET_TEST', weights_filename)
  mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
  output_dir = os.path.join(test_path, 'Detections')
  for j in range(1, num_classes):
    mkdir(os.path.join(output_dir, str(j))) 

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  #tracklet = Tracklet(object_type='car', l=3.83, w=3.83, h=1.35, first_frame=0)
  tracklets = [Tracklet(object_type='car', l=3.83, w=3.83, h=1.35, first_frame=0) for _ in range(1, num_classes)]

  for i, file in enumerate(files):
    path, basename = os.path.split(img_files[i])
    stem, ext = os.path.splitext(basename)

    _t['im_detect'].tic()
    scores, corners = im_detect(sess, net, file, num_classes)
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
      keep = nms_3d(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_corners[j][i] = cls_dets


    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      lidar_scores = np.hstack([all_corners[j][i][:, -1]
                    for j in range(1, num_classes)])
      if len(lidar_scores) > max_per_image:
        lidar_thresh = np.sort(lidar_scores)[-max_per_image]
	#print(lidar_thresh)
        for j in range(1, num_classes):
          keep = np.where(all_corners[j][i][:, -1] >= lidar_thresh)[0]
	  lidar = np.fromfile(file, dtype=np.float32)
	  lidar = lidar.reshape((-1, 4))
	  lidar_img = top_img(lidar)
	  lidar_img = _draw_on_image(lidar_img, corners[keep, j, :])
          tracklets[j-1].poses.extend(_get_poses(corners[keep, j, :]))
	  cv2.imwrite(os.path.join(output_dir, str(j), stem + '.png'), lidar_img)
          all_corners[j][i] = all_corners[j][i][keep, :]
    _t['misc'].toc()


    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_lidars, _t['im_detect'].average_time,
            _t['misc'].average_time))

  for j in range(1, num_classes):
      collection = TrackletCollection()
      collection.tracklets.append(tracklets[j-1])
      tracklet_file = os.path.join(output_dir, 'tracklet_labels_' + str(j) + '.xml')
      collection.write_xml(tracklet_file)

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_corners, f, pickle.HIGHEST_PROTOCOL)

  print('Generating  tracklets')
  #todo: generate tracklets

