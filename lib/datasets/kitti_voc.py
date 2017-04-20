# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
from model.common import *


class kitti_voc(imdb):
  def __init__(self, image_set, devkit_path=None):
    imdb.__init__(self, 'kitti3d_' + image_set)
    self._image_set = image_set
    self._devkit_path = self._get_default_path() if devkit_path is None \
      else devkit_path
    self._data_path = self._devkit_path
    self._classes = ('dontcare',  # always index 0
                     'pedestrian',
                     'car',
                     'cyclist')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._lidar_ext = '.bin'
    self._image_index = self._load_image_set_index()
    self._remove_empty_samples()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None,
                   'min_size': 2}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def lidar_path_at(self, i):
    """
    Return the absolute path to lidar i in the image sequence.
    """
    return self.lidar_path_from_index(self._image_index[i])

  def lidar_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    lidar_path = os.path.join(self._data_path, 'Lidar',
                              index + self._lidar_ext)
    assert os.path.exists(lidar_path), \
      'Path does not exist: {}'.format(lidar_path)
    return lidar_path


  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'KITTIVOC3D')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_selective_search_roidb(self, gt_roidb):
    filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                            'selective_search_data',
                                            self.name + '.mat'))
    assert os.path.exists(filename), \
      'Selective search data not found at: {}'.format(filename)
    raw_data = sio.loadmat(filename)['boxes'].ravel()

    box_list = []
    for i in range(raw_data.shape[0]):
      boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
      keep = ds_utils.unique_boxes(boxes)
      boxes = boxes[keep, :]
      keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
      boxes = boxes[keep, :]
      box_list.append(boxes)

    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _remove_empty_samples(self):
      """
      Remove images with zero annotation ()
      """
      print('Remove empty annotations')
      for i in range(len(self._image_index)-1, -1, -1):
            index = self._image_index[i]
            filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
            tree = ET.parse(filename)
            objs = tree.findall('object')
            non_diff_objs = [
                obj for obj in objs if \
                    int(obj.find('difficult').text) == 0 and obj.find('name').text.lower().strip() != 'dontcare']
            num_objs = len(non_diff_objs)
            if num_objs == 0:
                print(index)
                self._image_index.pop(i)
      print('Done. ')


  def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        top_boxes = np.zeros((num_objs, 6), dtype=np.int32)
        lidar_boxes = np.zeros((num_objs, 8, 3), dtype=np.float32)
        dimensions = np.zeros((num_objs, 3), dtype=np.float32)
        locations = np.zeros((num_objs, 3), dtype=np.float32)
        rotations_y = np.zeros((num_objs), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # just the same as gt_classes
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        ishards = np.zeros((num_objs), dtype=np.int32)
        care_inds = np.empty((0), dtype=np.int32)
        dontcare_inds = np.empty((0), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            dim =  obj.find('dimensions')
            loc = obj.find('location')

            # Find all corners
            crs = obj.find('lidar_box').findall('corner')
            lidarb = np.zeros((8, 3), dtype=np.float32)
            for ix, cr in enumerate(crs):
                lidarb[ix, :] = [float(cr.find('x').text),
                                  float(cr.find('y').text),
                                  float(cr.find('z').text)]
            top_box = corners_from_box(lidar_box_to_top_box(lidarb))

            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text) - 1, 0)
            y1 = max(float(bbox.find('ymin').text) - 1, 0)
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            # Load dimensions
            height = float(dim.find('height').text)
            width = float(dim.find('width').text)
            length = float(dim.find('length').text)
            # Load pose
            xp = float(loc.find('x').text)
            yp = float(loc.find('y').text)
            zp = float(loc.find('z').text)
            # Load y rotation
            rot_y = float(obj.find('rotation_y').text)

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            class_name = obj.find('name').text.lower().strip()
            if class_name != 'dontcare':
                care_inds = np.append(care_inds, np.asarray([ix], dtype=np.int32))
            if class_name == 'dontcare':
                dontcare_inds = np.append(dontcare_inds, np.asarray([ix], dtype=np.int32))
                boxes[ix, :] = [x1, y1, x2, y2]
                top_boxes[ix, :] = top_box
                lidar_boxes[ix, :, :] = lidarb
                dimensions[ix, :] = [height, width, length]
                locations[ix, :] = [xp, yp, zp]
                rotations_y[ix] = rot_y
                continue
            cls = self._class_to_ind[class_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            top_boxes[ix, :] = top_box
            lidar_boxes[ix, :, :] = lidarb
            dimensions[ix, :] = [height, width, length]
            locations[ix, :] = [xp, yp, zp]
            rotations_y[ix] = rot_y
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        # deal with dontcare areas
        dontcare_areas = boxes[dontcare_inds, :]
        boxes = boxes[care_inds, :]
        top_boxes = top_boxes[care_inds, :]
        dimensions = dimensions[care_inds, :]
        locations = locations[care_inds, :]
        rotations_y = rotations_y[care_inds]
        gt_classes = gt_classes[care_inds]
        overlaps = overlaps[care_inds, :]
        seg_areas = seg_areas[care_inds]
        ishards = ishards[care_inds]

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'top_boxes': top_boxes,
                'lidar_boxes': lidar_boxes,
                'gt_classes': gt_classes,
                'gt_ishard' : ishards,
                'dontcare_areas' : dontcare_areas,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}




  def corners_from_box(box):
    return np.hstack((box.min(axis=0), box.max(axis=0)))


  def box_from_corners(corners):
    umin,vmin,zmin,umax,vmax,zmax = corners
    box=np.array([[umin, vmin, zmin],
                  [umax, vmin, zmin],
                  [umax, vmax, zmin],
                  [umin, vmax, zmin],
                  [umin, vmin, zmax],
                  [umax, vmin, zmax],
                  [umax, vmax, zmax],
                  [umin, vmax, zmax]])

    return box

  def lidar_box_to_top_box(lidarb):
    x0 = b[0,0]
    y0 = b[0,1]
    x1 = b[1,0]
    y1 = b[1,1]
    x2 = b[2,0]
    y2 = b[2,1]
    x3 = b[3,0]
    y3 = b[3,1]
    u0,v0=lidar_to_top_coords(x0,y0)
    u1,v1=lidar_to_top_coords(x1,y1)
    u2,v2=lidar_to_top_coords(x2,y2)
    u3,v3=lidar_to_top_coords(x3,y3)

    z0 = max(b[0,2], b[1,2], b[2,2], b[3,2]) # top
    z4 = min(b[4,2], b[5,2], b[6,2], b[7,2]) # bottom
    Zn = int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    zmax = int((z0-TOP_Z_MIN)/TOP_Z_DIVISION)
    zmin = int((z4-TOP_Z_MIN)/TOP_Z_DIVISION)

    umin=min(u0,u1,u2,u3)
    umax=max(u0,u1,u2,u3)
    vmin=min(v0,v1,v2,v3)
    vmax=max(v0,v1,v2,v3)


    # start from the top left corner and go clockwise
    top_box = box_from_corners((umin,vmin,zmin,umax,vmax,zmax))

    return top_box

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    filedir = os.path.join(self._devkit_path, 'results', 'KITTI', 'Main')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == 'dontcare':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets', 'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == 'dontcare':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir,
                                     ovthresh=0.5, use_07_metric = use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == 'dontcare':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  d = kitti_voc('trainval')
  res = d.roidb
  from IPython import embed;

  embed()
