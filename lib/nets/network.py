# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

try:
  import cPickle as pickle
except ImportError:
  import pickle
from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer

from model.config import cfg
from model.common import *
from model.boxes3d import *


class Network(object):
  def __init__(self, batch_size=1):
    self._feat_stride = [4, ]
    self._feat_compress = [1. / 4., ]
    self._batch_size = batch_size
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}


  def _add_bv_lidar_summary(self, lidar, boxes, name="bv_truth"):
    image = tf.reduce_sum(lidar,axis=-1)
    #
    image = tf.reduce_max(image) - image
    image = (image/tf.reduce_max(image)*255)
    #
    #image = image - tf.reduce_min(image)
    #image = (image/tf.reduce_max(image)*255)
    image    = tf.stack ([image, image, image], axis=-1)
    # dims for normalization
    width  = tf.to_float(tf.shape(image)[2])
    height = tf.to_float(tf.shape(image)[1])
    # from [x1, y1, z1, x2, y2, z2 cls] to normalized [y1, x1, y1, x1]
    cols = tf.unstack(boxes, axis=1)
    boxes = tf.stack([cols[1] / height,
                      cols[0] / width,
                      cols[4] / height,
                      cols[3] / width], axis=1)
    # add batch dimension (assume batch_size==1)
    assert image.get_shape()[0] == 1
    boxes = tf.expand_dims(boxes, dim=0)
    image = tf.image.draw_bounding_boxes(image, boxes)
    
    return tf.summary.image(name, image)

  def _add_fv_lidar_summary(self, lidar, boxes, name="fv_truth"):
    image = lidar
    boxes = tf.slice(boxes, [0, 0], [-1, 6])
    boxes = tf.py_func(fv_projection_layer, [boxes], tf.float32)
    boxes.set_shape([None, 4])
    # dims for normalization
    width  = tf.to_float(tf.shape(image)[2])
    height = tf.to_float(tf.shape(image)[1])
    # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
    cols = tf.unstack(boxes, axis=1)
    boxes = tf.stack([cols[1] / height,
                      cols[0] / width,
                      cols[3] / height,
                      cols[2] / width], axis=1)
    # add batch dimension (assume batch_size==1)
    assert image.get_shape()[0] == 1
    boxes = tf.expand_dims(boxes, dim=0)
    image = tf.image.draw_bounding_boxes(image, boxes)
    
    return tf.summary.image(name, image)

  def _add_img_summary(self, image, boxes, name="img_truth"):
    # add back mean
    image += cfg.PIXEL_MEANS
    boxes = tf.slice(boxes, [0, 0], [-1, 6])
    boxes = tf.py_func(img_projection_layer, [boxes, self._image_info], tf.float32)
    boxes.set_shape([None, 4])
    # bgr to rgb (opencv uses bgr)
    channels = tf.unstack (image, axis=-1)
    image    = tf.stack ([channels[2], channels[1], channels[0]], axis=-1)
    # dims for normalization
    width  = tf.to_float(tf.shape(image)[2])
    height = tf.to_float(tf.shape(image)[1])
    # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
    cols = tf.unstack(boxes, axis=1)
    boxes = tf.stack([cols[1] / height,
                      cols[0] / width,
                      cols[3] / height,
                      cols[2] / width], axis=1)
    # add batch dimension (assume batch_size==1)
    assert image.get_shape()[0] == 1
    boxes = tf.expand_dims(boxes, dim=0)
    image = tf.image.draw_bounding_boxes(image, boxes)
    
    return tf.summary.image(name, image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name == 'rpn_cls_prob_reshape':
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_top_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._top_lidar_info,
                                     self._feat_stride, self._anchors, self._anchor_scales],
                                    [tf.float32, tf.float32])
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._top_lidar_info, self._mode,
                                     self._feat_stride, self._anchors, self._anchor_scales],
                                    [tf.float32, tf.float32])
      rois.set_shape([None, 7])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 4.)[0]

  def _crop_pool_bv_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      # account for the 4x upsampling 
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0] / 2.)
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0] / 2.)
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 4], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 5], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')


  def _crop_pool_fv_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      rois_front = tf.slice(rois, [0, 1], [-1, 6])
      rois_front = tf.py_func(fv_projection_layer, [rois_front], tf.float32)
      rois_front.set_shape([None, 4])
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      # account for the 4x upsampling 
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0] / 2.)
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0] / 2.)
      x1 = tf.slice(rois_front, [0, 0], [-1, 1], name="x1") / width
      y1 = tf.slice(rois_front, [0, 1], [-1, 1], name="y1") / height
      x2 = tf.slice(rois_front, [0, 2], [-1, 1], name="x2") / width
      y2 = tf.slice(rois_front, [0, 3], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')


  def _crop_pool_img_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      rois_img = tf.slice(rois, [0, 1], [-1, 6])
      rois_img = tf.py_func(img_projection_layer, [rois_img, self._image_info], tf.float32)
      rois_img.set_shape([None, 4])
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois_img, [0, 0], [-1, 1], name="x1") / width
      y1 = tf.slice(rois_img, [0, 1], [-1, 1], name="y1") / height
      x2 = tf.slice(rois_img, [0, 2], [-1, 1], name="x2") / width
      y2 = tf.slice(rois_img, [0, 3], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._top_lidar_info, self._feat_stride, self._anchors, self._anchor_scales],
        [tf.float32, tf.float32, tf.float32, tf.float32])

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_scales * 3 * 6])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_scales * 3 * 6])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_scales * 3 * 6])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 7])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 6])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 6])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 6])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      self._score_summaries.update(self._proposal_targets)

      return rois, roi_scores

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      height = tf.to_int32(tf.ceil(self._top_lidar_info[0, 0] / 4.))
      width = tf.to_int32(tf.ceil(self._top_lidar_info[0, 1] / 4.))
      anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                          [height, width,
                                           self._feat_stride, self._anchor_scales],
                                          [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 6])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def build_network(self, sess, is_training=True):
    raise NotImplementedError

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('vgg16-loss_' + self._tag) as scope:
      # RPN, class loss
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      rpn_select = tf.where(tf.not_equal(rpn_label, -1))
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
      rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

      # RCNN, class loss
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._proposal_targets["labels"], [-1])

      cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

      # RCNN, bbox loss
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
      self._losses['total_loss'] = loss

      self._event_summaries.update(self._losses)

    return loss

  def create_architecture(self, sess, mode, num_classes,
                          tag=None, anchor_scales=[8, 16, 32]):
    # calculate the depth of the input tensor dyanamically
    Zn = int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    channel = Zn + 2
    self._top_lidar = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, channel])
    self._front_lidar = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
    self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
    self._image_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
    self._top_lidar_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 7])
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = None
    else:
      biases_regularizer = tf.no_regularizer

    with arg_scope([slim.conv2d, slim.fully_connected], biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      rois, cls_prob, bbox_pred = self.build_network(sess, training)

    layers_to_output = {'rois': rois}
    layers_to_output.update(self._predictions)

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if mode == 'TEST':
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      self._add_losses()
      layers_to_output.update(self._losses)

    val_summaries = []
    with tf.device("/cpu:0"):
      val_summaries.append(self._add_bv_lidar_summary(self._top_lidar, self._gt_boxes))
      val_summaries.append(self._add_fv_lidar_summary(self._front_lidar, self._gt_boxes))
      val_summaries.append(self._add_img_summary(self._image, self._gt_boxes))
      rois = tf.slice(self._predictions["rois"], [0, 1], [-1, 6])
      val_summaries.append(self._add_bv_lidar_summary(self._top_lidar, rois, name="rois_bv"))
      val_summaries.append(self._add_fv_lidar_summary(self._front_lidar, rois, name="rois_fv"))
      val_summaries.append(self._add_img_summary(self._image, rois, name="rois_img"))
      for key, var in self._event_summaries.items():
        val_summaries.append(tf.summary.scalar(key, var))
      for key, var in self._score_summaries.items():
        self._add_score_summary(key, var)
      for var in self._act_summaries:
        self._add_act_summary(var)
      for var in self._train_summaries:
        self._add_train_summary(var)

    self._summary_op = tf.summary.merge_all()
    if not testing:
      self._summary_op_val = tf.summary.merge(val_summaries)

    return layers_to_output

  # only useful during testing mode
  def extract_conv5(self, sess, image):
    feed_dict = {self._top_lidar: image}
    feat = sess.run(self._layers["conv5_3"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_top_lidar(self, sess, image, top_lidar_info):
    feed_dict = {self._top_lidar: image,
                 self._top_lidar_info: top_lidar_info}
    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  def get_summary(self, sess, blobs):
    feed_dict = {self._top_lidar: blobs['top_lidar'], self._front_lidar: blobs['front_lidar'], self._image: blobs['image'], \
            self._image_info: blobs['im_info'], self._top_lidar_info: blobs['top_lidar_info'], self._gt_boxes: blobs['gt_boxes']}

    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._top_lidar: blobs['top_lidar'], self._front_lidar: blobs['front_lidar'], self._image: blobs['image'], \
            self._image_info: blobs['im_info'], self._top_lidar_info: blobs['top_lidar_info'], self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._top_lidar: blobs['top_lidar'], self._front_lidar: blobs['front_lidar'], self._image: blobs['image'], \
            self._image_info: blobs['im_info'], self._top_lidar_info: blobs['top_lidar_info'], self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._top_lidar: blobs['top_lidar'], self._front_lidar: blobs['front_lidar'], self._image: blobs['image'], \
            self._image_info: blobs['im_info'], self._top_lidar_info: blobs['top_lidar_info'], self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)

