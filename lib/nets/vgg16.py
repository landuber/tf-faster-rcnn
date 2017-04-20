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
from nets.network import Network
from model.config import cfg


class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)
    self._arch = 'vgg16'

  def build_network(self, sess, is_training=True):
    self.is_training = is_training
    with tf.variable_scope('vgg_16', 'vgg_16',
                           regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        self._initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self._initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        self._initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self._initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

      rois, net_bv = self.build_bv()
      net_fv = self.build_fv()
      #net_img = self.build_img()

      bv_pool = self._crop_pool_bv_layer(net_bv, rois, "bv/pool5")
      fv_pool = self._crop_pool_fv_layer(net_fv, rois, "fv/pool5")

      self.build_rcnn(tf.add(bv_pool, fv_pool) / 2.0)

      self._score_summaries.update(self._predictions)

      return self._predictions["rois"], self._predictions["cls_prob"], self._predictions["bbox_pred"]


  def build_bv(self):
      net = slim.repeat(self._top_lidar, 2, slim.conv2d, 64, [3, 3],
                        trainable=self.is_training, scope='bv/conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='bv/pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=self.is_training, scope='bv/conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='bv/pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=self.is_training, scope='bv/conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='bv/pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=self.is_training, scope='bv/conv4')
      # Remove the 4th pooling operation for BirdsView rpn
      #net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=self.is_training, scope='bv/conv5')
      self._layers['bv/conv5_3'] = net
      self._act_summaries.append(net)

      # build the anchors for the image
      self._anchor_component()



      # rpn
      size = tf.shape(net)
      #2x deconv per the paper for the proposal net
      rpn = tf.image.resize_images(net, [size[1] * 2, size[2] * 2])
      rpn = slim.conv2d(rpn, 512, [3, 3], trainable=self.is_training, weights_initializer=self._initializer, scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_scales * 3 * 2, [1, 1], trainable=self.is_training,
                                  weights_initializer=self._initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_scales * 6, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_scales * 3 * 6, [1, 1], trainable=self.is_training,
                                  weights_initializer=self._initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if self.is_training:
        # Get all top scoring anchors after filtering via non-maximal suppression
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        # Using IoU determine the labels for each anchor type at all locations
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError


      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      self._predictions["rois"] = rois

      return rois, net

  def build_fv(self):
      net = slim.repeat(self._front_lidar, 2, slim.conv2d, 64, [3, 3],
                        trainable=self.is_training, scope='fv/conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='fv/pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=self.is_training, scope='fv/conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='fv/pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=self.is_training, scope='fv/conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='fv/pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=self.is_training, scope='fv/conv4')
      # Remove the 4th pooling operation for BirdsView rpn
      #net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=self.is_training, scope='fv/conv5')
      self._layers['fv/conv5_3'] = net
      self._act_summaries.append(net)

      return net

  def build_img(self):
      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                        trainable=self.is_training, scope='im/conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='im/pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=self.is_training, scope='im/conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='img/pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=self.is_training, scope='im/conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='im/pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=self.is_training, scope='im/conv4')
      # Remove the 4th pooling operation for BirdsView rpn
      #net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=self.is_training, scope='im/conv5')
      self._layers['im/conv5_3'] = net
      self._act_summaries.append(net)
      # doing bilinear upsampling
      # 4x deconv for lidar
      size = tf.shape(net)
      net = tf.image.resize_images(net, [size[1] * 2, size[2] * 2])

      return net

  def roi_pool(self, rois, net_bv, net_fv, net_img):
      # pooling
      if cfg.POOLING_MODE == 'crop':
        #todo: implement pool transformation and mean calculation
        mean_pool = self._crop_pool_layer(net_bv, rois, "bv/pool5")
        return mean_pool
      else:
        raise NotImplementedError

  def build_rcnn(self, pool5):
      # rcnn
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if self.is_training:
        fc6 = slim.dropout(fc6, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if self.is_training:
        fc7 = slim.dropout(fc7, scope='dropout7')
      # Adding fc8 for the lidar
      fc8 = slim.fully_connected(fc7, 4096, scope='fc8')
      if self.is_training:
        fc8 = slim.dropout(fc8, scope='dropout8')
      cls_score = slim.fully_connected(fc8, self._num_classes, weights_initializer=self._initializer, trainable=self.is_training,
                              activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 6, weights_initializer=self._initializer_bbox,
                              trainable=self.is_training,
                              activation_fn=None, scope='bbox_pred')
      self._predictions["cls_score"] = cls_score
      self._predictions["cls_prob"] = cls_prob
      self._predictions["bbox_pred"] = bbox_pred

