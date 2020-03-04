# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ImgCNN(object):
    '''
    A cnn for image classification.
    '''
    def __init__(self, n_classes, img_height, img_width, img_channel, device_name='/cpu:0'):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, img_height, img_width, img_channel], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        with tf.device(device_name):
            self.input_image = tf.reshape(self.input_x, [-1,img_height,img_width,img_channel])
            with tf.name_scope('conv_layer_1'):
                filter_shape_1 = [5,5,img_channel,8]
                self.h_conv_1 = self.conv2d(x=self.input_x, W=self.w_variable(shape=filter_shape_1), stride=1, padding='SAME')
                self.h_conv_1 = tf.nn.relu(features=self.h_conv_1, name='relu_conv_1')
            with tf.name_scope('pooling_layer_1'):
                self.h_pool_1 = self.max_pool(x=self.h_conv_1, ksize=2, stride=2, padding='SAME')   # shape: [112 * 112 * 8]

            with tf.name_scope('conv_layer_2'):
                filter_shape_2 = [3,3,8,16]
                self.h_conv_2 = self.conv2d(x=self.h_pool_1, W=self.w_variable(shape=filter_shape_2), stride=1, padding='SAME')
                self.h_conv_2 = tf.nn.relu(features=self.h_conv_2, name='relu_conv_2')
            with tf.name_scope('pooling_layer_2'):
                self.h_pool_2 = self.max_pool(x=self.h_conv_2, ksize=2, stride=2, padding='SAME')   # shape: [56 * 56 * 16]

            with tf.name_scope('conv_layer_3'):
                filter_shape_3 = [3,3,16,32]
                self.h_conv_3 = self.conv2d(x=self.h_pool_2, W=self.w_variable(shape=filter_shape_3), stride=1, padding='SAME')
                self.h_conv_3 = tf.nn.relu(features=self.h_conv_3, name='relu_conv_3')
            with tf.name_scope('pooling_layer_3'):
                self.h_pool_3 = self.max_pool(x=self.h_conv_3, ksize=2, stride=2, padding='SAME')   # shape: [28 * 28 * 32]

            num_total_unit = self.h_pool_3.get_shape()[1:4].num_elements()
            self.h_pool_3_flat = tf.reshape(self.h_pool_3, shape=[-1, num_total_unit]) #铺平

            with tf.name_scope('fc_layer_1'):
                self.h_fc_1 = self.fc_layer(self.h_pool_3_flat, num_total_unit, 128, activation_function=tf.nn.relu) #全连接

            with tf.name_scope('dropout'):
                 self.h_drop = tf.nn.dropout(self.h_fc_1, keep_prob=self.dropout_keep_prob, name='h_drop')

            with tf.name_scope('fc_layer_2'):
                self.output = self.fc_layer(self.h_drop, 128, n_classes, activation_function=None)

        with tf.device('/cpu:0'):
            with tf.name_scope('prediction'):
                self.y_pred = tf.argmax(input=self.output, axis=1, name='y_pred')

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.output), name='cross_entropy_loss')

            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.y_pred, tf.argmax(self.input_y, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

    def w_variable(self, shape):
        return tf.Variable(initial_value=tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1), dtype=tf.float32, name='W')

    def b_variable(self, shape):
        return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), dtype=tf.float32, name='b')

    def conv2d(self, x, W, stride, padding='SAME'):
        return tf.nn.conv2d(input=x, filter=W, strides=[1,stride,stride,1], padding=padding, name='conv')

    def max_pool(self, x, ksize, stride, padding='VALID'):
        return tf.nn.max_pool(value=x, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding=padding, name='max-pool')

    def fc_layer(self, x, in_size, out_size, activation_function=None):
        w = self.w_variable(shape=[in_size, out_size])
        b = self.b_variable(shape=[out_size])
        z = tf.nn.xw_plus_b(x, w, b, name='Wx_plus_b')
        if activation_function is None:
            outputs = z
        else:
            outputs = activation_function(z)
        return outputs
