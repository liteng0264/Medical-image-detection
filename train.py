# -*- coding: utf-8 -*-

import sys
import os
import time
import datetime
import gflags
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import data_helper
from img_cnn import ImgCNN


### parameters ###
# ===============================================
FLAGS = gflags.FLAGS

# data loading parameters
gflags.DEFINE_string('train_data_dir', './inputs/train/', 'Directory of the training data.')
gflags.DEFINE_float('dev_sample_percentage', 0.01, 'Percentage of the training data to user for validation (dev set).')

# model parameters
gflags.DEFINE_integer('img_height', 224, 'The height of the image for training (default: 227).')
gflags.DEFINE_integer('img_width', 224, 'The width of the image for training (default: 227).')
gflags.DEFINE_integer('img_channels', 1, 'The number of channels of the image for training (default: 3).')
gflags.DEFINE_float('dropout_keep_prob', 0.7, 'Dropout keep probability (default: 0.7).')

# training parameters
gflags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')
gflags.DEFINE_integer('batch_size', 32, 'The batch size for each train step.')
gflags.DEFINE_integer('num_epochs', 200, 'Number of training epochs (default: 200).')
gflags.DEFINE_integer('evaluate_every', 100, 'Evaluate model on dev set after this many of steps (default: 100).')
gflags.DEFINE_integer('checkpoint_every', 100, 'Save model after this many steps (default: 100).')
gflags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store (default: 5).')

# device parameters
gflags.DEFINE_string('device_name', '/cpu:0', 'Device name for training.')
gflags.DEFINE_bool('allow_soft_placement', True, 'Allow device soft device placement.')
gflags.DEFINE_bool('log_device_placement', False, 'Log placement of ops on devices.')

FLAGS(sys.argv)
# show parameters
print('\nPARAMETERS:')
print('================================')
for attr, value in FLAGS.flag_values_dict().items():
    print('{0}: {1}'.format(attr.lower(), value))
print('================================\n\n')


### data preparation ###
# ===============================================

# load data
print('Loading data...\n')
x_path, y = data_helper.get_filenames_and_labels(FLAGS.train_data_dir)


# split train/dev set
split_index = -int(float(len(y)) * FLAGS.dev_sample_percentage)
x_path_train, x_path_dev = x_path[:split_index], x_path[split_index:]
y_train, y_dev = y[:split_index], y[split_index:]

del x_path, y

x_dev = []
for i in range(len(x_path_dev)):
    img_data = data_helper.img_resize(img_path=x_path_dev[i], img_height=FLAGS.img_height, img_width=FLAGS.img_width)
    #img_data_min, img_data_max = np.min(img_data), np.max(img_data)
    #img_data = (img_data - img_data_min) / (img_data_max - img_data_min)
    img_data = data_helper.rgb2gray(img_data)
    x_dev.append(img_data)
x_dev = np.array(x_dev)
y_dev = np.array(y_dev)


input('press enter to start training...\n\n')
### training
# ===============================================
print('start training...\n')
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = ImgCNN(
            n_classes=y_train.shape[1],
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            img_channel=FLAGS.img_channels,
            device_name=FLAGS.device_name
            )

        # define training procedure
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.curdir, 'log', timestamp))
        print('Writing log to {}\n'.format(out_dir))

        # summary the input images
        tf.summary.image('input_image', cnn.input_image, max_outputs=FLAGS.batch_size)

        # summary all the trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(name=var.name, values=var)

        # summary loss and accuracy
        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        # train summaries
        # train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph())

        # test summaries
        # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_op = tf.summary.merge_all()
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())

        # checkpointing, tensorflow assumes this directory already existed, so we need to create it
        checkpoint_dir = os.path.join(out_dir, 'checkpoints')
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        def train_step(x_batch, y_batch, writer=None):
            '''
            A single training step.
            '''
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            timestr = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(timestr, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            '''
            Evaluate the model on test set.
            '''
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            timestr = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(timestr, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        ### training loop
        # train loop, for each batch
        sess.run(tf.global_variables_initializer())
        batches = data_helper.batch_iter(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, img_path_list=x_path_train, label_list=y_train,
            img_height=FLAGS.img_height, img_width=FLAGS.img_width)
        for x_batch, y_batch in batches:
            train_step(x_batch, y_batch, writer=train_summary_writer)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print('\nEvaluation on dev set:')
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print('')
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess=sess, save_path=checkpoint_prefix, global_step=global_step)
                print('\nSaved model checkpoint to {}\n'.format(path))

# end
print('\n--- Done! ---\n')
