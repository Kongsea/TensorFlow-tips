#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf

# files under model path
# checkpoint
# model.ckpt-****
# model.ckpt-****.meta

model_path = 'path'

# the first method
# model definitions
saver = tf.train.Saver()
with tf.Session() as sess:
  ckpt = tf.train.get_checkpoint_state(model_path)
  ckpt.model_checkpoint_path = os.path.join(model_path, ckpt.model_checkpoint_path)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Model restored successfully.')

# the second method
# don't need model definitions
# use import_meta_graph to import graph definition
with tf.Session() as sess:
  # use meta file, such as: model.ckpt-1000.meta
  saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.ckpt-****.meta'))
  # use ckpt file, such as: model.ckpt-1000 (file name may be: model.ckpt-1000.data-000...)
  saver.restore(sess, os.path.join(model_path, 'model.ckpt-****'))
  # or use model path: folder_path, which contains 4 files:
  #                                                checkpoint
  #                                                model.ckpt-1000.index
  #                                                model.ckpt-1000.meta
  #                                                model.ckpt-1000.data-00000-of-...
  saver.restore(sess, tf.train.latest_checkpoint('./'))
  # use tf.trainable_variables() to get all variables in the model
  # or operate on the variables as you need
  all_vars = tf.trainable_variables()
  for av in all_vars:
    print('{}: {}'.format(av.name, sess.run(av).shape))
