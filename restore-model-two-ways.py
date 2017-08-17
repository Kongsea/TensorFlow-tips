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
# model definitions
with tf.Session() as sess:
  saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.ckpt-****.meta'))
  saver.restore(sess, os.path.join(model_path, 'model.ckpt-****'))
  # use tf.trainable_variables() to get all variables in the model
  all_vars = tf.trainable_variables()
  for av in all_vars:
    print('{}: {}'.format(av.name, sess.run(av).shape))
