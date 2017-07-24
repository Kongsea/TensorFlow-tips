#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# sometimes we want to operate upon a tensor element-wise
# for example:
# we want to set elements greater than 100 in a tensor to 100.
# we can use the funtion tf.map_fn().


def set_threshold(tensor):
  tensor = tf.cond(tensor > 100, lambda: 100., lambda: tensor)
  return tensor


def process(ta, tb):
  ta = tf.map_fn(set_threshold, ta)
  tb = tf.map_fn(set_threshold, tb)
  return ta * tb


def main():
  a = tf.placeholder(tf.float32, shape=(3))
  b = tf.placeholder(tf.float32, shape=(3))
  result = process(a, b)
  with tf.Session() as sess:
    print(sess.run(result, feed_dict={a: np.array([200, 10, 3]), b: np.array([3, 5, 300])}))
    # will print: [300. 50. 300.]


if __name__ == '__main__':
  main()
