#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# sometimes we need to check if element of a tensor in elements of another tensor.
# in python, it is very easy, just use:
#   a in b
# However, it is not very easy regarding to tensor.
# But we can use the following method.
# Note: Be sure there is only one element in the first tensor.


# Make sure the first tensor a must be a one element tensor.
def tensor_in_operator(a, b):
  return tf.cast(tf.reduce_sum(tf.cast(tf.equal(a, b), tf.float32)), tf.bool)


def main():
  a1 = tf.constant(2, tf.int32)
  a2 = tf.constant(20, tf.int32)
  b = tf.constant(range(10), tf.int32)
  result1 = tensor_in_operator(a1, b)
  result2 = tensor_in_operator(a2, b)
  with tf.Session() as sess:
    print(sess.run(result1)  # will print True
    print(sess.run(result2)  # will print False


if __name__ == '__main__':
  main()
