#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf

# There are many GPUs on the computer
# We want to use one of them, say, No.2
# use following settings:
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# or run python program through terminal
# use:
# CUDA_VISIBLE_DEVICES=2 python my_script.py

# There is only one GPU on the computer
# TensorFlow will allocate all GPU memory
# use following commands can set the fraction
# How much TensorFlow can allocate GPU memory
# for the program, say, 0.5 of all memory:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# or
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#   ...
