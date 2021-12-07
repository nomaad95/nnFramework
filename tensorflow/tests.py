import sys
import random
import numpy as np
import nnfs
import datetime as dt
from nnfs.datasets import spiral_data
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

tensor1 = tf.range(5)

X, y = spiral_data(samples=10, classes=3)
X = tf.convert_to_tensor(X,dtype=tf.float32)
X = tf.split(X, num_or_size_splits=30, axis=0)
print(X)
