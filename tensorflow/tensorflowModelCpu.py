import sys
import random
import numpy as np
import nnfs
import datetime as dt
from nnfs.datasets import spiral_data
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

nnfs.init()
np.random.seed(0)

# initialisation of the weights and inputs
@tf.function
def init(n_inputs, n_neurons):
    with tf.device('/device:CPU:0'):
        weights = tf.experimental.numpy.random.randn(n_inputs, n_neurons)
        biases = tf.zeros((1, n_neurons))
        #Ensure the type of the tensors are float32
        weights = weights.astype('float32')
        biases = biases.astype('float32')
        return [weights, biases]

#Computes the output of the first layer
@tf.function
def forward1(inputs, weights, biases):
    with tf.device('/device:CPU:0'):
        output = tf.tensordot(inputs, weights, axes=1) + biases
        return output


@tf.function
def relu(inputs):
    with tf.device('/device:CPU:0'):
        output = tf.math.maximum(tf.constant(0).astype('float32'),inputs)
        return output

@tf.function
def softmax(inputs):
    with tf.device('/device:CPU:0'):
        exp_values = tf.math.exp(tf.transpose(tf.transpose(inputs) - tf.reduce_max(inputs, axis=1)))
        #Does an exponential normalization for each rows of the tensor
        #Uses transposed tensor to prevent from shape error
        probabilities = tf.transpose(exp_values) / tf.math.reduce_sum(exp_values,axis=1)
        return tf.transpose(probabilities)


X, y = spiral_data(samples=100000, classes=3)
print(X)


dense1 = init(2,3)
start = dt.datetime.now()
forwardDense1 = forward1(X, dense1[0], dense1[1])
reluTensor = relu(forwardDense1)
softmaxTensor = softmax(reluTensor)
end = dt.datetime.now()

print("exp_values", softmaxTensor)

print("execution time:", end - start)
