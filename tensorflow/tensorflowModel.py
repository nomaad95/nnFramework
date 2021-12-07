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
    with tf.device('/device:GPU:0'):
        weights = tf.experimental.numpy.random.randn(n_inputs, n_neurons)
        biases = tf.zeros((1, n_neurons))
        #Ensure the type of the tensors are float32
        weights = weights.astype('float32')
        biases = biases.astype('float32')
        return [weights, biases]

#Computes the output of the first layer
@tf.function
def forward1(inputs, weights, biases):
    with tf.device('/device:GPU:0'):

    #tf.split(inputs, num_or_size_splits=30, axis=0)
        output = tf.tensordot(inputs, weights, axes=1) + biases
        return output


@tf.function
def relu(inputs):
    with tf.device('/device:GPU:0'):
        output = tf.math.maximum(tf.constant(0).astype('float32'),inputs)
        return output

@tf.function
def softmax(inputs):
    with tf.device('/device:GPU:0'):
        exp_values = tf.math.exp(tf.transpose(tf.transpose(inputs) - tf.reduce_max(inputs, axis=1)))
    #exp_values = tf.math.exp(inputs - tf.reduce_max(inputs))
    #exp_values = tf.math.reduce_sum(exp_values, axis=1)

        probabilities = tf.transpose(exp_values) / tf.math.reduce_sum(exp_values,axis=1)
        return tf.transpose(probabilities)
"""
@tf.function
def datasets(rows_number):
    dataset = []
    for i in range(rows_number):
        dataset.append(tf.constant([random.uniform(0,1), random.uniform(0,1)]).astype('float32'))
    return dataset
"""
#traitement nn

X, y = spiral_data(samples=100000, classes=3)
#X = tf.convert_to_tensor(X,dtype=tf.float32)
#X = tf.split(X, num_or_size_splits=30, axis=0)
print(X)

#dataset = tf.data.Dataset.from_tensor_slices(X)

dense1 = init(2,3)
start = dt.datetime.now()
forwardDense1 = forward1(X, dense1[0], dense1[1])
reluTensor = relu(forwardDense1)
softmaxTensor = softmax(reluTensor)
end = dt.datetime.now()

print("exp_values", softmaxTensor)

print("execution time:", end - start)
"""



x = tf.constant([0.10738789,0.02852226], dtype=tf.float32)
x = x.astype('float32')



dense1 = init(2,3)
forwardDense1 = forward1(x, dense1[0], dense1[1])
print("forward", forwardDense1)

reluTensor = relu(forwardDense1)
print("reluTensor", reluTensor)

softmaxTensor = softmax(reluTensor)
print("softmaxTensor", softmaxTensor)

"""
#fin traitement nn


"""
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

start = dt.datetime.now()

activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

end = dt.datetime.now()

print(activation2.output[:5])

print(end - start)
"""
