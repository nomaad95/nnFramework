import numpy as np
import nnfs
import datetime as dt
from nnfs.datasets import spiral_data
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

nnfs.init()
np.random.seed(0)

"""
class Layer_Dense:

    @tf.function
    def __init__(self, n_inputs, n_neurons):
        self.weights = tf.experimental.numpy.random.randn(n_inputs, n_neurons)
        self.biases = tf.zeros((1, n_neurons))
    @tf.function
    def forward(self, inputs):
        print(inputs)
        print(self.weights)
        self.output = tf.tensordot(inputs, self.weights, axes=1) + self.biases



class Activation_ReLu:
    @tf.function
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    @tf.function
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites
"""


X, y = spiral_data(samples=10, classes=3)
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLu()
a = tf.constant(np.random.rand(10,2), dtype=tf.float64)
#print(tf.convert_to_tensor(a, dtype=tf.float64))


dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

start = dt.datetime.now()
dense1.forward(a)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

end = dt.datetime.now()

print(activation2.output[:5])

print(end - start)

"""

layer1 = Layer_Dense(2,5)

activation1 = Activation_ReLu()
#layer2 = Layer_Dense(5,2)

layer1.forward(X)
#layer2.forward(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
"""
