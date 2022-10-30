F   import numpy as np
from nnfs.datasets import spiral_data

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # print("sample losses", sample_losses)
        self.data_loss = np.mean(sample_losses)
        return self.data_loss
class Activation_Softmax:
    def forward(self,inputs):
        self.inputs = inputs
        # Pass inputs in the first neuron
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Creates uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate output and gradients
        # loop through map of (output, dvalues)
        # index : number of iteration
        # single_output: self.output
        # single_dvalues: dvalues
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # print("single output")
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - (np.dot(single_output, single_output.T))
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    	#prevents from 0 values and errors
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        self.negative_log_likelyhoods = -np.log(correct_confidences)
        # print("negative likelyhoods", self.negative_log_likelyhoods)
        return self.negative_log_likelyhoods

    def backward(self,dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        # final output of the model
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        # array with scores of likelyhood for each class
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded
        # Turn them into discrete values
        if(len(y_true.shape) == 2):
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        # loop through dinputs rows
        # and substract column at y_true position for each rows
        # (simplier way to do Softmax backward)
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # random initialization of weights array
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        # random initialization of biases array
        self.biases = np.zeros((1, n_neurons))
        #self.weights = self.weights.astype('float32')
        #self.biases = self.biases.astype('float32')

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = self.dinputs = dvalues.copy()
        self.dinputs[self.dinputs <= 0] = 0

class Optimizer_SGD:
    # Initialize optimizer - set settings
    # Learning rate of 1
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = self.learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1

    # Update parameters
    def update_params(self, layer):
        layer.weights += - self.current_learning_rate * layer.dweights
        layer.dbiases += - self.current_learning_rate * layer.dbiases



class Model:
    def __init__(self):
        self.optimizer = Optimizer_SGD(decay=1e-3)
        self.activation1 = Activation_Relu()
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    def iterate(self,dense1, dense2, X, y):
        self.dense1 = dense1
        self.dense2 = dense2
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.loss = self.loss_activation.forward(self.dense2.output,y)
        #print("last activation output",loss_activation.output)
        # gives the position of the highest probability for each array
        predictions = np.argmax(self.loss_activation.output, axis=1)
        #print("predictions",predictions)
        if len(y.shape) == 2:
            y = np.argmax(y,axis=1)
        # gives the percentage of the highest probability
        # being in the right position
        self.accuracy = np.mean(predictions == y)
        #print("accuracy",accuracy)

    def backward(self):
        # pass final outputs and labels through backward function
        self.loss_activation.backward(self.loss_activation.output, y)
        self.dense2.backward(self.loss_activation.dinputs)
        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)

    def optimize(self):
        self.optimizer.pre_update_params()
        self.optimizer.update_params(dense1)
        self.optimizer.update_params(dense2)
        self.optimizer.post_update_params()



#dataset
# X positions (coordinates x,y)
# y : list of value of labels
X, y = spiral_data(samples=100, classes=3)
# 2 input features and 3 output values
dense1 = Layer_Dense(2,64)
# 3 input features and 3 output values (takes the output of previous layer)
dense2 = Layer_Dense(64,3)

model = Model()

for i in range(10001):
    model.iterate(dense1, dense2, X,y)
    #   print(model.dense1.weights)
    model.backward()
    model.optimize()
    if not i%1000:
        print(f'epoch : {i}, ' + f'acc: {model.accuracy:.3f} '
        + f'loss: {model.loss:.4f} ' + f'lr: {model.optimizer.current_learning_rate:0.10f}')
