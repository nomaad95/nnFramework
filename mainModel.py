import numpy as np
from nnfs.datasets import spiral_data

# Common loss class
class Loss:
    # Calculates the data and the regularization losses
    def calculate(self, output, y):
        # calculates sample losse
        sample_losses = self.forward(output, y)
        # Calculates mean loss
        self.data_loss = np.mean(sample_losses)
        return self.data_loss

class Activation_Softmax:
    # Forward pass
    def forward(self,inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize probabilities
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
            # Flatten output array
            single_output = single_output.reshape(-1,1)
            # Calculate jacobian matrixof the output
            jacobian_matrix = np.diagflat(single_output) - (np.dot(single_output, single_output.T))
            # Calculate sample wise gradient
            # And add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Loss_CategoricalCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clips data to prevent division by 0
        # Clips both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values
        # Only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        self.negative_log_likelyhoods = -np.log(correct_confidences)
        # print("negative likelyhoods", self.negative_log_likelyhoods)
        return self.negative_log_likelyhoods

    def backward(self,dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculates grdient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
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
        # Creates weights momentums
        self.weights_momentums = 0
        # Creates biases biases_momentums
        self.biases_momentums = 0


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
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where inputs values were negative
        self.dinputs[self.inputs <= 0] = 0

class Optimizer_SGD:
    # Initialize optimizer - set settings
    # Learning rate of 1
    def __init__(self, learning_rate, decay, momentum):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay * self.iterations))
            #print(self.current_learning_rate)
        self.learning_rate = self.current_learning_rate

    def post_update_params(self):
        self.iterations += 1

    # Update parameters
    def update_params(self, layer):
        if self.momentum:
            # If layer doesn't contain momentum arrays, creates them
            # filled with zeros
            if not hasattr(layer, 'weights_momentums'):
                layer.weights_momentums = np.zeros_like(layer.weights)
                # If there is not momentum array for weights
                # there is not momentum for biases either
                layer.biases_momentums = np.zeros_like(layer.biases)
            weights_updates = self.momentum * layer.weights_momentums - self.current_learning_rate * layer.dweights
            biases_updates = self.momentum * layer.biases_momentums - self.current_learning_rate * layer.dbiases
            layer.weights_momentums = weights_updates
            layer.biases_momentums = biases_updates
        else:
            weights_updates = - self.current_learning_rate * layer.dweights
            biases_updates = - self.current_learning_rate * layer.dbiases
        # layer.weights_momentums = weights_updates
        layer.weights += -self.learning_rate * layer.dweights
        # layer.biases_momentums = biases_updates
        layer.biases += -self.learning_rate * layer.dbiases

class Optimizer_Adagrad:
    # Initialize optimizer - set settings
    # Learning rate of 1
    def __init__(self, learning_rate, decay, epsilon):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    #Processes learning rate for the next iteration
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay * self.iterations))
            #print(self.current_learning_rate)
        self.learning_rate = self.current_learning_rate

    def post_update_params(self):
        self.iterations += 1

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            # If there is not cache array for weights
            # there is not cache for biases either
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        layer.weights += -self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)
        print(layer.weights)

class Optimizer_RMSProp:
    # Initialize optimizer - set settings
    # Learning rate of 1
    def __init__(self, learning_rate, decay, epsilon, rho):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    #Processes learning rate for the next iteration
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay * self.iterations))
            #print(self.current_learning_rate)
        self.learning_rate = self.current_learning_rate

    def post_update_params(self):
        self.iterations += 1

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            # If there is not cache array for weights
            # there is not cache for biases either
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        layer.weights += -self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)


class Model:
    def __init__(self, dense1, dense2, learning_rate = 0.02,decay=1e-7, epsilon=1e-7, rho=0.999):
        self.optimizer = Optimizer_RMSProp(learning_rate, decay,epsilon, rho)
        self.activation1 = Activation_Relu()
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.dense1 = dense1
        self.dense2 = dense2

    def iterate(self, X, y):
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

model = Model(dense1, dense2)

for i in range(10001):
    model.iterate(X,y)
    model.backward()
    model.optimize()
    if not i % 1000:
        #print(model.dense1.weights)
        print(f'epoch : {i}, ' + f'acc: {model.accuracy: .3f} '
        + f'loss: {model.loss:.4f} ' +
        f'lr: {model.optimizer.learning_rate:0.10f}')
        #print(model.dense1.weights)
