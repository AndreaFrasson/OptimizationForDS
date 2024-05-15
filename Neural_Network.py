import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, l1_lambda):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1_lambda = l1_lambda
        self.n = 4 

        # Initialize weights and biases
        sigma_hidden = np.sqrt(6)/np.sqrt(input_size + hidden_size)
        sigma_output = np.sqrt(6)/np.sqrt(hidden_size + output_size)

        self.weights_input_hidden = np.random.uniform(low=-sigma_hidden, high=sigma_hidden, size = (self.input_size, self.hidden_size)) 
        self.bias_hidden = np.random.uniform(low=-sigma_hidden, high=sigma_hidden, size =(1, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(low=-sigma_output, high=sigma_output, size=(self.hidden_size, self.output_size))
        self.bias_output = np.random.uniform(low=-sigma_output, high=sigma_output, size =(1, self.output_size))

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def tanh_derivative(self, x):
        t = self.tanh(x)
        return 1-t**2

    def l1_regularization(self, weights):
        return self.l1_lambda * np.sign(weights)

    def forward(self, X):
        # Forward propagation with tanh activation
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.tanh(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.tanh(self.output_layer_input)

        return self.predicted_output
    

    def backward(self, X, y):
        # Backward propagation    
        output_error = self.predicted_output - y
        output_delta = output_error * self.tanh_derivative(self.predicted_output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.tanh_derivative(self.hidden_layer_output)

        # Compute gradients with L1 regularization
        dW2 = np.dot(self.hidden_layer_output.T, output_delta) / len(y) + self.l1_lambda * np.sign(self.weights_hidden_output)
        db2 = np.sum(output_delta, axis=0) / len(y)
        dW1 = np.dot(X.T, hidden_delta) / len(y) + self.l1_lambda * np.sign(self.weights_input_hidden)
        db1 = np.sum(hidden_delta, axis=0) / len(y)

        return dW1, db1, dW2, db2
    
    def evaluate(self, x, y):

    # Forward pass
        yhat = self.forward(x)

        # compute loss
        mse = 0.5 * np.mean(np.square(yhat-y))
        loss_regularization = np.sum(np.abs(self.weights_input_hidden)) + np.sum(np.abs(self.weights_hidden_output))
        total_loss = mse + self.l1_lambda * loss_regularization

        return total_loss

    def gradient(self, x, y):
        return self.backward(x, y)