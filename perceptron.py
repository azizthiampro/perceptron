import numpy as np


class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def train(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)


# Logical AND function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d_and = np.array([0, 0, 0, 1])

and_perceptron = Perceptron(input_size=2)
and_perceptron.train(X, d_and)
print("AND Perceptron weights:", and_perceptron.W)

# Logical OR function
d_or = np.array([0, 1, 1, 1])

or_perceptron = Perceptron(input_size=2)
or_perceptron.train(X, d_or)
print("OR Perceptron weights:", or_perceptron.W)

# Logical XOR function
d_xor = np.array([0, 1, 1, 0])


class XORPerceptron:
    def __init__(self):
        self.hl1 = Perceptron(input_size=2)
        self.hl2 = Perceptron(input_size=2)
        self.ol = Perceptron(input_size=2)

    def predict(self, x):
        y1 = self.hl1.predict(x)
        y2 = self.hl2.predict(x)
        y = self.ol.predict(np.array([y1, y2]))
        return y


xor_perceptron = XORPerceptron()
for i in range(1000):  # Training multiple times for better convergence
    xor_perceptron.hl1.train(X, d_xor)
    xor_perceptron.hl2.train(X, d_xor)
    xor_perceptron.ol.train(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), d_xor)

print("XOR Perceptron (Multilayer) weights:")
print("Hidden Layer 1:", xor_perceptron.hl1.W)
print("Hidden Layer 2:", xor_perceptron.hl2.W)
print("Output Layer:", xor_perceptron.ol.W)

# Results
# Logical AND function
print("\nGround truth for AND:", d_and)
print("Results for AND:")
for i in range(len(X)):
    print("Input:", X[i], "Expected:", d_and[i], "Predicted:", and_perceptron.predict(X[i]))

# Logical OR function
print("\nGround truth for OR:", d_or)
print("Results for OR:")
for i in range(len(X)):
    print("Input:", X[i], "Expected:", d_or[i], "Predicted:", or_perceptron.predict(X[i]))

# Logical XOR function
print("\nGround truth for XOR:", d_xor)
print("Results for XOR:")
for i in range(len(X)):
    print("Input:", X[i], "Expected:", d_xor[i], "Predicted:", xor_perceptron.predict(X[i]))



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros(self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output

    def backward(self, X, y, output):
        # Backpropagation
        error = y - output
        d_output = error
        d_hidden_output = d_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_hidden_output += np.sum(d_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_output) * self.learning_rate
        self.bias_input_hidden += np.sum(d_hidden_output) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return np.round(self.forward(X))


# XOR truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the neural network
input_size = 2
hidden_size = 2  # You can experiment with different values
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, epochs=10000)

# Predict XOR outputs
print("Predictions for XOR with neural network:")
for i in range(len(X)):
    prediction = nn.predict(X[i])[0]  # Access scalar prediction directly
    print("Input:", X[i], "Predicted:", int(prediction))
