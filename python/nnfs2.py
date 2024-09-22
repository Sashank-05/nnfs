import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(output, y_true):
    return -np.sum(y_true * np.log(output)) / y_true.shape[0]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# load weights
try:
    weights = np.load('weights.npy')
    bias = np.load('bias.npy')
    output_weights = np.load('output_weights.npy')
    output_bias = np.load('output_bias.npy')
except:
    weights = np.random.randn(784, 128) * 0.01
    bias = np.zeros((1, 128))
    output_weights = np.random.randn(128, 10) * 0.01
    output_bias = np.zeros((1, 10))


def forward_prop(input_data):
    hidden_input = np.dot(input_data, weights) + bias
    hidden_output = sigmoid(hidden_input)
    out_input = np.dot(hidden_output, output_weights) + output_bias
    out_output = softmax(out_input)
    return hidden_output, out_output


def backprop(input_data, hidden_output, out_output, labels):
    global weights, bias, output_weights, output_bias

    # Output layer gradients
    dO = out_output - labels
    dOW = np.dot(hidden_output.T, dO) / input_data.shape[0]
    dOB = np.sum(dO, axis=0, keepdims=True) / input_data.shape[0]

    # Hidden layer gradients
    dH = np.dot(dO, output_weights.T) * sigmoid_derivative(hidden_output)
    dW = np.dot(input_data.T, dH) / input_data.shape[0]
    dB = np.sum(dH, axis=0, keepdims=True) / input_data.shape[0]

    # Update weights and biases
    weights -= learning_rate * dW
    bias -= learning_rate * dB
    output_weights -= learning_rate * dOW
    output_bias -= learning_rate * dOB


learning_rate = 0.01

losses = []

for epoch in range(200):
    hidden_output, out_output = forward_prop(x_train)
    loss = cross_entropy_loss(out_output, y_train)
    backprop(x_train, hidden_output, out_output, y_train)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
    losses.append(loss)

# Test
hidden_output, out_output = forward_prop(x_test)
test_loss = cross_entropy_loss(out_output, y_test)
print(f'Test Loss: {test_loss:.4f}')

# Save weights
np.save('weights.npy', weights)
np.save('bias.npy', bias)
np.save('output_weights.npy', output_weights)
np.save('output_bias.npy', output_bias)

# Predict
img = int(input('Enter image number: '))
plt.imshow(x_test[img].reshape(28, 28), cmap='gray')
plt.show()
prediction = np.argmax(out_output[img])
print(f'Predicted class: {prediction}')

# plot losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
