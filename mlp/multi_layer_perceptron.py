import numpy as np


class Linear:
    def __init__(self, input_dim, output_dim):
        # Weight and bias tensors
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x  # Save input for backprop
        return np.dot(x, self.W.T) + self.b

    def backward(self, grad_output):
        # grad_output: (m, output_dim)
        self.dW = np.dot(grad_output.T, self.x)  # (output_dim, input_dim)
        self.db = np.sum(grad_output, axis=0, keepdims=True)  # (1, output_dim)
        dx = np.dot(grad_output, self.W)  # (m, input_dim)
        return dx


class Sigmoid:
    def forward(self, x):
        """
        Applies the sigmoid activation function.
        Saves the output for use in backward pass.

        Args:
            x (np.ndarray): Input of any shape

        Returns:
            np.ndarray: Sigmoid output (same shape as input)
        """
        self.out = 1 / (1 + np.exp(-x))  # Save output for backward
        return self.out

    def backward(self, grad_output):
        """
        Computes the gradient of the loss w.r.t. the input of sigmoid.

        Uses the saved output from the forward pass.

        Args:
            grad_output (np.ndarray): Gradient from next layer, shape same as self.out

        Returns:
            np.ndarray: Gradient to propagate to previous layer
        """
        return grad_output * self.out * (1 - self.out)


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = Linear(input_dim, hidden_dim)
        self.act1 = Sigmoid()
        self.layer2 = Linear(hidden_dim, output_dim)
        self.act2 = Sigmoid()

    def forward(self, x):
        self.z1 = self.layer1.forward(x)
        self.a1 = self.act1.forward(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = self.act2.forward(self.z2)
        return self.a2

    def backward(self, grad_output):
        grad_z2 = self.act2.backward(grad_output)
        grad_a1 = self.layer2.backward(grad_z2)
        grad_z1 = self.act1.backward(grad_a1)
        self.layer1.backward(grad_z1)


def mse_loss(pred, target):
    loss = np.mean((pred - target) ** 2)
    grad = 2 * (pred - target) / pred.shape[0]
    return loss, grad
