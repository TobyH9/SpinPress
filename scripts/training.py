import numpy as np
from mlp.multi_layer_perceptron import MLP, mse_loss
from data_generation.ising_lattice import IsingLattice

# Initialize model
model = MLP(input_dim=100, hidden_dim=200, output_dim=100)
learning_rate = 0.1

# Creating a training example
example = IsingLattice(10, 10, 1, 1)
x_raw = example.lattice
y_raw = example.unfixed_entries

x = x_raw.reshape(1, 100)
y_true = y_raw.reshape(1, 100)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model.forward(x)
    loss, grad_loss = mse_loss(y_pred, y_true)

    # Backward pass
    model.backward(grad_loss)

    # Gradient descent update
    for layer in [model.layer1, model.layer2]:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")
