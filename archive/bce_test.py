import torch
from matplotlib import pyplot as plt
import numpy as np

# Define BCE loss
bce = torch.nn.BCELoss()

# Generate x values from 0 to 1
x_values = np.linspace(0, 1, 100)

# Compute BCE loss for each x value
bce_ones = [
    bce(
        torch.tensor([x], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)
    ).item()
    for x in x_values
]
bce_zeros = [
    bce(
        torch.tensor([x], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)
    ).item()
    for x in x_values
]

# Plot BCE(x, 1) and BCE(x, 0)
plt.figure(figsize=(10, 5))
plt.plot(x_values, bce_ones, label="BCE(x, 1)", color="blue")
plt.plot(x_values, bce_zeros, label="BCE(x, 0)", color="red")
plt.xlabel("x")
plt.ylabel("BCE Loss")
plt.title("Binary Cross Entropy Loss")
plt.legend()
plt.grid(True)
plt.savefig("BCE_Loss.png")  # Save the plot as a PNG file
