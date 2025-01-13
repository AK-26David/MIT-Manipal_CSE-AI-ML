import torch

# Define the value of x (can be a scalar or a tensor)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # Example for multiple x values, requires_grad=True to track gradients

# Compute the polynomial y = 8x^4 + 3x^3 + 7x^2 + 6x + 3
y = 8 * torch.pow(x, 4) + 3 * torch.pow(x, 3) + 7 * torch.pow(x, 2) + 6 * x + 3

# Retain gradients
y.retain_grad()

# Compute gradients
y.backward(torch.ones_like(x))  # Specify gradient for each element of y

# Print the gradients with respect to x
print(x.grad)
