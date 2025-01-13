import torch
import torch.nn as nn

# Define ReLU activation
relu = nn.ReLU()

# Define tensors with gradients
b = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0], requires_grad=True)
w = torch.tensor([1.5], requires_grad=True)

# Perform forward pass
u = w * x
u.retain_grad()  # Retain gradient for u
v = u + b
v.retain_grad()  # Retain gradient for v
a = relu(v)
a.retain_grad()  # Retain gradient for a

# Perform backward pass
a.backward()

# Print gradients of inputs
print("Gradient of a (undefined):", a.grad)  # Will always be None
print("Gradient of u:", u.grad.item())       # Gradient of u
print("Gradient of v:", v.grad.item())       # Gradient of v
print("Gradient of b:", b.grad.item())       # Gradient of b
print("Gradient of x:", x.grad.item())       # Gradient of x
print("Gradient of w:", w.grad.item())       # Gradient of w
