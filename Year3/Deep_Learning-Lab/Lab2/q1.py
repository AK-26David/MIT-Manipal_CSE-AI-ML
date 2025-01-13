import torch

# Define tensors with requires_grad=True
a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Compute intermediate values
x = 2 * a + 3 * b
x.retain_grad()  # Retain gradient for intermediate tensor x
y = 5 * a * a + 3 * b * b
y.retain_grad()  # Retain gradient for intermediate tensor y

# Compute final value
z = 2 * x + 3 * y
z.retain_grad()  # Retain gradient for intermediate tensor z

# Perform backward pass
z.backward(retain_graph=True)

# Print gradients
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)
print("Gradient of z:", z.grad)
print("Gradient of a:", a.grad.item())  # Corrected to use item() method
