import torch
import matplotlib.pyplot as plt

# Data
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2], dtype=torch.float32)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32)

# Initialize parameters
b = torch.rand([1], requires_grad=True, dtype=torch.float32)
w = torch.rand([1], requires_grad=True, dtype=torch.float32)
learning_rate = 0.001
loss_list = []

# Training loop
for epoch in range(100):
    loss = 0.0
    for j in range(len(x)):
        a = w * x[j]
        y_p = a + b
        loss += (y[j] - y_p) ** 2
    loss = loss / len(x)  # Mean Squared Error
    loss_list.append(loss.item())

    # Backward pass
    loss.backward()

    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero gradients
    w.grad.zero_()
    b.grad.zero_()

# Plot loss over epochs
plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs for Linear Regression")
plt.grid(True)
plt.show()
