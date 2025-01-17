import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define custom dataset
class MyDataset(Dataset):
    def __init__(self, X1, X2, Y):
        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.Y[idx]


# Dataset
x1 = torch.tensor([3, 4, 5, 6, 2], dtype=torch.float32)
x2 = torch.tensor([8, 5, 7, 3, 1], dtype=torch.float32)
y = torch.tensor([-3.5, 3.5, 2.5, 11.5, 5.7], dtype=torch.float32)
dataset = MyDataset(x1, x2, y)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Print batch data for verification
for data in iter(data_loader):
    print(data)


# Regression Model
class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.rand([1], requires_grad=True))
        self.w2 = torch.nn.Parameter(torch.rand([1], requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand([1], requires_grad=True))

    def forward(self, x1, x2):
        return self.w1 * x1 + self.w2 * x2 + self.b


# Loss function and optimizer
loss_fn = torch.nn.MSELoss()
model = RegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
loss_list = []  # To store loss values for graph
for epoch in range(100):
    epoch_loss = 0.0
    for i, data in enumerate(data_loader):
        x1, x2, labels = data
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    loss_list.append(avg_loss)  # Append average loss for the epoch

    if epoch % 33 == 0:
        print(
            f"After {epoch} epochs, The parameters are w1={model.w1.item()}, w2={model.w2.item()}, b={model.b.item()}, and loss={avg_loss:.4f}"
        )

# Final parameters and loss
print(
    f"Final parameters: w1={model.w1.item()}, w2={model.w2.item()}, b={model.b.item()}, final loss={avg_loss:.4f}"
)

# Plot the graph
plt.plot(range(100), loss_list, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs for Multi-Feature Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
