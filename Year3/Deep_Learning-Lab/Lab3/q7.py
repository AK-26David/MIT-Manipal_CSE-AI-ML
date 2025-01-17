import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define dataset
class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Input data
x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
dataset = MyDataset(x, y)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Logistic Regression Model
class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand([1], requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand([1], requires_grad=True))

    def forward(self, x):
        return self.w * x + self.b


# Define loss function and optimizer
loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
model = RegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
loss_list = []  # To store loss values for graph

for epoch in range(100):
    epoch_loss = 0.0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(outputs)  # Apply sigmoid activation
        labels = labels.to(torch.float32)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    loss_list.append(avg_loss)  # Record loss for the epoch

    if epoch % 33 == 0:
        print(
            f"After {epoch} epochs, Parameters: w={model.w.item():.4f}, b={model.b.item():.4f}, Loss={avg_loss:.4f}"
        )

# Final parameters and loss
print(
    f"Final Parameters: w={model.w.item():.4f}, b={model.b.item():.4f}, Final Loss={avg_loss:.4f}"
)

# Plot the graph
plt.plot(range(100), loss_list, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs for Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
