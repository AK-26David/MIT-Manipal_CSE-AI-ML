import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
dataset = MyDataset(x, y)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Hyperparameters
learning_rate = 0.003
epochs = 100

# Define Model
class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand([1], requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand([1], requires_grad=True))

    def forward(self, x):
        return self.w * x + self.b


# Loss and Optimizer
loss_fn = torch.nn.MSELoss()
model = RegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
loss_list = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(data_loader)
    loss_list.append(avg_loss)

# Print final parameters and loss
print("The parameters are w={}, b={}, and final loss={}".format(model.w.item(), model.b.item(), avg_loss))

# Plot loss over epochs
plt.plot(range(epochs), loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs for Linear Regression")
plt.grid(True)
plt.show()
