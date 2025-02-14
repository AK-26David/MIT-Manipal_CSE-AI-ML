import torch
import torch.nn as nn
import torch.optim as optim


class XORNeuralNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=4):
        super(XORNeuralNetwork, self).__init__()

        # Define network layers with ReLU
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output layer remains sigmoid for binary classification
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# Prepare training data
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# Initialize model, loss, and optimizer
model = XORNeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict and verify results
print("XOR Predictions:")
with torch.no_grad():
    for i in range(len(X)):
        prediction = model(X[i].unsqueeze(0))
        print(f"{X[i].numpy()} → {round(prediction.item())}")

# Count learnable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal Learnable Parameters: {total_params}")