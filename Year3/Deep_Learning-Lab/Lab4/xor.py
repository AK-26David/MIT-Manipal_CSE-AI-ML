import torch
import torch.nn as nn
import torch.optim as optim

class XORNeuralNetwork(nn.Module):
    def __init__(self):
        super(XORNeuralNetwork, self).__init__()
        
        # Input layer to hidden layer
        self.hidden_layer = nn.Linear(2, 2)
        
        # Hidden layer to output layer
        self.output_layer = nn.Linear(2, 1)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Hidden layer with sigmoid activation
        hidden = self.sigmoid(self.hidden_layer(x))
        
        # Output layer with sigmoid activation
        output = self.sigmoid(self.output_layer(hidden))
        
        return output

def count_parameters(model):
    """
    Count the number of learnable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_xor_network():
    # Create the model
    model = XORNeuralNetwork()
    
    # XOR input data
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    
    # XOR target outputs
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Training loop
    epochs = 10000
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        
        # Compute loss
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss occasionally
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    
    # Print model parameters and their count
    print("\nModel Architecture:")
    print(model)
    
    # Count and print learnable parameters
    param_count = count_parameters(model)
    print(f"\nTotal Learnable Parameters: {param_count}")
    
    # Test the trained model
    print("\nModel Predictions:")
    with torch.no_grad():
        predictions = model(X)
        for i, (inp, pred, actual) in enumerate(zip(X, predictions, y)):
            print(f"Input: {inp.numpy()}, Predicted: {pred.item():.4f}, Actual: {actual.item()}")

# Run the training and testing
if __name__ == "__main__":
    train_xor_network()