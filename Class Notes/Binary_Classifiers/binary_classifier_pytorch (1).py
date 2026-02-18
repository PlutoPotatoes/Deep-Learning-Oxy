"""
Binary Classification with PyTorch nn.Module
COMP 395 – Deep Learning

Complete implementation using PyTorch's nn.Module and optimizer.
This is the idiomatic way to write neural networks in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# =============================================================================
# Model Definition
# =============================================================================

class BinaryClassifier(nn.Module):
    """
    Binary classifier: z = w·x + b, y_hat = sigmoid(z)
    """
    
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        
        # Initialize to zeros (to match our from-scratch version)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and preprocess the breast cancer dataset."""
    data = load_breast_cancer()
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test, y_train, y_test, data.feature_names


# =============================================================================
# Training
# =============================================================================

def train(model, X_train, y_train, lr=0.01, n_epochs=100):
    """Train the model using SGD."""
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        for i in range(len(y_train)):
            x_i = X_train[i:i+1]
            y_i = y_train[i:i+1]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_hat = model(x_i)
            
            # Compute loss
            loss = criterion(y_hat, y_i)
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
        
        losses.append(epoch_loss / len(y_train))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}, Loss: {losses[-1]:.4f}')
    
    return losses


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, X, y):
    """Compute accuracy."""
    model.eval()
    with torch.no_grad():
        y_hat = model(X)
        predictions = (y_hat >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    model.train()
    return accuracy


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Create model
    model = BinaryClassifier(n_features=X_train.shape[1])
    
    # Train
    print("\nTraining...")
    losses = train(model, X_train, y_train, lr=0.01, n_epochs=100)
    
    # Evaluate
    train_acc = evaluate(model, X_train, y_train)
    test_acc = evaluate(model, X_test, y_test)
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        weights = model.linear.weight.squeeze().numpy()
    plt.bar(range(len(weights)), weights)
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.title('Learned Weights')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()
