"""
Binary Classification with PyTorch + MLflow Tracking
COMP 395 – Deep Learning

Same model as before, but with experiment tracking.
Run `mlflow ui` to view results at http://localhost:5000
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# =============================================================================
# Model
# =============================================================================

class BinaryClassifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# =============================================================================
# Data
# =============================================================================

def load_data():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    mean, std = X_train.mean(dim=0), X_train.std(dim=0)
    X_train, X_test = (X_train - mean) / std, (X_test - mean) / std
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Training with MLflow
# =============================================================================

def train(lr=0.01, n_epochs=100):
    X_train, X_test, y_train, y_test = load_data()
    model = BinaryClassifier(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("optimizer", "SGD")
        mlflow.log_param("loss", "MSE")
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for i in range(len(y_train)):
                optimizer.zero_grad()
                y_hat = model(X_train[i:i+1])
                loss = criterion(y_hat, y_train[i:i+1])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(y_train)
            
            # Log metrics every epoch
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_acc = ((model(X_train) >= 0.5) == y_train).float().mean().item()
            test_acc = ((model(X_test) >= 0.5) == y_test).float().mean().item()
        
        # Log final metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\nTrain accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"\nRun `mlflow ui` to view results")


if __name__ == "__main__":
    mlflow.set_experiment("binary-classification")
    train(lr=0.01, n_epochs=100)
