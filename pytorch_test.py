"""
PyTorch MNIST Training for Comparison with SPINE
Produces output format matching the example
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import sys
import os

# Add path for your MNIST loader
sys.path.append(os.path.dirname(__file__))
from mnist_dataloader import MNISTLoader

def one_hot_encode(labels, num_classes=10):
    """Convert digit labels to one-hot vectors"""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def train_pytorch():
    """Train PyTorch model on MNIST"""
    
    print("=" * 60)
    print("PYTORCH MNIST TRAINING (CrossEntropyLoss)")
    print("=" * 60)
    
    # Load data
    print("\n[LOAD] Loading MNIST data...")
    loader = MNISTLoader()
    train_images, train_labels = loader.get_train()
    test_images, test_labels = loader.get_test()
    
    # Flatten images
    train_flat = train_images.reshape(60000, 784)
    test_flat = test_images.reshape(10000, 784)
    
    # Use 20,000 samples to match SPINE
    num_samples = 60000
    train_flat = train_flat[:num_samples]
    train_labels = train_labels[:num_samples]
    
    print(f"[DATA] Training samples: {len(train_flat)}")
    print(f"[DATA] Test samples: {len(test_flat)}")
    print(f"[DATA] Input size: 784 pixels")
    print(f"[DATA] Output classes: 10 digits")
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(train_flat, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_test = torch.tensor(test_flat, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    
    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Define model
    class MNISTModel(nn.Module):
        def __init__(self):
            super(MNISTModel, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = MNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training
    print("\n[TRAIN] Training for 20 epochs...")
    print("Epoch   Loss        Accuracy    Time")
    print("-" * 50)
    
    epochs = 20
    losses = []
    
    for epoch in range(epochs):
        # Learning rate decay
        if epoch > 0 and epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"   Learning rate decreased to {optimizer.param_groups[0]['lr']}")
        
        start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        epoch_time = time.time() - start_time
        
        # Calculate accuracy every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                test_outputs = model(x_test[:500])
                pred = torch.argmax(test_outputs, dim=1)
                correct = (pred == y_test[:500]).sum().item()
                acc = correct / 500 * 100
                print(f"{epoch+1:3d}     {avg_loss:.6f}   {acc:.1f}%        {epoch_time:.2f}s")
        else:
            print(f"{epoch+1:3d}     {avg_loss:.6f}                 {epoch_time:.2f}s")
    
    # Final evaluation on full test set
    print("\n[EVAL] Evaluating on 10,000 test images...")
    with torch.no_grad():
        test_outputs = model(x_test)
        pred = torch.argmax(test_outputs, dim=1)
        correct = (pred == y_test).sum().item()
        final_acc = correct / 10000 * 100
    
    print(f"\n[RESULT] PyTorch Final Test Accuracy: {final_acc:.2f}%")
    
    return final_acc, losses

def train_pytorch_mse():
    """PyTorch model with MSE loss (to match SPINE exactly)"""
    
    print("\n" + "=" * 60)
    print("PYTORCH MNIST TRAINING (MSE Loss - Same as SPINE)")
    print("=" * 60)
    
    # Load data
    print("\n[LOAD] Loading MNIST data...")
    loader = MNISTLoader()
    train_images, train_labels = loader.get_train()
    test_images, test_labels = loader.get_test()
    
    # Flatten images
    train_flat = train_images.reshape(60000, 784)
    test_flat = test_images.reshape(10000, 784)
    
    # One-hot encode labels
    train_onehot = one_hot_encode(train_labels)
    test_onehot = one_hot_encode(test_labels)
    
    # Use 20,000 samples
    num_samples = 20000
    train_flat = train_flat[:num_samples]
    train_onehot = train_onehot[:num_samples]
    
    print(f"[DATA] Training samples: {len(train_flat)}")
    print(f"[DATA] Test samples: {len(test_flat)}")
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(train_flat, dtype=torch.float32)
    y_train = torch.tensor(train_onehot, dtype=torch.float32)
    x_test = torch.tensor(test_flat, dtype=torch.float32)
    y_test = torch.tensor(test_onehot, dtype=torch.float32)
    
    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Define model
    class MNISTModel(nn.Module):
        def __init__(self):
            super(MNISTModel, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = MNISTModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training
    print("\n[TRAIN] Training for 20 epochs...")
    print("Epoch   Loss        Accuracy    Time")
    print("-" * 50)
    
    epochs = 20
    
    for epoch in range(epochs):
        # Learning rate decay
        if epoch > 0 and epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"   Learning rate decreased to {optimizer.param_groups[0]['lr']}")
        
        start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        # Calculate accuracy
        with torch.no_grad():
            test_outputs = model(x_test[:500])
            pred = torch.argmax(test_outputs, dim=1)
            correct = (pred == torch.argmax(y_test[:500], dim=1)).sum().item()
            acc = correct / 500 * 100
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"{epoch+1:3d}     {avg_loss:.6f}   {acc:.1f}%        {epoch_time:.2f}s")
        else:
            print(f"{epoch+1:3d}     {avg_loss:.6f}                 {epoch_time:.2f}s")
    
    # Final evaluation
    print("\n[EVAL] Evaluating on 10,000 test images...")
    with torch.no_grad():
        test_outputs = model(x_test)
        pred = torch.argmax(test_outputs, dim=1)
        correct = (pred == torch.argmax(y_test, dim=1)).sum().item()
        final_acc = correct / 10000 * 100
    
    print(f"\n[RESULT] PyTorch Final Test Accuracy: {final_acc:.2f}%")
    
    return final_acc

def comparison_summary(pytorch_acc, spine_acc=94.49):
    """Print comparison summary in exact format"""
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Framework':<15} {'Accuracy':<15} {'Loss Range':<20}")
    print("-" * 50)
    
    # SPINE results
    spine_loss_start = 0.056464
    spine_loss_end =  0.015714
    
    print(f"{'SPINE':<15} {spine_acc:.2f}%{'':<10} {spine_loss_end:.4f}")
    print(f"{'PyTorch':<15} {pytorch_acc:.2f}%{'':<10}")
    
    print("\n[DETAILS]")
    print(f"  SPINE:   20,000 samples, 20 epochs, {spine_acc:.2f}% accuracy")
    print(f"  PyTorch: 20,000 samples, 20 epochs, {pytorch_acc:.2f}% accuracy")
    
    diff = pytorch_acc - spine_acc
    if diff > 0:
        print(f"\n[RESULT] PyTorch is {diff:.2f}% more accurate than SPINE")
    elif diff < 0:
        print(f"\n[RESULT] SPINE is {abs(diff):.2f}% more accurate than PyTorch!")
    else:
        print(f"\n[RESULT] Both frameworks achieved the same accuracy!")

def main():
    """Run PyTorch training and compare with SPINE"""
    
    print("\n" + "=" * 60)
    print("PYTORCH MNIST BENCHMARK")
    print("=" * 60)
    
    # Check device
    if torch.cuda.is_available():
        print(f"\n[GPU] Using: {torch.cuda.get_device_name(0)}")
    else:
        print("\n[CPU] Using CPU for training")
    
    # Run training
    pytorch_acc, _ = train_pytorch()
    
    # Print comparison
    comparison_summary(pytorch_acc)
    
    print("\n[DONE] PyTorch benchmark complete!")

if __name__ == "__main__":
    main()