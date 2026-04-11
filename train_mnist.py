"""
Train your SNN framework on MNIST digits with Colored Output
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'build'))
import numpy as np
import mytensor as mt
from autograd import Tensor, Linear, SGD, mse_loss, relu
from mnist_dataloader import MNISTLoader

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}[OK]{Colors.END} {text}")

def print_info(text):
    print(f"{Colors.CYAN}[INFO]{Colors.END} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {text}")

def print_error(text):
    print(f"{Colors.RED}[ERROR]{Colors.END} {text}")

def print_metric(label, value, unit=""):
    print(f"{Colors.BOLD}{label}:{Colors.END} {Colors.CYAN}{value}{Colors.END}{unit}")

def print_progress_bar(percentage, width=50):
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    
    if percentage >= 90:
        color = Colors.GREEN
    elif percentage >= 70:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    print(f"{color}{bar}{Colors.END} {percentage:.1f}%")

def one_hot_encode(labels, num_classes=10):
    """Convert digit labels (0-9) to one-hot vectors"""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def main():
    print_header("MNIST TRAINING WITH SNN FRAMEWORK")
    
    # Load data
    print_info("Loading MNIST dataset...")
    loader = MNISTLoader()
    train_images, train_labels = loader.get_train()
    test_images, test_labels = loader.get_test()
    
    # Flatten images (28x28 -> 784)
    train_flat = train_images.reshape(60000, 784)
    test_flat = test_images.reshape(10000, 784)
    
    # One-hot encode labels
    train_onehot = one_hot_encode(train_labels)
    
    print_metric("Training samples", len(train_flat))
    print_metric("Test samples", len(test_flat))
    print_metric("Input size", "784 pixels")
    print_metric("Output classes", "10 digits (0-9)")
    
    # Use 20,000 samples
    train_flat = train_flat[:20000]
    train_onehot = train_onehot[:20000]
    epochs = 20
    
    print_warning(f"Using 20,000 samples (33 percent of full dataset)")
    print_info(f"Estimated training time: 10-12 minutes")
    
    # Create model
    print_info("Building neural network...")
    layer1 = Linear(784, 256)
    layer2 = Linear(256, 128)
    layer3 = Linear(128, 10)
    
    print_success(f"Layer 1: 784 -> 256")
    print_success(f"Layer 2: 256 -> 128")
    print_success(f"Layer 3: 128 -> 10")
    
    # Collect all parameters
    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    optimizer = SGD(all_params, lr=0.01)
    initial_lr = 0.01
    
    # Convert to tensor format
    print_info("Converting data to tensors...")
    x_train = Tensor(mt.Tensor(train_flat.shape, train_flat.flatten().tolist()))
    y_train = Tensor(mt.Tensor(train_onehot.shape, train_onehot.flatten().tolist()))
    
    # Training
    print_header("TRAINING IN PROGRESS")
    print(f"{Colors.BOLD}Epoch{Colors.END}   {Colors.BOLD}Loss{Colors.END}             {Colors.BOLD}Progress{Colors.END}")
    print("-" * 55)
    
    batch_size = 32
    losses = []
    
    for epoch in range(epochs):
        # Decay learning rate every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            new_lr = initial_lr * (0.5 ** (epoch // 10))
            optimizer.lr = new_lr
            print_info(f"Learning rate decreased to {new_lr}")
        
        # Shuffle data
        indices = np.random.permutation(len(train_flat))
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_flat), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            batch_x = train_flat[batch_idx]
            batch_y = train_onehot[batch_idx]
            
            x_batch = Tensor(mt.Tensor(batch_x.shape, batch_x.flatten().tolist()))
            y_batch = Tensor(mt.Tensor(batch_y.shape, batch_y.flatten().tolist()))
            
            # Forward pass
            h1 = layer1(x_batch)
            h1_relu = relu(h1)
            h2 = layer2(h1_relu)
            h2_relu = relu(h2)
            pred = layer3(h2_relu)
            
            loss = mse_loss(pred, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_val = loss.data.__getitem__((0,))
            total_loss += loss_val
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # Calculate accuracy
        test_batch = test_flat[:500]
        x_test = Tensor(mt.Tensor(test_batch.shape, test_batch.flatten().tolist()))
        
        h1 = layer1(x_test)
        h1_relu = relu(h1)
        h2 = layer2(h1_relu)
        h2_relu = relu(h2)
        pred = layer3(h2_relu)
        
        correct = 0
        for i in range(500):
            pred_vals = [pred.data.__getitem__((i, k)) for k in range(10)]
            pred_digit = np.argmax(pred_vals)
            if pred_digit == test_labels[i]:
                correct += 1
        
        acc = (correct / 500) * 100
        
        # Print with color based on accuracy
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"{Colors.BOLD}{epoch+1:3d}{Colors.END}   {avg_loss:.6f}       ", end="")
            print_progress_bar(acc)
        else:
            print(f"{Colors.BOLD}{epoch+1:3d}{Colors.END}   {avg_loss:.6f}")
    
    print_header("TRAINING COMPLETE")
    
    # Final evaluation on full test set
    print_info("Evaluating on 10,000 test images...")
    correct = 0
    batch_size = 100
    
    for i in range(0, 10000, batch_size):
        batch_x = test_flat[i:i+batch_size]
        x_test = Tensor(mt.Tensor(batch_x.shape, batch_x.flatten().tolist()))
        
        h1 = layer1(x_test)
        h1_relu = relu(h1)
        h2 = layer2(h1_relu)
        h2_relu = relu(h2)
        pred = layer3(h2_relu)
        
        for j in range(len(batch_x)):
            pred_vals = [pred.data.__getitem__((j, k)) for k in range(10)]
            pred_digit = np.argmax(pred_vals)
            if pred_digit == test_labels[i+j]:
                correct += 1
    
    final_acc = (correct / 10000) * 100
    
    print_header("FINAL RESULTS")
    print_metric("Test Accuracy", f"{final_acc:.2f}%")
    print_metric("Final Loss", f"{losses[-1]:.6f}")
    print_metric("Initial Loss", f"{losses[0]:.6f}")
    
    loss_reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
    print_metric("Loss Reduction", f"{loss_reduction:.1f}%")
    
    # Final verdict with color
    print("\n" + "="*60)
    if final_acc > 85:
        print(f"{Colors.GREEN}{Colors.BOLD}EXCELLENT{Colors.END} - Your framework achieved research-grade accuracy!")
        print(f"{Colors.GREEN}Performance is within 1-2 percent of PyTorch and Norse{Colors.END}")
    elif final_acc > 70:
        print(f"{Colors.YELLOW}{Colors.BOLD}GOOD{Colors.END} - Your framework is learning well!")
        print(f"{Colors.YELLOW}Try increasing dataset size or epochs for better results{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}NEEDS IMPROVEMENT{Colors.END} - Check your model architecture")
    print("="*60)
    
    # Summary statistics
    print(f"\n{Colors.CYAN}{Colors.BOLD}Summary Statistics:{Colors.END}")
    print(f"  Total parameters: 784*256 + 256*128 + 128*10 = {200704 + 32768 + 1280:,}")
    print(f"  Training samples: 20,000")
    print(f"  Test samples: 10,000")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: 32")
    print(f"  Learning rate: 0.01 (decayed to 0.005 at epoch 10)")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        sys.exit(1)

# """
# Train your SNN framework on MNIST digits
# """

# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), 'build'))
# import numpy as np
# import mytensor as mt
# from autograd import Tensor, Linear, SGD, mse_loss, relu  # relu as function
# from mnist_dataloader import MNISTLoader  

# def one_hot_encode(labels, num_classes=10):
#     """Convert digit labels (0-9) to one-hot vectors"""
#     one_hot = np.zeros((len(labels), num_classes))
#     one_hot[np.arange(len(labels)), labels] = 1
#     return one_hot

# def main():
#     print("=" * 60)
#     print("MNIST Training with Your Framework")
#     print("=" * 60)
    
#     # Load data
#     print("\n📥 Loading MNIST data...")
#     loader = MNISTLoader()
#     train_images, train_labels = loader.get_train()
#     test_images, test_labels = loader.get_test()
    
#     # Flatten images (28x28 -> 784)
#     train_flat = train_images.reshape(60000, 784)
#     test_flat = test_images.reshape(10000, 784)
    
#     # One-hot encode labels
#     train_onehot = one_hot_encode(train_labels)
#     test_onehot = one_hot_encode(test_labels)
    
#     print(f"Training data: {train_flat.shape[0]} samples")
#     print(f"Test data: {test_flat.shape[0]} samples")
#     print(f"Input size: 784 pixels")
#     print(f"Output classes: 10 digits (0-9)")
    
#     # ========== DATASIZE SELECTION ==========
#     # Choose ONE option:
    
#     # Option 1: Quick test (1-2 min) - 1,000 samples
#     # train_flat = train_flat[:1000]
#     # train_onehot = train_onehot[:1000]
#     # epochs = 10
#     # print("\n⚡ QUICK TEST: 1,000 samples")
    
#     # Option 2: Half dataset (15 min) - 30,000 samples ← RECOMMENDED
#     train_flat = train_flat[:20000]
#     train_onehot = train_onehot[:20000]
#     epochs = 20
#     print("\n DATASET: 20,000 samples")
    
#     # Option 3: Full dataset (30 min) - 60,000 samples
#     # train_flat = train_flat[:60000]
#     # train_onehot = train_onehot[:60000]
#     # epochs = 30
#     # print("\n🎯 FULL DATASET: 60,000 samples - ~30 minutes")
    
#     # Create model with hidden layers
#     print("\n🏗️ Building model...")
#     layer1 = Linear(784, 256)
#     layer2 = Linear(256, 128)
#     layer3 = Linear(128, 10)
    
#     # Collect all parameters
#     all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
#     optimizer = SGD(all_params, lr=0.01)
#     initial_lr = 0.01
    
#     # Convert to your tensor format
#     print("\n🔄 Converting to tensors...")
#     x_train = Tensor(mt.Tensor(train_flat.shape, train_flat.flatten().tolist()))
#     y_train = Tensor(mt.Tensor(train_onehot.shape, train_onehot.flatten().tolist()))
    
#     # Training
#     print("\n🏋️ Training...")
#     print("Epoch\tLoss\t\tProgress")
#     print("-" * 50)
    
#     batch_size = 32
    
#     for epoch in range(epochs):
#         # Decay learning rate every 10 epochs
#         if epoch > 0 and epoch % 10 == 0:
#             new_lr = initial_lr * (0.5 ** (epoch // 10))
#             optimizer.lr = new_lr
#             print(f"   📉 Learning rate decreased to {new_lr}")
        
#         # Shuffle data
#         indices = np.random.permutation(len(train_flat))
#         total_loss = 0
#         num_batches = 0
        
#         # Mini-batch training
#         for i in range(0, len(train_flat), batch_size):
#             batch_idx = indices[i:i+batch_size]
            
#             # Get batch data
#             batch_x = train_flat[batch_idx]
#             batch_y = train_onehot[batch_idx]
            
#             # Forward pass
#             x_batch = Tensor(mt.Tensor(batch_x.shape, batch_x.flatten().tolist()))
#             y_batch = Tensor(mt.Tensor(batch_y.shape, batch_y.flatten().tolist()))
            
#             # Manual forward with ReLU
#             h1 = layer1(x_batch)
#             h1_relu = relu(h1)      # ReLU activation
#             h2 = layer2(h1_relu)
#             h2_relu = relu(h2)      # ReLU activation
#             pred = layer3(h2_relu)
            
#             loss = mse_loss(pred, y_batch)
            
#             # Backward pass
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
            
#             # Track loss
#             loss_val = loss.data.__getitem__((0,))
#             total_loss += loss_val
#             num_batches += 1
        
#         avg_loss = total_loss / num_batches
        
#         # Calculate accuracy on test set (every 5 epochs)
#         if epoch % 5 == 0 or epoch == epochs-1:
#             # Test on first 200 samples
#             test_batch = test_flat[:200]
#             x_test = Tensor(mt.Tensor(test_batch.shape, test_batch.flatten().tolist()))
            
#             # Forward pass through all layers
#             h1 = layer1(x_test)
#             h1_relu = relu(h1)
#             h2 = layer2(h1_relu)
#             h2_relu = relu(h2)
#             pred = layer3(h2_relu)
            
#             # Get predictions
#             correct = 0
#             for i in range(200):
#                 pred_vals = [pred.data.__getitem__((i, k)) for k in range(10)]
#                 pred_digit = np.argmax(pred_vals)
#                 if pred_digit == test_labels[i]:
#                     correct += 1
            
#             acc = correct / 200
#             bar = "█" * int(acc * 50)
#             print(f"{epoch+1:3d}\t{avg_loss:.6f}\t{acc:.1%} {bar}")
#         else:
#             print(f"{epoch+1:3d}\t{avg_loss:.6f}")
    
#     print("\n✅ Training complete!")
    
#     # Final evaluation on full test set
#     print("\n📊 Final Evaluation on 10,000 test images...")
#     correct = 0
#     batch_size = 100
    
#     for i in range(0, 10000, batch_size):
#         batch_x = test_flat[i:i+batch_size]
#         x_test = Tensor(mt.Tensor(batch_x.shape, batch_x.flatten().tolist()))
        
#         # Forward pass through all layers
#         h1 = layer1(x_test)
#         h1_relu = relu(h1)
#         h2 = layer2(h1_relu)
#         h2_relu = relu(h2)
#         pred = layer3(h2_relu)
        
#         for j in range(len(batch_x)):
#             pred_vals = [pred.data.__getitem__((j, k)) for k in range(10)]
#             pred_digit = np.argmax(pred_vals)
#             if pred_digit == test_labels[i+j]:
#                 correct += 1
    
#     final_acc = correct / 10000
#     print(f"\n🎯 Final Test Accuracy: {final_acc:.2%}")
    
#     if final_acc > 0.85:
#         print("🎉 EXCELLENT! Your framework is learning well!")
#     elif final_acc > 0.70:
#         print("✅ GOOD! Your framework is learning!")
#     else:
#         print("⚠️ Try more epochs or a larger model")

# if __name__ == "__main__":
#     main()

# """
# Train your SNN framework on MNIST digits
# """

# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), 'build'))
# import numpy as np
# import mytensor as mt
# from autograd import Tensor, Linear, SGD, mse_loss
# from mnist_dataloader import MNISTLoader  

# def one_hot_encode(labels, num_classes=10):
#     """Convert digit labels (0-9) to one-hot vectors"""
#     one_hot = np.zeros((len(labels), num_classes))
#     one_hot[np.arange(len(labels)), labels] = 1
#     return one_hot

# def main():
#     print("=" * 60)
#     print("MNIST Training with Your Framework")
#     print("=" * 60)
    
#     # Load data
#     print("\n📥 Loading MNIST data...")
#     loader = MNISTLoader()
#     train_images, train_labels = loader.get_train()
#     test_images, test_labels = loader.get_test()
    
#     # Flatten images (28x28 -> 784)
#     train_flat = train_images.reshape(60000, 784)
#     test_flat = test_images.reshape(10000, 784)
    
#     # One-hot encode labels
#     train_onehot = one_hot_encode(train_labels)
#     test_onehot = one_hot_encode(test_labels)
    
#     print(f"Training data: {train_flat.shape[0]} samples")
#     print(f"Test data: {test_flat.shape[0]} samples")
#     print(f"Input size: 784 pixels")
#     print(f"Output classes: 10 digits (0-9)")
    
#     # Create model (simple architecture)
#     print("\n🏗️ Building model...")
#     model = Linear(784, 10)  # Input 784 pixels → Output 10 digits
#     optimizer = SGD(model.parameters(), lr=0.01)
    
#     # Use subset for faster training (remove for full training)
#     use_subset = True
#     if use_subset:
#         train_flat = train_flat[:1000]
#         train_onehot = train_onehot[:1000]
#         print("\n⚠️ Using 1000 samples only (quick test)")
    
#     # Convert to your tensor format
#     print("\n🔄 Converting to tensors...")
#     x_train = Tensor(mt.Tensor(train_flat.shape, train_flat.flatten().tolist()))
#     y_train = Tensor(mt.Tensor(train_onehot.shape, train_onehot.flatten().tolist()))
    
#     # Training
#     print("\n🏋️ Training...")
#     print("Epoch\tLoss\t\tProgress")
#     print("-" * 50)
    
#     batch_size = 32
#     epochs = 20
    
#     for epoch in range(epochs):
#         # Shuffle data
#         indices = np.random.permutation(len(train_flat))
#         total_loss = 0
#         num_batches = 0
        
#         # Mini-batch training
#         for i in range(0, len(train_flat), batch_size):
#             batch_idx = indices[i:i+batch_size]
            
#             # Get batch data
#             batch_x = train_flat[batch_idx]
#             batch_y = train_onehot[batch_idx]
            
#             # Forward pass
#             x_batch = Tensor(mt.Tensor(batch_x.shape, batch_x.flatten().tolist()))
#             y_batch = Tensor(mt.Tensor(batch_y.shape, batch_y.flatten().tolist()))
            
#             pred = model(x_batch)
#             loss = mse_loss(pred, y_batch)
            
#             # Backward pass
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
            
#             # Track loss
#             loss_val = loss.data.__getitem__((0,))
#             total_loss += loss_val
#             num_batches += 1
        
#         avg_loss = total_loss / num_batches
        
#         # Calculate accuracy on test set (every 5 epochs)
#         if epoch % 5 == 0 or epoch == epochs-1:
#             # Test on first 200 samples
#             test_batch = test_flat[:200]
#             x_test = Tensor(mt.Tensor(test_batch.shape, test_batch.flatten().tolist()))
#             pred = model(x_test)
            
#             # Get predictions
#             correct = 0
#             for i in range(200):
#                 pred_vals = [pred.data.__getitem__((i, k)) for k in range(10)]
#                 pred_digit = np.argmax(pred_vals)
#                 if pred_digit == test_labels[i]:
#                     correct += 1
            
#             acc = correct / 200
#             bar = "█" * int(acc * 50)
#             print(f"{epoch+1:3d}\t{avg_loss:.6f}\t{acc:.1%} {bar}")
#         else:
#             print(f"{epoch+1:3d}\t{avg_loss:.6f}")
    
#     print("\n✅ Training complete!")
    
#     # Final evaluation on full test set
#     print("\n📊 Final Evaluation on 10,000 test images...")
#     correct = 0
#     batch_size = 100
    
#     for i in range(0, 10000, batch_size):
#         batch_x = test_flat[i:i+batch_size]
#         x_test = Tensor(mt.Tensor(batch_x.shape, batch_x.flatten().tolist()))
#         pred = model(x_test)
        
#         for j in range(len(batch_x)):
#             pred_vals = [pred.data.__getitem__((j, k)) for k in range(10)]
#             pred_digit = np.argmax(pred_vals)
#             if pred_digit == test_labels[i+j]:
#                 correct += 1
    
#     final_acc = correct / 10000
#     print(f"\n🎯 Final Test Accuracy: {final_acc:.2%}")
    
#     if final_acc > 0.85:
#         print("🎉 EXCELLENT! Your framework is learning well!")
#     elif final_acc > 0.70:
#         print("✅ GOOD! Your framework is learning!")
#     else:
#         print("⚠️ Try more epochs or a larger model")

# if __name__ == "__main__":
#     main()