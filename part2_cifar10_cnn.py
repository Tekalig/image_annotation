"""
Part 2: Simple CNN on CIFAR-10

This script demonstrates:
1. Building a simple CNN with 1-2 convolutional layers
2. Training the model and tracking loss
3. Visualizing training progress (loss curves)
4. Evaluating model performance
5. Saving and loading model checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture with 2 convolutional layers.
    
    Architecture:
    - Conv1: 3 -> 32 channels, 3x3 kernel, ReLU, MaxPool
    - Conv2: 32 -> 64 channels, 3x3 kernel, ReLU, MaxPool
    - Fully Connected: 64*8*8 -> 128 -> 10
    
    This is deliberately simple to show the training process clearly.
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_cifar10_data(batch_size=128):
    """
    Load and prepare CIFAR-10 dataset with appropriate transformations.
    
    Returns:
        trainloader, testloader, classes
    """
    print("Loading CIFAR-10 dataset...")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load training data
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Download and load test data
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"Training samples: {len(trainset)}")
    print(f"Testing samples: {len(testset)}")
    print(f"Classes: {classes}")
    
    return trainloader, testloader, classes


def train_model(model, trainloader, testloader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the CNN model and track metrics.
    
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    print(f"\nTraining on device: {device}")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rate': []
    }
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss = test_loss / len(testloader)
        test_acc = 100. * correct / total
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 70)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'best_model.pth')
            print(f"✓ Best model saved with accuracy: {best_acc:.2f}%")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*70)
    
    return model, history


def plot_training_history(history, save_path='training_curves.png'):
    """
    Visualize training history (loss and accuracy curves).
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()


def evaluate_model(model, testloader, device='cpu'):
    """
    Comprehensive evaluation of the model.
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    overall_acc = 100. * correct / total
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print("\nPer-Class Accuracy:")
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {classes[i]:10s}: {acc:.2f}%")
    
    return overall_acc


def load_and_test_model(model_path='best_model.pth', batch_size=128):
    """
    Load a saved model and test it on sample images.
    This demonstrates model reloading for inference.
    """
    print("\n" + "="*60)
    print("LOADING MODEL FOR INFERENCE")
    print("="*60)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=10)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        print(f"Saved model accuracy: {checkpoint['test_acc']:.2f}%")
    else:
        print(f"Error: Model file {model_path} not found!")
        return None
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    _, testloader, classes = load_cifar10_data(batch_size=batch_size)
    
    # Test on a few sample images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
    
    # Display results for first 10 images
    print("\nSample Predictions:")
    print("-" * 60)
    for i in range(min(10, images.size(0))):
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        confidence = probabilities[i][predicted[i]].item() * 100
        correct = "✓" if labels[i] == predicted[i] else "✗"
        print(f"{correct} Image {i+1}: True: {true_label:8s} | Pred: {pred_label:8s} | Confidence: {confidence:.2f}%")
    
    # Evaluate on full test set
    accuracy = evaluate_model(model, testloader, device)
    
    return model, accuracy


def suggest_improvements_for_99_percent():
    """
    Suggest strategies to improve model from 90% to 99% accuracy.
    """
    print("\n" + "="*70)
    print("STRATEGIES TO IMPROVE FROM 90% TO 99% ACCURACY")
    print("="*70)
    
    print("\n1. ARCHITECTURE IMPROVEMENTS:")
    print("   ✓ Add more convolutional layers (3-5 conv blocks)")
    print("   ✓ Use residual connections (ResNet architecture)")
    print("   ✓ Increase network depth and width")
    print("   ✓ Try different architectures: ResNet, DenseNet, EfficientNet")
    print("   ✓ Use attention mechanisms")
    
    print("\n2. DATA AUGMENTATION:")
    print("   ✓ More aggressive augmentation: rotation, scaling, translation")
    print("   ✓ Cutout or random erasing")
    print("   ✓ Mixup or CutMix augmentation")
    print("   ✓ AutoAugment or RandAugment policies")
    print("   ✓ Test-time augmentation (TTA)")
    
    print("\n3. TRAINING TECHNIQUES:")
    print("   ✓ Train for more epochs (50-200 epochs)")
    print("   ✓ Use learning rate scheduling (cosine annealing, warm restarts)")
    print("   ✓ Implement early stopping with patience")
    print("   ✓ Use gradient clipping")
    print("   ✓ Mixed precision training for faster training")
    
    print("\n4. REGULARIZATION:")
    print("   ✓ Increase dropout rate or add dropout to more layers")
    print("   ✓ Use weight decay (L2 regularization)")
    print("   ✓ Implement label smoothing")
    print("   ✓ Add more batch normalization layers")
    print("   ✓ Try dropout alternatives: DropBlock, Stochastic Depth")
    
    print("\n5. OPTIMIZER & HYPERPARAMETERS:")
    print("   ✓ Try different optimizers: SGD with momentum, AdamW")
    print("   ✓ Use learning rate warmup")
    print("   ✓ Tune learning rate (try 0.01, 0.001, 0.0001)")
    print("   ✓ Adjust batch size (try 64, 128, 256)")
    print("   ✓ Grid search or Bayesian optimization for hyperparameters")
    
    print("\n6. ENSEMBLE METHODS:")
    print("   ✓ Train multiple models with different initializations")
    print("   ✓ Use different architectures in ensemble")
    print("   ✓ Snapshot ensembling")
    print("   ✓ Use knowledge distillation from larger models")
    
    print("\n7. TRANSFER LEARNING:")
    print("   ✓ Use pre-trained models (ResNet, EfficientNet on ImageNet)")
    print("   ✓ Fine-tune on CIFAR-10")
    print("   ✓ Progressive unfreezing of layers")
    
    print("\n8. ADVANCED TECHNIQUES:")
    print("   ✓ Use Squeeze-and-Excitation (SE) blocks")
    print("   ✓ Implement Shake-Shake or ShakeDrop regularization")
    print("   ✓ Try neural architecture search (NAS)")
    print("   ✓ Use exponential moving average (EMA) of model weights")
    
    print("\n9. DATA QUALITY:")
    print("   ✓ Check for and fix mislabeled samples")
    print("   ✓ Analyze hard negatives and focus training on them")
    print("   ✓ Use semi-supervised learning with additional unlabeled data")
    
    print("\n10. PRACTICAL STEPS (Priority Order):")
    print("    1. Use a proven architecture like ResNet-18 or ResNet-34")
    print("    2. Train for 100-200 epochs with cosine annealing")
    print("    3. Apply strong data augmentation")
    print("    4. Use SGD with momentum (0.9) and learning rate 0.1")
    print("    5. Implement learning rate warmup and decay")
    print("    6. Add weight decay (1e-4)")
    print("    7. Use test-time augmentation")
    print("    8. Ensemble 3-5 models")
    
    print("\n" + "="*70)


def main():
    """
    Main execution function for Part 2: Simple CNN on CIFAR-10
    """
    print("="*70)
    print("PART 2: SIMPLE CNN ON CIFAR-10")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    
    # Step 1: Load data
    print("\nStep 1: Loading CIFAR-10 dataset...")
    trainloader, testloader, classes = load_cifar10_data(batch_size=batch_size)
    
    # Step 2: Initialize model
    print("\nStep 2: Initializing SimpleCNN model...")
    model = SimpleCNN(num_classes=10)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    model, history = train_model(
        model, 
        trainloader, 
        testloader, 
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Step 4: Plot training curves
    print("\nStep 4: Plotting training curves...")
    plot_training_history(history)
    
    # Step 5: Evaluate model
    print("\nStep 5: Evaluating model...")
    final_acc = evaluate_model(model, testloader, device)
    
    # Step 6: Demonstrate model reloading for inference
    print("\nStep 6: Demonstrating model reload for inference...")
    loaded_model, loaded_acc = load_and_test_model('best_model.pth', batch_size=batch_size)
    
    # Step 7: Provide improvement suggestions
    print("\nStep 7: Suggesting improvements for 99% accuracy...")
    suggest_improvements_for_99_percent()
    
    print("\n" + "="*70)
    print("PART 2 COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    print("\nGenerated files:")
    print("  - best_model.pth: Saved model checkpoint")
    print("  - training_curves.png: Loss and accuracy curves")
    print("  - ./data: CIFAR-10 dataset")


if __name__ == "__main__":
    main()
