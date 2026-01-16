# Image Annotation - Neural Network Assessment

This repository contains solutions for a neural network and model training assessment, demonstrating practical understanding of CNNs, PyTorch, model evaluation, and optimization.

## Overview

The project consists of two main parts:

1. **Part 1**: Model Evaluation & Threshold Optimization (15 minutes)
2. **Part 2**: Simple CNN on CIFAR-10 with Training Process (17 minutes)

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 0.24.0
- torch >= 1.9.0
- torchvision >= 0.10.0
- seaborn >= 0.11.0

## Part 1: Model Evaluation & Threshold Optimization

### Description

This part demonstrates model evaluation and threshold optimization for binary classification. It includes:
- Analysis of model outputs (true labels, predicted labels, confidence scores)
- Finding the optimal confidence threshold
- Visualizing precision-recall tradeoffs
- Post-processing improvement suggestions

### Running the Script

```bash
python part1_threshold_optimization.py
```

### What It Does

1. **Generates Sample Dataset**: Creates a CSV file (`model_outputs.csv`) with simulated model predictions including:
   - True labels (ground truth)
   - Confidence scores (model output probabilities)
   - Predicted labels (using default 0.5 threshold)

2. **Default Threshold Analysis**: Evaluates performance at threshold = 0.5

3. **Optimal Threshold Search**: Tests multiple thresholds (0.1 to 0.9) and finds the optimal one based on F1 score

4. **Comprehensive Visualizations**: Creates `threshold_analysis.png` with:
   - Precision, Recall, and F1 Score vs Threshold
   - Precision-Recall Curve
   - ROC Curve with AUC
   - Multiple Metrics Comparison

5. **Performance Analysis**: Detailed metrics including:
   - Accuracy, Precision, Recall, Specificity
   - Confusion Matrix
   - F1 Score, Youden's J statistic
   - Balanced Accuracy

6. **Improvement Suggestions**: Provides actionable recommendations for:
   - Threshold adjustment strategies
   - Confidence score calibration
   - Ensemble methods
   - Uncertainty-based filtering
   - Error analysis
   - Model-specific improvements

### Output Files

- `model_outputs.csv`: Sample dataset with model predictions
- `threshold_results.csv`: Complete results for all tested thresholds
- `threshold_analysis.png`: Visualization of threshold analysis

### Key Findings

The script identifies the optimal threshold that maximizes F1 score and provides detailed analysis of:
- Trade-offs between precision and recall
- False positive vs false negative rates
- Strategies for different use cases (high-risk vs recall-critical)

## Part 2: Simple CNN on CIFAR-10

### Description

This part demonstrates building, training, and evaluating a simple Convolutional Neural Network on CIFAR-10 dataset. It includes:
- Simple CNN architecture with 2 convolutional layers
- Complete training pipeline with loss tracking
- Loss and accuracy curve visualization
- Model checkpointing and reloading
- Inference on sample images
- Strategies to improve accuracy from 90% to 99%

### Running the Script

```bash
python part2_cifar10_cnn.py
```

### What It Does

1. **Loads CIFAR-10 Dataset**: 
   - 50,000 training images
   - 10,000 test images
   - 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
   - Applies data augmentation (random flip, crop) for training

2. **Defines Simple CNN Architecture**:
   ```
   - Conv1: 3 -> 32 channels, 3x3 kernel, BatchNorm, ReLU, MaxPool
   - Conv2: 32 -> 64 channels, 3x3 kernel, BatchNorm, ReLU, MaxPool
   - FC1: 64*8*8 -> 128, ReLU, Dropout(0.5)
   - FC2: 128 -> 10 (output)
   ```
   - Total parameters: ~270,000 (trainable)

3. **Trains the Model**:
   - 10 epochs (configurable)
   - Adam optimizer with learning rate 0.001
   - Learning rate scheduling (step decay)
   - Cross-entropy loss
   - Batch size: 128
   - Progress updates every 100 batches

4. **Tracks Training Progress**:
   - Training loss and accuracy per epoch
   - Test loss and accuracy per epoch
   - Learning rate changes
   - Time per epoch
   - Saves best model based on test accuracy

5. **Visualizes Training**:
   - Creates `training_curves.png` with:
     - Training vs Test Loss curves
     - Training vs Test Accuracy curves
   - Clear visualization of model convergence

6. **Model Evaluation**:
   - Overall accuracy on test set
   - Per-class accuracy breakdown
   - Sample predictions with confidence scores

7. **Model Reloading for Inference**:
   - Demonstrates loading saved checkpoint
   - Runs inference on sample images
   - Shows predictions with confidence scores
   - Verifies model state restoration

8. **Improvement Strategies** (for 90% → 99% accuracy):
   - Architecture improvements (ResNet, DenseNet, more layers)
   - Data augmentation (Cutout, Mixup, AutoAugment)
   - Training techniques (more epochs, better scheduling)
   - Regularization (dropout, weight decay, label smoothing)
   - Optimizer tuning (SGD with momentum, AdamW, hyperparameter search)
   - Ensemble methods
   - Transfer learning
   - Advanced techniques (SE blocks, NAS, EMA)

### Output Files

- `best_model.pth`: Saved model checkpoint with best test accuracy
- `training_curves.png`: Visualization of training progress
- `./data/`: CIFAR-10 dataset (downloaded automatically)

### Expected Results

With the simple architecture (10 epochs):
- Training Accuracy: ~70-75%
- Test Accuracy: ~65-70%

This demonstrates the complete training process. To achieve higher accuracy (90%+), implement the suggested improvements.

## Project Structure

```
image_annotation/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── part1_threshold_optimization.py     # Part 1: Threshold optimization
├── part2_cifar10_cnn.py               # Part 2: CNN training on CIFAR-10
├── model_outputs.csv                  # Generated: Sample model predictions
├── threshold_results.csv              # Generated: Threshold analysis results
├── threshold_analysis.png             # Generated: Visualization
├── best_model.pth                     # Generated: Saved model
├── training_curves.png                # Generated: Training visualization
└── data/                              # Generated: CIFAR-10 dataset
```

## Running Both Parts

To run the complete assessment:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Part 1: Threshold Optimization
python part1_threshold_optimization.py

# Run Part 2: CIFAR-10 CNN Training
python part2_cifar10_cnn.py
```

## Key Concepts Demonstrated

### Neural Networks & CNNs
- Convolutional layer design
- Batch normalization
- Activation functions (ReLU)
- Pooling layers
- Fully connected layers
- Dropout for regularization

### Model Training
- Data loading and preprocessing
- Forward and backward propagation
- Loss computation (Cross-entropy)
- Optimization (Adam, SGD concepts)
- Learning rate scheduling
- Checkpoint saving and loading

### Data Preparation
- Data augmentation (random flip, crop)
- Normalization
- Train/test splitting
- Batch processing

### Model Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC Curve and AUC
- Precision-Recall Curve
- Per-class accuracy analysis
- Threshold optimization

### Optimization Techniques
- Hyperparameter tuning
- Regularization strategies
- Ensemble methods
- Transfer learning approaches
- Architecture improvements

## Skills Demonstrated

1. **Understanding of Neural Networks**: Clear implementation of CNN architecture with proper layer design
2. **PyTorch Proficiency**: Complete training pipeline from data loading to model evaluation
3. **Data Analysis**: Threshold optimization, precision-recall analysis, and visualization
4. **Model Evaluation**: Comprehensive metrics and performance analysis
5. **Problem-Solving**: Practical improvement strategies for accuracy optimization
6. **Communication**: Clear code documentation and structured output

## Assessment Summary

### Decisions Made

1. **Architecture Choice**: Simple 2-layer CNN to clearly demonstrate training process without excessive complexity
2. **Hyperparameters**: Conservative choices (10 epochs, lr=0.001) for quick demonstration while showing convergence
3. **Data Augmentation**: Standard augmentations (flip, crop) to improve generalization
4. **Threshold Metric**: Optimized F1 score as it balances precision and recall
5. **Visualization**: Comprehensive plots to show all relevant metrics and tradeoffs

### Challenges Addressed

1. **Balancing Simplicity and Completeness**: Created simple but complete implementations
2. **Clear Documentation**: Added extensive comments and print statements for understanding
3. **Practical Focus**: Emphasized real-world applicability over theoretical complexity
4. **Reproducibility**: Set random seeds and provided clear instructions

### Next Steps

1. **Part 1 Extensions**:
   - Test with real model outputs from production systems
   - Implement class-specific thresholds for multi-class problems
   - Add cost-sensitive threshold selection for business metrics
   - Integrate with live model monitoring systems

2. **Part 2 Extensions**:
   - Implement suggested improvements (ResNet architecture, better augmentation)
   - Train for 100+ epochs with proper scheduling
   - Add learning rate warmup and cosine annealing
   - Implement test-time augmentation
   - Create model ensemble
   - Achieve 90%+ accuracy target
   - Deploy for production inference

3. **General Improvements**:
   - Add unit tests for key functions
   - Create interactive dashboard for threshold selection
   - Implement automated hyperparameter tuning
   - Add model interpretability (Grad-CAM, attention maps)
   - Create API endpoints for model serving

## Notes for Video Recording

When demonstrating this work, cover:

1. **Part 1 (15 min)**:
   - Run the script and explain each step
   - Show the generated visualizations
   - Explain precision-recall tradeoffs
   - Discuss the optimal threshold selection
   - Walk through improvement suggestions
   - Show how to use this in production

2. **Part 2 (17 min)**:
   - Explain the CNN architecture (show code)
   - Run training and explain the output
   - Show loss curves and discuss convergence
   - Demonstrate model reloading
   - Show sample predictions
   - Discuss improvement strategies (2 min)
   - Explain which techniques would be most effective

## Author

This assessment demonstrates practical understanding of neural networks, CNNs, and the complete machine learning pipeline from data preparation to model deployment.

## License

MIT License - See LICENSE file for details