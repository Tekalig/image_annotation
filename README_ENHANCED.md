# Image Annotation - Neural Network Assessment

This repository contains solutions for a neural network and model training assessment, demonstrating practical understanding of CNNs, PyTorch, model evaluation, and optimization.

## 🆕 Enhanced Version with Custom Dataset Support

The repository now includes **enhanced versions** that support custom datasets:
- `part1_threshold_optimization_enhanced.py` - Works with your own CSV model predictions
- `part2_cifar10_cnn_enhanced.py` - Works with your own image datasets

See [datasets/README.md](datasets/README.md) for dataset format and structure.

## Overview

The project consists of two main parts:

1. **Part 1**: Model Evaluation & Threshold Optimization (15 minutes)
2. **Part 2**: Simple CNN with Training Process (17 minutes)

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

### Running the Scripts

**Original version (with sample data):**
```bash
python part1_threshold_optimization.py
```

**Enhanced version (supports custom datasets):**
```bash
python part1_threshold_optimization_enhanced.py
```

The enhanced version will:
1. Look for CSV files in `datasets/part1/` folder
2. If found, use your custom dataset
3. If not found, generate sample data automatically

### Custom Dataset Format

Place your CSV file in `datasets/part1/` with columns:
- `true_label`: Ground truth (0 or 1)
- `confidence_score`: Model confidence (0.0 to 1.0)
- `predicted_label`: (optional, will be generated if missing)

Example:
```csv
true_label,confidence_score,predicted_label
0,0.23,0
1,0.87,1
```

See [datasets/README.md](datasets/README.md) for more details.

### What It Does

1. **Loads or Generates Dataset**: Uses custom data if available, generates sample otherwise
2. **Default Threshold Analysis**: Evaluates performance at threshold = 0.5
3. **Optimal Threshold Search**: Tests multiple thresholds (0.1 to 0.9) and finds optimal based on F1 score
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
6. **Improvement Suggestions**: Actionable recommendations for:
   - Threshold adjustment strategies
   - Confidence score calibration
   - Ensemble methods
   - Uncertainty-based filtering
   - Error analysis

### Output Files

- `model_outputs.csv`: Dataset used for analysis
- `threshold_results.csv`: Complete results for all tested thresholds
- `threshold_analysis.png`: Comprehensive visualization

## Part 2: Simple CNN Training

### Description

This part demonstrates building, training, and evaluating a simple Convolutional Neural Network. It includes:
- Simple CNN architecture with 2 convolutional layers
- Complete training pipeline with loss tracking
- Loss and accuracy curve visualization
- Model checkpointing and reloading
- Inference on sample images
- Strategies to improve accuracy from 90% to 99%

### Running the Scripts

**Original version (CIFAR-10 only):**
```bash
python part2_cifar10_cnn.py
```

**Enhanced version (supports custom datasets):**
```bash
python part2_cifar10_cnn_enhanced.py
```

**Architecture test (no internet required):**
```bash
python test_part2_architecture.py
```

The enhanced version will:
1. Look for image folders in `datasets/part2/train/` and `datasets/part2/test/`
2. If found, use your custom image dataset
3. If not found, download and use CIFAR-10 (requires internet)

### Custom Dataset Format

Organize your images as follows:
```
datasets/part2/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       └── image1.jpg
└── test/
    ├── class1/
    │   └── image1.jpg
    └── class2/
        └── image1.jpg
```

Supported formats: JPG, JPEG, PNG, BMP, GIF

See [datasets/README.md](datasets/README.md) for more details.

### What It Does

1. **Loads Dataset**: Custom images or CIFAR-10
   - Applies data augmentation (random flip, crop) for training
   - Resizes images to 32x32 automatically

2. **Defines Simple CNN Architecture**:
   ```
   - Conv1: 3 -> 32 channels, 3x3 kernel, BatchNorm, ReLU, MaxPool
   - Conv2: 32 -> 64 channels, 3x3 kernel, BatchNorm, ReLU, MaxPool
   - FC1: 64*8*8 -> 128, ReLU, Dropout(0.5)
   - FC2: 128 -> num_classes (output)
   ```
   - Total parameters: ~270K-545K (depending on number of classes)

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

5. **Visualizes Training**: Creates `training_curves.png` with loss and accuracy curves

6. **Model Evaluation**:
   - Overall accuracy on test set
   - Per-class accuracy breakdown
   - Sample predictions with confidence scores

7. **Model Reloading**: Demonstrates loading saved checkpoint and running inference

8. **Improvement Strategies**: Documents approaches for 90% → 99% accuracy

### Output Files

- `best_model.pth`: Saved model checkpoint
- `training_curves.png`: Training progress visualization
- `./data/`: CIFAR-10 dataset (if used)

### Expected Results

With the simple architecture (10 epochs):
- CIFAR-10: Training ~70-75%, Test ~65-70%
- Custom datasets: Varies by dataset complexity

## Project Structure

```
image_annotation/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── part1_threshold_optimization.py              # Part 1: Original version
├── part1_threshold_optimization_enhanced.py     # Part 1: Enhanced with custom data
├── part2_cifar10_cnn.py                        # Part 2: Original version
├── part2_cifar10_cnn_enhanced.py               # Part 2: Enhanced with custom data
├── test_part2_architecture.py                   # Architecture test (no internet)
├── ASSESSMENT_SUMMARY.md                        # Detailed assessment summary
├── datasets/                                    # Dataset folder
│   ├── README.md                               # Dataset format guide
│   ├── part1/                                  # CSV datasets for Part 1
│   └── part2/                                  # Image datasets for Part 2
│       ├── train/                              # Training images
│       └── test/                               # Test images
├── threshold_analysis.png                       # Generated: Visualization
├── training_curves.png                          # Generated: Training curves
└── best_model.pth                              # Generated: Saved model
```

## Quick Start

### For Quick Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test Part 1 (uses generated data)
python part1_threshold_optimization_enhanced.py

# Test Part 2 architecture (no internet required)
python test_part2_architecture.py
```

### With Custom Datasets

```bash
# 1. Prepare your datasets (see datasets/README.md)

# 2. For Part 1: Place CSV in datasets/part1/
# Example: datasets/part1/my_predictions.csv

# 3. For Part 2: Organize images in datasets/part2/
# Example: datasets/part2/train/cat/*.jpg, datasets/part2/train/dog/*.jpg

# 4. Run enhanced scripts
python part1_threshold_optimization_enhanced.py
python part2_cifar10_cnn_enhanced.py
```

### With CIFAR-10 (Internet Required)

```bash
# Just run the enhanced Part 2 without custom dataset
python part2_cifar10_cnn_enhanced.py
# Will automatically download and use CIFAR-10
```

## Key Improvements in Enhanced Versions

### Part 1 Enhancements:
1. ✅ **Custom Dataset Loading**: Automatically detects and loads CSV files from `datasets/part1/`
2. ✅ **Flexible Format**: Handles missing `predicted_label` column
3. ✅ **Data Validation**: Checks for required columns and validates data quality
4. ✅ **Clear Instructions**: Provides guidance on using custom datasets
5. ✅ **Backwards Compatible**: Falls back to generating sample data if no custom dataset

### Part 2 Enhancements:
1. ✅ **Custom Image Dataset Support**: Loads images from folder structure
2. ✅ **Automatic Image Resizing**: Handles any image size, resizes to 32x32
3. ✅ **Multiple Format Support**: JPG, PNG, BMP, GIF
4. ✅ **Flexible Class Count**: Works with any number of classes
5. ✅ **Fallback to CIFAR-10**: Uses CIFAR-10 if no custom dataset found
6. ✅ **Class Detection**: Automatically detects classes from folder names

## Skills Demonstrated

1. **Understanding of Neural Networks**: Clear implementation of CNN architecture with proper layer design
2. **PyTorch Proficiency**: Complete training pipeline from data loading to model evaluation
3. **Data Analysis**: Threshold optimization, precision-recall analysis, and visualization
4. **Model Evaluation**: Comprehensive metrics and performance analysis
5. **Problem-Solving**: Practical improvement strategies for accuracy optimization
6. **Code Flexibility**: Support for custom datasets and various data formats
7. **Communication**: Clear code documentation and structured output

## Assessment Summary

See [ASSESSMENT_SUMMARY.md](ASSESSMENT_SUMMARY.md) for:
- Detailed decisions and rationale
- Challenges encountered and solutions
- Next steps for improvement
- Time estimates and deliverables

## Notes for Video Recording

When demonstrating this work, you can:

1. **Show Custom Dataset Support**:
   - Demonstrate loading your own CSV predictions
   - Show loading custom image datasets
   - Explain the flexible structure

2. **Part 1 (15 min)**:
   - Run enhanced script with custom or generated data
   - Explain threshold optimization process
   - Show precision-recall tradeoffs
   - Discuss improvement strategies

3. **Part 2 (17 min)**:
   - Show CNN architecture
   - Run training (custom data or CIFAR-10)
   - Explain training curves
   - Demonstrate model reloading
   - Discuss improvement strategies for 99% accuracy

## License

MIT License - See LICENSE file for details
