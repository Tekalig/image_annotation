# Sample Datasets README

This folder contains sample datasets to demonstrate the enhanced functionality of the image annotation assessment scripts.

## Overview

Sample datasets have been created to show how the enhanced scripts work with custom data:

1. **Part 1**: CSV dataset with model predictions
2. **Part 2**: Image dataset with 4 classes of synthetic shapes

## Part 1: Model Predictions Dataset

### File: `sample_model_predictions.csv`

**Description**: Simulates real-world model predictions with realistic characteristics.

**Specifications**:
- **Samples**: 2,000 predictions
- **Classes**: Binary (0 and 1)
- **Distribution**: 59% negative (0), 41% positive (1) - realistic imbalance
- **Confidence Scores**: Range from 0.003 to 0.997
- **Hard Cases**: Includes 50 intentionally challenging predictions to test threshold optimization

**Columns**:
- `true_label`: Ground truth (0 or 1)
- `confidence_score`: Model's confidence in predicting class 1 (0.0 to 1.0)

**Example Usage**:
```bash
python part1_threshold_optimization_enhanced.py
# Automatically detects and uses sample_model_predictions.csv
```

**Key Features**:
- Realistic class imbalance
- Confidence scores correlate with true labels (as in real models)
- Includes edge cases and hard-to-classify samples
- Tests threshold optimization algorithms effectively

## Part 2: Synthetic Shapes Image Dataset

### Structure: `train/` and `test/` folders

**Description**: Synthetically generated images of geometric shapes for classification.

**Specifications**:

#### Classes (4):
1. **Circles**: Round shapes with varying sizes and colors
2. **Squares**: Rectangular shapes with varying sizes and colors
3. **Triangles**: Triangular shapes with varying sizes and colors
4. **Crosses**: Cross/plus shapes with varying sizes and colors

#### Training Set:
- **Total Images**: 200
- **Per Class**: 50 images
- **Location**: `datasets/part2/train/[class_name]/`

#### Test Set:
- **Total Images**: 80
- **Per Class**: 20 images
- **Location**: `datasets/part2/test/[class_name]/`

#### Image Properties:
- **Format**: JPEG (.jpg)
- **Size**: 64x64 pixels (automatically resized to 32x32 by the script)
- **Channels**: RGB (3 channels)
- **Background**: White with added noise for realism
- **Shapes**: Randomly sized and colored for variation

**Example Usage**:
```bash
python part2_cifar10_cnn_enhanced.py
# Automatically detects and uses datasets/part2/train/ and test/
```

**Key Features**:
- Simple but clear classification task
- Demonstrates multi-class classification (4 classes)
- Shows data augmentation effects during training
- Fast to train (small dataset size)
- Easy to visualize and understand results

## Creating Your Own Sample Datasets

### Automated Creation

Run the provided script to regenerate sample datasets:

```bash
python create_sample_datasets.py
```

This will:
1. Create `datasets/part1/sample_model_predictions.csv` (2000 samples)
2. Create `datasets/part2/train/` and `test/` with 280 images

### Manual Creation

#### For Part 1 (CSV):
```python
import pandas as pd
import numpy as np

# Create your predictions
df = pd.DataFrame({
    'true_label': [0, 1, 1, 0, 1],
    'confidence_score': [0.2, 0.8, 0.9, 0.3, 0.7]
})

df.to_csv('datasets/part1/my_predictions.csv', index=False)
```

#### For Part 2 (Images):
```bash
# Organize your images
mkdir -p datasets/part2/train/cat datasets/part2/train/dog
mkdir -p datasets/part2/test/cat datasets/part2/test/dog

# Copy your images
cp /path/to/cat/images/*.jpg datasets/part2/train/cat/
cp /path/to/dog/images/*.jpg datasets/part2/train/dog/
# ... and test images
```

## Running Demonstrations

### Quick Demo (Both Parts)

```bash
python demo_custom_datasets.py
```

This will:
1. Run Part 1 with the sample CSV dataset
2. Run Part 2 with the sample image dataset (2 epochs)
3. Generate all visualizations
4. Show comprehensive summary

### Individual Demos

**Part 1 Only**:
```bash
python part1_threshold_optimization_enhanced.py
```

**Part 2 Only**:
```bash
python part2_cifar10_cnn_enhanced.py
```

## Expected Results

### Part 1 (Threshold Optimization)
- **Optimal Threshold**: ~0.51
- **F1 Score**: ~0.93
- **Accuracy**: ~0.94
- **Generated Files**:
  - `threshold_analysis.png` (4 plots: PR-F1 curves, PR curve, ROC curve, multi-metric)
  - `threshold_results.csv` (metrics for 81 thresholds)

### Part 2 (CNN Training)
With 2 epochs (demo):
- **Training Accuracy**: 40-60%
- **Test Accuracy**: 35-50%
- Note: Simple shapes are easy to learn, so accuracy improves quickly

With 10+ epochs (full training):
- **Training Accuracy**: 80-95%
- **Test Accuracy**: 70-85%
- Per-class accuracy varies by shape complexity

**Generated Files**:
- `best_model.pth` (saved model checkpoint)
- `training_curves_custom.png` (loss and accuracy curves)

## Dataset Statistics

### Part 1 CSV Dataset
```
Total Samples: 2,000
Class 0: 1,183 (59.15%)
Class 1: 817 (40.85%)

Confidence Score Distribution:
  Mean: 0.448
  Std Dev: 0.298
  Min: 0.003
  Max: 0.997

Hard Cases: 50 (2.5% of dataset)
```

### Part 2 Image Dataset
```
Total Images: 280
Training: 200 (71.4%)
Testing: 80 (28.6%)

Per-Class Distribution:
  circles: 50 train, 20 test
  squares: 50 train, 20 test
  triangles: 50 train, 20 test
  crosses: 50 train, 20 test

Image Properties:
  Original Size: 64x64 pixels
  Processed Size: 32x32 pixels (auto-resized)
  Format: JPEG RGB
  File Size: ~2-4 KB per image
```

## Comparison: Sample vs Real Data

### When to Use Sample Data
✓ Testing the scripts  
✓ Learning how the tools work  
✓ Demonstrating functionality  
✓ Quick prototyping  
✓ Understanding output formats  

### When to Use Real Data
✓ Actual model evaluation  
✓ Production threshold optimization  
✓ Real-world model training  
✓ Performance benchmarking  
✓ Publication or presentation  

## Tips for Best Results

### Part 1 (Threshold Optimization)
1. **Dataset Size**: Use at least 1,000 samples for stable threshold estimation
2. **Class Balance**: Include both classes, but imbalance is OK (reflects reality)
3. **Hard Cases**: Include edge cases to test threshold robustness
4. **Confidence Range**: Ensure scores span 0.0 to 1.0 for full analysis

### Part 2 (Image Classification)
1. **Dataset Size**: Minimum 50 images per class for training
2. **Class Balance**: Try to keep classes roughly balanced
3. **Image Quality**: Clear, well-lit images work best
4. **Variety**: Include different angles, sizes, and contexts
5. **Train/Test Split**: Use 70-80% for training, 20-30% for testing

## Troubleshooting

### Part 1 Issues

**Error: "No CSV files found"**
- Solution: Place CSV file in `datasets/part1/` folder
- Ensure file has `.csv` extension

**Error: "Missing required columns"**
- Solution: CSV must have `true_label` and `confidence_score` columns
- Check column names (case-sensitive)

### Part 2 Issues

**Error: "No class folders found"**
- Solution: Create class subfolders in `train/` and `test/`
- Example: `datasets/part2/train/class_name/`

**Error: "No images found"**
- Solution: Add images to class folders
- Supported formats: JPG, JPEG, PNG, BMP, GIF

**Low Accuracy**
- Solution: Train for more epochs (10-20)
- Add more images per class (100+)
- Check if images are clear and distinct

## Next Steps

1. **Try the sample datasets**: Run `python demo_custom_datasets.py`
2. **Review the outputs**: Check generated visualizations
3. **Replace with your data**: Use your own CSV or images
4. **Compare results**: See how your data performs
5. **Iterate and improve**: Use suggestions from analysis

## Related Files

- `create_sample_datasets.py` - Script to generate sample datasets
- `demo_custom_datasets.py` - Comprehensive demonstration script
- `part1_threshold_optimization_enhanced.py` - Part 1 enhanced script
- `part2_cifar10_cnn_enhanced.py` - Part 2 enhanced script
- `../README_ENHANCED.md` - Complete documentation
- `../ENHANCEMENTS.md` - Technical details of improvements

## Support

For questions or issues:
1. Check `datasets/README.md` (this file)
2. Review `../README_ENHANCED.md` for usage instructions
3. See `../ENHANCEMENTS.md` for technical details
4. Check error messages - they provide helpful guidance
