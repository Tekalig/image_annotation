# Datasets Folder

This folder contains datasets for the two parts of the assessment.

## Part 1: Model Evaluation & Threshold Optimization

Place your CSV file with model predictions in `datasets/part1/`:

### Expected CSV Format:
- **Required columns:**
  - `true_label`: Ground truth labels (0 or 1 for binary classification)
  - `confidence_score`: Model's confidence scores (0.0 to 1.0)
  
- **Optional columns:**
  - `predicted_label`: Predicted labels (will be generated if missing)

### Example CSV:
```csv
true_label,confidence_score,predicted_label
0,0.23,0
1,0.87,1
0,0.45,0
1,0.92,1
```

### Usage:
```bash
python part1_threshold_optimization_enhanced.py
```

The script will automatically detect and use your custom dataset.

---

## Part 2: CNN Training on Image Data

Place your image dataset in `datasets/part2/` with the following structure:

### Expected Folder Structure:
```
datasets/part2/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    ├── class2/
    │   └── ...
    └── ...
```

### Supported Image Formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)

### Image Requirements:
- Images will be automatically resized to 32x32 pixels
- RGB images (3 channels)
- Any original size supported

### Usage:
```bash
python part2_cifar10_cnn_enhanced.py
```

The script will automatically detect and use your custom dataset. If no custom dataset is found, it will fallback to CIFAR-10.

---

## Example Datasets

If you don't have custom datasets, the scripts will:
- **Part 1**: Generate a sample dataset with 1000 predictions
- **Part 2**: Use CIFAR-10 dataset (downloaded automatically)

---

## Notes

1. **Data Quality**: Ensure your data is properly labeled and cleaned
2. **Class Balance**: Check for class imbalance in your datasets
3. **File Naming**: File names don't matter, only folder structure
4. **Large Datasets**: For large datasets, consider using data generators to avoid memory issues

---

## Dataset Examples

### Part 1 Example Script:
```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'true_label': np.random.randint(0, 2, 1000),
    'confidence_score': np.random.random(1000)
})

# Save to datasets folder
df.to_csv('datasets/part1/my_predictions.csv', index=False)
```

### Part 2 Example (using existing images):
```bash
# Create structure
mkdir -p datasets/part2/train/cat datasets/part2/train/dog
mkdir -p datasets/part2/test/cat datasets/part2/test/dog

# Copy your images
cp /path/to/cat/images/* datasets/part2/train/cat/
cp /path/to/dog/images/* datasets/part2/train/dog/
# ... and test images
```
