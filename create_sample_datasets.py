"""
Create sample datasets for demonstration purposes
This script generates:
1. Sample CSV dataset for Part 1 (threshold optimization)
2. Sample image dataset for Part 2 (CNN training)
"""

import numpy as np
import pandas as pd
from PIL import Image
import os

def create_sample_csv_dataset():
    """Create sample CSV dataset for Part 1"""
    print("Creating sample CSV dataset for Part 1...")
    
    np.random.seed(42)
    
    # Simulate a real-world scenario with 2000 predictions
    # 60% negative, 40% positive (imbalanced)
    n_samples = 2000
    true_labels = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Generate confidence scores that correlate with true labels
    confidence_scores = np.zeros(n_samples)
    for i in range(n_samples):
        if true_labels[i] == 1:
            # Positives: higher confidence with realistic noise
            confidence_scores[i] = np.clip(np.random.beta(7, 2), 0, 1)
        else:
            # Negatives: lower confidence with realistic noise
            confidence_scores[i] = np.clip(np.random.beta(2, 7), 0, 1)
    
    # Add some hard cases (intentional misclassifications)
    n_hard = 50
    hard_indices = np.random.choice(n_samples, n_hard, replace=False)
    for idx in hard_indices:
        # Flip the confidence to create challenging cases
        if true_labels[idx] == 1:
            confidence_scores[idx] = np.random.uniform(0.2, 0.4)
        else:
            confidence_scores[idx] = np.random.uniform(0.6, 0.8)
    
    df = pd.DataFrame({
        'true_label': true_labels,
        'confidence_score': confidence_scores
    })
    
    os.makedirs('datasets/part1', exist_ok=True)
    output_path = 'datasets/part1/sample_model_predictions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Created {output_path}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Class distribution: {df['true_label'].value_counts().to_dict()}")
    print(f"  - Confidence score range: [{df['confidence_score'].min():.3f}, {df['confidence_score'].max():.3f}]")
    print()

def create_sample_image_dataset():
    """Create sample image dataset for Part 2"""
    print("Creating sample image dataset for Part 2...")
    
    np.random.seed(42)
    
    # Create 4 classes: circles, squares, triangles, crosses
    classes = ['circles', 'squares', 'triangles', 'crosses']
    n_train_per_class = 50
    n_test_per_class = 20
    img_size = 64
    
    for split in ['train', 'test']:
        n_per_class = n_train_per_class if split == 'train' else n_test_per_class
        
        for class_name in classes:
            class_dir = f'datasets/part2/{split}/{class_name}'
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(n_per_class):
                # Create a simple image with shapes
                img = Image.new('RGB', (img_size, img_size), color='white')
                pixels = np.array(img)
                
                # Add some noise
                noise = np.random.randint(-20, 20, (img_size, img_size, 3))
                pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
                
                # Draw shape
                center = img_size // 2
                radius = np.random.randint(15, 25)
                
                if class_name == 'circles':
                    # Draw a circle
                    y, x = np.ogrid[:img_size, :img_size]
                    mask = (x - center)**2 + (y - center)**2 <= radius**2
                    color = np.random.randint(100, 255, 3)
                    pixels[mask] = color
                    
                elif class_name == 'squares':
                    # Draw a square
                    size = radius * 2
                    top = center - radius
                    bottom = center + radius
                    left = center - radius
                    right = center + radius
                    color = np.random.randint(100, 255, 3)
                    pixels[top:bottom, left:right] = color
                    
                elif class_name == 'triangles':
                    # Draw a triangle
                    for row in range(img_size):
                        for col in range(img_size):
                            # Simple triangle condition
                            if (abs(col - center) < (img_size - row) / 3 and 
                                row > center - radius and row < center + radius):
                                color = np.random.randint(100, 255, 3)
                                pixels[row, col] = color
                                
                elif class_name == 'crosses':
                    # Draw a cross
                    thickness = 5
                    color = np.random.randint(100, 255, 3)
                    # Vertical line
                    pixels[center-radius:center+radius, center-thickness:center+thickness] = color
                    # Horizontal line
                    pixels[center-thickness:center+thickness, center-radius:center+radius] = color
                
                # Save image
                img = Image.fromarray(pixels)
                img.save(f'{class_dir}/img_{i:03d}.jpg')
            
            print(f"✓ Created {n_per_class} images in {class_dir}/")
    
    print(f"\n✓ Created image dataset with {len(classes)} classes")
    print(f"  - Training images: {n_train_per_class * len(classes)}")
    print(f"  - Test images: {n_test_per_class * len(classes)}")
    print()

def main():
    print("="*70)
    print("CREATING SAMPLE DATASETS FOR DEMONSTRATION")
    print("="*70)
    print()
    
    # Create Part 1 dataset
    create_sample_csv_dataset()
    
    # Create Part 2 dataset
    create_sample_image_dataset()
    
    print("="*70)
    print("✓ ALL SAMPLE DATASETS CREATED SUCCESSFULLY")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Run: python part1_threshold_optimization_enhanced.py")
    print("   (Will use datasets/part1/sample_model_predictions.csv)")
    print()
    print("2. Run: python part2_cifar10_cnn_enhanced.py")
    print("   (Will use datasets/part2/train/ and datasets/part2/test/)")
    print()

if __name__ == "__main__":
    main()
