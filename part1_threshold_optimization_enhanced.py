"""
Enhanced Part 1: Model Evaluation & Threshold Optimization
Now supports loading custom datasets from /datasets/part1 folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import os
import glob

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_custom_dataset(dataset_path='datasets/part1'):
    """
    Load custom dataset from the datasets folder.
    
    The CSV should have columns: true_label, confidence_score, predicted_label (optional)
    If predicted_label is missing, it will be generated using threshold 0.5
    
    Args:
        dataset_path: Path to the dataset folder
        
    Returns:
        DataFrame with model outputs
    """
    print(f"\nLooking for datasets in: {dataset_path}")
    
    # Find CSV files in the dataset path
    csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {dataset_path}")
        return None
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Use the first CSV file or let user specify
    csv_file = csv_files[0]
    print(f"\nLoading: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['true_label', 'confidence_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nError: Missing required columns: {missing_cols}")
        print("CSV should have columns: true_label, confidence_score, predicted_label (optional)")
        return None
    
    # Generate predicted_label if not present
    if 'predicted_label' not in df.columns:
        print("Generating predicted_label using threshold 0.5")
        df['predicted_label'] = (df['confidence_score'] >= 0.5).astype(int)
    
    # Validate data
    if df['true_label'].isnull().any() or df['confidence_score'].isnull().any():
        print("\nWarning: Dataset contains null values. Removing them...")
        df = df.dropna(subset=['true_label', 'confidence_score'])
    
    print(f"\nDataset preview:")
    print(df.head(10))
    print(f"\nClass distribution:")
    print(df['true_label'].value_counts())
    print(f"\nConfidence score statistics:")
    print(df['confidence_score'].describe())
    
    return df


def generate_sample_dataset(n_samples=1000, save_path='datasets/part1/model_outputs.csv'):
    """
    Generate a sample dataset simulating model outputs.
    
    In real scenarios, this would be your actual model predictions.
    The dataset includes:
    - true_label: Ground truth labels (0 or 1 for binary classification)
    - confidence_score: Model's confidence in predicting class 1 (0.0 to 1.0)
    - predicted_label: Predicted label using default threshold of 0.5
    """
    np.random.seed(42)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate true labels (imbalanced dataset - more negatives than positives)
    true_labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Generate confidence scores based on true labels with some noise
    # Positive samples tend to have higher confidence scores
    confidence_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        if true_labels[i] == 1:
            # True positives: higher confidence with some noise
            confidence_scores[i] = np.clip(np.random.beta(8, 2), 0, 1)
        else:
            # True negatives: lower confidence with some noise
            confidence_scores[i] = np.clip(np.random.beta(2, 8), 0, 1)
    
    # Generate predicted labels using default threshold of 0.5
    predicted_labels = (confidence_scores >= 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'true_label': true_labels,
        'confidence_score': confidence_scores,
        'predicted_label': predicted_labels
    })
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Sample dataset saved to {save_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset preview:")
    print(df.head(10))
    print(f"\nClass distribution:")
    print(df['true_label'].value_counts())
    
    return df


def analyze_model_performance(df, threshold=0.5):
    """
    Analyze model performance at a given threshold.
    """
    y_true = df['true_label'].values
    y_pred = (df['confidence_score'].values >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Performance at threshold = {threshold:.3f}")
    print(f"{'='*60}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def find_optimal_threshold(df, metric='f1'):
    """
    Find the optimal confidence threshold by testing multiple thresholds.
    
    Args:
        df: DataFrame with true_label and confidence_score columns
        metric: Metric to optimize ('f1', 'accuracy', 'balanced', or 'youden')
    
    Returns:
        optimal_threshold: Best threshold value
        results_df: DataFrame with metrics for all thresholds
    """
    y_true = df['true_label'].values
    y_scores = df['confidence_score'].values
    
    # Test thresholds from 0.1 to 0.9 in steps of 0.01
    thresholds = np.arange(0.1, 0.91, 0.01)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate precision and recall manually to handle edge cases
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Youden's J statistic (sensitivity + specificity - 1)
        youden = recall + specificity - 1
        
        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'youden_j': youden,
            'balanced_accuracy': balanced_acc
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold based on chosen metric
    if metric == 'f1':
        optimal_idx = results_df['f1_score'].idxmax()
        metric_name = 'F1 Score'
    elif metric == 'accuracy':
        optimal_idx = results_df['accuracy'].idxmax()
        metric_name = 'Accuracy'
    elif metric == 'balanced':
        optimal_idx = results_df['balanced_accuracy'].idxmax()
        metric_name = 'Balanced Accuracy'
    elif metric == 'youden':
        optimal_idx = results_df['youden_j'].idxmax()
        metric_name = "Youden's J"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL THRESHOLD ANALYSIS (optimizing {metric_name})")
    print(f"{'='*60}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Metrics at optimal threshold:")
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']:
        print(f"  {col:20s}: {results_df.loc[optimal_idx, col]:.4f}")
    
    return optimal_threshold, results_df


def visualize_threshold_analysis(df, results_df, optimal_threshold, save_path='threshold_analysis.png'):
    """
    Create comprehensive visualizations for threshold analysis.
    """
    y_true = df['true_label'].values
    y_scores = df['confidence_score'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Precision-Recall vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['f1_score'], 'r-', label='F1 Score', linewidth=2)
    ax1.axvline(x=optimal_threshold, color='orange', linestyle='--', 
                label=f'Optimal (F1) = {optimal_threshold:.3f}', linewidth=2)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Precision, Recall, and F1 Score vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    ax2 = axes[0, 1]
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    
    ax2.plot(recall_curve, precision_curve, 'b-', linewidth=2, label='PR Curve')
    
    # Mark the optimal threshold point
    y_pred_opt = (y_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    opt_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    opt_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    ax2.plot(opt_recall, opt_precision, 'ro', markersize=10, 
             label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ROC Curve
    ax3 = axes[1, 0]
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Mark optimal threshold on ROC curve
    y_pred_opt = (y_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    opt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    opt_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    ax3.plot(opt_fpr, opt_tpr, 'ro', markersize=10, 
             label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Multiple metrics vs threshold
    ax4 = axes[1, 1]
    ax4.plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['f1_score'], 'r-', label='F1 Score', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['balanced_accuracy'], 'g-', 
             label='Balanced Accuracy', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['youden_j'], 'm-', 
             label="Youden's J", linewidth=2)
    ax4.axvline(x=optimal_threshold, color='orange', linestyle='--', 
                label=f'Optimal = {optimal_threshold:.3f}', linewidth=2)
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Multiple Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def suggest_improvements(df, optimal_threshold):
    """
    Analyze results and suggest post-processing improvements.
    """
    y_true = df['true_label'].values
    y_scores = df['confidence_score'].values
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*60}")
    print("POST-PROCESSING IMPROVEMENT SUGGESTIONS")
    print(f"{'='*60}")
    
    print("\n1. THRESHOLD ADJUSTMENT STRATEGIES:")
    print("   - Current optimal threshold maximizes F1 score")
    print("   - For high-risk applications: Increase threshold to reduce FP")
    print("   - For recall-critical tasks: Decrease threshold to reduce FN")
    print("   - Consider using different thresholds for different use cases")
    
    print("\n2. CONFIDENCE SCORE CALIBRATION:")
    print("   - Apply Platt scaling or isotonic regression for better calibration")
    print("   - Use temperature scaling if using neural networks")
    print("   - Verify confidence scores match empirical frequencies")
    
    print("\n3. ENSEMBLE METHODS:")
    print("   - Combine predictions from multiple models")
    print("   - Use weighted voting based on model confidence")
    print("   - Apply stacking with a meta-classifier")
    
    print("\n4. UNCERTAINTY-BASED FILTERING:")
    low_conf_count = np.sum((y_scores > 0.4) & (y_scores < 0.6))
    print(f"   - {low_conf_count} samples have confidence between 0.4-0.6 (uncertain)")
    print("   - Consider flagging uncertain predictions for human review")
    print("   - Implement reject option for very uncertain predictions")
    
    print("\n5. CLASS-SPECIFIC THRESHOLDS:")
    print("   - Use different thresholds for different classes if needed")
    print("   - Particularly useful for imbalanced datasets")
    print("   - Can help balance precision/recall per class")
    
    print("\n6. ERROR ANALYSIS:")
    print(f"   - False Positives: {fp} ({fp/(fp+tn)*100:.1f}% of negatives)")
    print(f"   - False Negatives: {fn} ({fn/(fn+tp)*100:.1f}% of positives)")
    print("   - Analyze misclassified samples to identify patterns")
    print("   - Look for common features in errors to improve model")
    
    print("\n7. ADDITIONAL DATA QUALITY CHECKS:")
    print("   - Remove or correct mislabeled data")
    print("   - Add more training data for underrepresented patterns")
    print("   - Use active learning to select informative samples")
    
    print("\n8. MODEL-SPECIFIC IMPROVEMENTS:")
    print("   - If deep learning: Try different architectures")
    print("   - Add regularization to prevent overfitting")
    print("   - Use data augmentation for better generalization")
    print("   - Fine-tune hyperparameters systematically")


def main():
    """
    Main execution function for Part 1: Threshold Optimization
    """
    print("="*60)
    print("PART 1: MODEL EVALUATION & THRESHOLD OPTIMIZATION")
    print("(Enhanced version with custom dataset support)")
    print("="*60)
    
    # Step 1: Try to load custom dataset, or generate sample
    print("\nStep 1: Loading dataset...")
    df = load_custom_dataset('datasets/part1')
    
    if df is None:
        print("\nNo custom dataset found. Generating sample dataset...")
        df = generate_sample_dataset(n_samples=1000, save_path='datasets/part1/model_outputs.csv')
    
    # Step 2: Analyze performance at default threshold (0.5)
    print("\nStep 2: Analyzing performance at default threshold...")
    analyze_model_performance(df, threshold=0.5)
    
    # Step 3: Find optimal threshold
    print("\nStep 3: Finding optimal threshold...")
    optimal_threshold, results_df = find_optimal_threshold(df, metric='f1')
    
    # Step 4: Analyze performance at optimal threshold
    print("\nStep 4: Analyzing performance at optimal threshold...")
    analyze_model_performance(df, threshold=optimal_threshold)
    
    # Step 5: Create visualizations
    print("\nStep 5: Creating visualizations...")
    visualize_threshold_analysis(df, results_df, optimal_threshold)
    
    # Step 6: Suggest improvements
    print("\nStep 6: Generating improvement suggestions...")
    suggest_improvements(df, optimal_threshold)
    
    # Step 7: Save results summary
    print("\nStep 7: Saving results summary...")
    summary = {
        'optimal_threshold': optimal_threshold,
        'metrics_at_optimal': results_df[results_df['threshold'] == optimal_threshold].to_dict('records')[0]
    }
    
    # Save results to CSV
    results_df.to_csv('threshold_results.csv', index=False)
    print("Results saved to threshold_results.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print(f"1. Optimal threshold: {optimal_threshold:.3f}")
    print(f"2. F1 Score: {summary['metrics_at_optimal']['f1_score']:.4f}")
    print(f"3. Precision: {summary['metrics_at_optimal']['precision']:.4f}")
    print(f"4. Recall: {summary['metrics_at_optimal']['recall']:.4f}")
    print("\nCheck 'threshold_analysis.png' for detailed visualizations")
    print("Check 'threshold_results.csv' for complete results")
    print("\nTo use your own dataset:")
    print("  1. Place CSV file in datasets/part1/ folder")
    print("  2. CSV should have columns: true_label, confidence_score")
    print("  3. Re-run this script")


if __name__ == "__main__":
    main()
