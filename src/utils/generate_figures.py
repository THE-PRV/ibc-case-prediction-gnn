"""
IBC Paper Visualizations

Generates publication-ready figures for the paper.

Outputs:
  - confusion_matrix.png
  - training_curves.png
  - baseline_comparison.png
  - class_distribution.png
  - confidence_distribution.png
  - roc_curves.png
  - pr_curves.png
  - per_class_metrics.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from pathlib import Path
import json
from typing import Dict, List

from ..utils.config import paths_config, inference_config


# Style setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colors
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#6C757D',
}
CLASS_COLORS = ['#2E86AB', '#A23B72', '#F18F01']
CLASS_NAMES = inference_config.class_names


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Publication-ready confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Proportion'})
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(history, save_path=None):
    """Training loss and validation metrics over epochs."""
    epochs = [h['epoch'] for h in history]
    loss = [h['loss'] for h in history]
    val_f1 = [h['val_f1'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(epochs, loss, color=COLORS['primary'], linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Validation F1
    axes[1].plot(epochs, val_f1, color=COLORS['secondary'], linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_baseline_comparison(results, save_path=None):
    """Compare model performance against baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    f1_scores = [results[m].get('f1', 0) for m in models]
    accuracies = [results[m].get('accuracy', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color=COLORS['secondary'])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_class_distribution(class_counts, save_path=None):
    """Distribution of classes in the dataset."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(CLASS_NAMES, class_counts, color=CLASS_COLORS)
    ax.set_xlabel('Outcome Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in Dataset')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_confidence_distribution(y_true, probs, save_path=None):
    """Distribution of prediction confidence."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidences = np.max(probs, axis=1)
    correct_mask = (probs.argmax(axis=1) == y_true)
    
    bins = np.linspace(0, 1, 21)
    
    ax.hist(confidences[correct_mask], bins=bins, alpha=0.7, 
            label='Correct', color=COLORS['success'])
    ax.hist(confidences[~correct_mask], bins=bins, alpha=0.7,
            label='Incorrect', color=COLORS['neutral'])
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_roc_curves(y_true, probs, save_path=None):
    """ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                color=CLASS_COLORS[i])
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_precision_recall_curves(y_true, probs, save_path=None):
    """Precision-Recall curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        
        ax.plot(recall, precision, linewidth=2,
                label=f'{class_name}',
                color=CLASS_COLORS[i])
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_per_class_metrics(report, save_path=None):
    """Bar chart of per-class metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in CLASS_NAMES]
        ax.bar(x + i*width, values, width, label=metric.capitalize(),
               color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']][i])
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def generate_all_figures(output_dir=None):
    """
    Generate all figures from training results.
    
    Args:
        output_dir: Directory to save figures (default: config figures_dir)
    """
    if output_dir is None:
        output_dir = paths_config.figures_dir
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating figures...")
    print(f"Output directory: {output_dir}")
    
    # Load results
    results_path = paths_config.output_dir / 'results.json'
    predictions_path = paths_config.output_dir / 'predictions.npz'
    history_path = paths_config.output_dir / 'history.json'
    
    if not results_path.exists():
        print(f"Warning: {results_path} not found")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Load predictions
    if predictions_path.exists():
        data = np.load(predictions_path)
        y_true = data['y_true']
        y_pred = data['y_pred']
        probs = data['probs']
    else:
        print(f"Warning: {predictions_path} not found")
        return
    
    # Load history
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []
    
    # Generate figures
    print("\n1. Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    
    if history:
        print("2. Training Curves")
        plot_training_curves(history, output_dir / "training_curves.png")
    
    print("3. Class Distribution")
    class_counts = np.bincount(y_true, minlength=3)
    plot_class_distribution(class_counts, output_dir / "class_distribution.png")
    
    print("4. Confidence Distribution")
    plot_confidence_distribution(y_true, probs, output_dir / "confidence_distribution.png")
    
    print("5. ROC Curves")
    plot_roc_curves(y_true, probs, output_dir / "roc_curves.png")
    
    print("6. Precision-Recall Curves")
    plot_precision_recall_curves(y_true, probs, output_dir / "pr_curves.png")
    
    print("7. Per-Class Metrics")
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    plot_per_class_metrics(report, output_dir / "per_class_metrics.png")
    
    print(f"\n✓ All figures saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate figures from training results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for figures")
    
    args = parser.parse_args()
    
    generate_all_figures(args.output)
