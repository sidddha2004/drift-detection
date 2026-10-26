"""
Test 2 Enhanced: LSTM Multi-Client Analysis with Visualizations

This test extends the single-client test to:
1. Compare LSTM performance across ALL subjects
2. Generate comparison charts (accuracy, loss, training curves)
3. Calculate per-subject feature importance

Features (9 total - 3 sensors × 3 axes):
- Sensor 1 (chest): x, y, z
- Sensor 2 (ankle): x, y, z
- Sensor 3 (hand): x, y, z
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
import json
from typing import Dict, List, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from src.data_pipeline import PAMAP2Preprocessor
from src.data_pipeline.windowing import create_data_pipeline, ActivityID
from src.drift_detection.lstm_model import LSTMDriftDetector, LSTMConfig

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ClientMetrics:
    """Track training and evaluation metrics for a single client."""

    def __init__(self, subject_id: int):
        self.subject_id = subject_id
        self.train_losses = []
        self.train_accuracies = []
        self.eval_accuracies = []
        self.training_time = 0.0
        self.label_names = []
        self.n_train_samples = 0
        self.n_test_samples = 0

    def add_train_metrics(self, loss: float, accuracy: float):
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)

    def get_summary(self) -> Dict:
        return {
            'subject_id': self.subject_id,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else None,
            'training_time': self.training_time,
            'n_train_samples': self.n_train_samples,
            'n_test_samples': self.n_test_samples
        }


class FeatureImportanceCalculator:
    """Calculate feature importance for LSTM models."""

    FEATURE_NAMES = [
        'Chest X', 'Chest Y', 'Chest Z',
        'Ankle X', 'Ankle Y', 'Ankle Z',
        'Hand X', 'Hand Y', 'Hand Z'
    ]

    SENSOR_NAMES = ['Chest', 'Ankle', 'Hand']

    @staticmethod
    def permutation_importance(
        detector: LSTMDriftDetector,
        windows: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 500
    ) -> np.ndarray:
        """
        Calculate permutation importance by measuring accuracy drop
        when each feature is shuffled.

        Returns:
            Array of importance scores (one per feature)
        """
        original_acc, _ = detector.evaluate(windows, labels)

        importance_scores = []

        # Sample subset for efficiency
        n_eval = min(n_samples, len(windows))
        indices = np.random.choice(len(windows), n_eval, replace=False)
        eval_windows = windows[indices].copy()
        eval_labels = labels[indices]

        for feature_idx in range(9):
            # Shuffle this feature
            shuffled_windows = eval_windows.copy()
            np.random.seed(42)
            shuffled_windows[:, :, feature_idx] = np.random.permutation(
                shuffled_windows[:, :, feature_idx]
            )

            # Evaluate with shuffled feature
            acc_with_noise, _ = detector.evaluate(shuffled_windows, eval_labels)

            # Importance = drop in accuracy
            importance = original_acc - acc_with_noise
            importance_scores.append(importance)

        return np.array(importance_scores)

    @staticmethod
    def gradient_importance(
        detector: LSTMDriftDetector,
        windows: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 200
    ) -> np.ndarray:
        """
        Calculate gradient-based importance using saliency maps.
        Measures how much output changes with small input changes.

        Returns:
            Array of importance scores (one per feature)
        """
        detector.model.eval()

        n_eval = min(n_samples, len(windows))
        indices = np.random.choice(len(windows), n_eval, replace=False)

        windows_tensor = torch.from_numpy(windows[indices]).float()
        labels_tensor = torch.from_numpy(labels[indices]).long()
        windows_tensor.requires_grad = True

        # Forward pass
        logits = detector.model(windows_tensor)
        loss = detector.criterion(logits, labels_tensor)

        # Backward pass
        detector.model.zero_grad()
        loss.backward()

        # Get gradients - absolute magnitude = importance
        gradients = windows_tensor.grad.abs()  # (n, seq_len, 9)

        # Average over samples and time steps
        importance = gradients.mean(dim=(0, 1)).detach().numpy()

        return importance

    @staticmethod
    def aggregate_by_sensor(feature_importance: np.ndarray) -> Dict[str, float]:
        """Aggregate 9 feature values by sensor (3 features each)."""
        sensor_importance = {}
        for i, sensor in enumerate(FeatureImportanceCalculator.SENSOR_NAMES):
            start_idx = i * 3
            end_idx = start_idx + 3
            sensor_importance[sensor] = feature_importance[start_idx:end_idx].sum()
        return sensor_importance


def load_and_prepare_subject_data(
    subject_id: int,
    test_split: float = 0.2
) -> Tuple:
    """Load data for a single subject."""
    preprocessor = PAMAP2Preprocessor()
    filepath = f"PAMAP2_Dataset/Protocol/subject{subject_id}.dat"

    try:
        accel_data, activity_labels = preprocessor.preprocess_file(filepath)
    except Exception as e:
        print(f"  [ERROR] Failed to load subject {subject_id}: {e}")
        return None

    # Create pipeline
    pipeline = create_data_pipeline(
        window_size=256,
        stride=128,
        return_raw_windows=True
    )

    windows, labels = pipeline.process(accel_data, activity_labels)

    # Filter out transient activity
    valid_mask = labels > 0
    windows = windows[valid_mask]
    labels = labels[valid_mask]

    if len(windows) == 0:
        print(f"  [WARN] No valid windows for subject {subject_id}")
        return None

    # Map labels to contiguous range
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])

    # Get label names
    label_names = [ActivityID(l).name for l in unique_labels]

    # Split
    n_test = int(len(windows) * test_split)
    indices = np.random.permutation(len(windows))

    train_indices = indices[n_test:]
    test_indices = indices[:n_test]

    return {
        'train_windows': windows[train_indices],
        'train_labels': remapped_labels[train_indices],
        'test_windows': windows[test_indices],
        'test_labels': remapped_labels[test_indices],
        'label_names': label_names
    }


def train_subject_model(
    subject_id: int,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    n_epochs: int = 15,
    batch_size: int = 64
) -> Tuple[LSTMDriftDetector, ClientMetrics]:
    """Train LSTM model for a single subject."""
    metrics = ClientMetrics(subject_id)
    metrics.n_train_samples = len(train_windows)

    lstm_config = LSTMConfig(
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001
    )

    detector = LSTMDriftDetector(lstm_config=lstm_config)

    n_samples = len(train_windows)
    start_time = time.time()

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_windows = train_windows[batch_idx]
            batch_labels = train_labels[batch_idx]

            detector.train_batch(batch_windows, batch_labels)

        # Evaluate on training set
        train_acc, train_loss = detector.evaluate(train_windows, train_labels)
        metrics.add_train_metrics(train_loss, train_acc)

    metrics.training_time = time.time() - start_time
    metrics.n_test_samples = 0  # Will be updated after split

    return detector, metrics


def evaluate_model(
    detector: LSTMDriftDetector,
    test_windows: np.ndarray,
    test_labels: np.ndarray
) -> Dict:
    """Evaluate model and return comprehensive metrics."""
    detector.model.eval()

    all_predictions = []

    batch_size = 64
    for i in range(0, len(test_windows), batch_size):
        batch_windows = test_windows[i:i + batch_size]

        windows_tensor = torch.from_numpy(batch_windows).float()

        with torch.no_grad():
            logits = detector.model(windows_tensor)
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_predictions.append(predictions.numpy())

    predictions = np.concatenate(all_predictions)

    accuracy = (predictions == test_labels).mean()

    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)

    cm = confusion_matrix(test_labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions
    }


def generate_comparison_plots(all_results: List[Dict], output_dir: str = "results"):
    """Generate comparison visualizations across all subjects."""
    os.makedirs(output_dir, exist_ok=True)

    subject_ids = [r['subject_id'] for r in all_results]
    accuracies = [r['eval_metrics']['accuracy'] for r in all_results]
    precisions = [r['eval_metrics']['precision'] for r in all_results]
    recalls = [r['eval_metrics']['recall'] for r in all_results]
    f1_scores = [r['eval_metrics']['f1'] for r in all_results]
    training_times = [r['train_metrics']['training_time'] for r in all_results]

    # 1. Main metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    axes[0, 0].bar(subject_ids, accuracies, color='steelblue', edgecolor='black')
    axes[0, 0].axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
    axes[0, 0].set_xlabel('Subject ID')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Test Accuracy by Subject')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([min(accuracies) - 0.05, 1.0])

    # Precision, Recall, F1
    x = np.arange(len(subject_ids))
    width = 0.25
    axes[0, 1].bar(x - width, precisions, width, label='Precision', color='royalblue')
    axes[0, 1].bar(x, recalls, width, label='Recall', color='forestgreen')
    axes[0, 1].bar(x + width, f1_scores, width, label='F1', color='orange')
    axes[0, 1].set_xlabel('Subject ID')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision, Recall, F1 by Subject')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(subject_ids)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])

    # Training time
    axes[1, 0].bar(subject_ids, training_times, color='coral', edgecolor='black')
    axes[1, 0].axhline(y=np.mean(training_times), color='red', linestyle='--', label=f'Mean: {np.mean(training_times):.1f}s')
    axes[1, 0].set_xlabel('Subject ID')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time by Subject')
    axes[1, 0].legend()

    # Combined radar-style metrics (normalized)
    metrics_matrix = np.array([accuracies, precisions, recalls, f1_scores])
    im = axes[1, 1].imshow(metrics_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_xticks(np.arange(len(subject_ids)))
    axes[1, 1].set_xticklabels(subject_ids)
    axes[1, 1].set_yticks(np.arange(4))
    axes[1, 1].set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
    axes[1, 1].set_title('Metrics Heatmap')
    axes[1, 1].figure.colorbar(im, ax=axes[1, 1])

    # Add values to heatmap
    for i in range(4):
        for j in range(len(subject_ids)):
            text = axes[1, 1].text(j, i, f'{metrics_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/subject_comparison.png", dpi=150)
    plt.close()

    # 2. Training curves
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for r in all_results:
        sid = r['subject_id']
        train_acc = r.get('training_curves', {}).get('train_accuracies', [])
        epochs = range(1, len(train_acc) + 1)
        axes[0].plot(epochs, train_acc, marker='o', label=f'Subject {sid}', alpha=0.7)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Accuracy')
    axes[0].set_title('Training Accuracy Curves')
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for r in all_results:
        sid = r['subject_id']
        train_loss = r.get('training_curves', {}).get('train_losses', [])
        epochs = range(1, len(train_loss) + 1)
        axes[1].plot(epochs, train_loss, marker='s', label=f'Subject {sid}', alpha=0.7)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Training Loss Curves')
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()

    # 3. Box plot summary
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [accuracies, precisions, recalls, f1_scores]
    bp = ax.boxplot(data_to_plot, labels=['Accuracy', 'Precision', 'Recall', 'F1'],
                    patch_artist=True, showmeans=True)

    for patch, color in zip(bp['boxes'], ['steelblue', 'royalblue', 'forestgreen', 'orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Score')
    ax.set_title('Metric Distribution Across All Subjects')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add statistics
    stats_text = ""
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
        data = data_to_plot[i]
        stats_text += f"{metric}: μ={np.mean(data):.3f}, σ={np.std(data):.3f}\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metric_distribution.png", dpi=150)
    plt.close()

    print(f"\n  [SAVED] Comparison plots to {output_dir}/")


def generate_feature_importance_plots(
    all_importance_results: List[Dict],
    output_dir: str = "results"
):
    """Generate feature importance visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    subject_ids = [r['subject_id'] for r in all_importance_results]

    # 1. Per-subject feature importance (bar plots)
    n_subjects = len(subject_ids)
    n_cols = 3
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_subjects > 1 else [axes]

    for i, (sid, importance_data) in enumerate(zip(subject_ids, all_importance_results)):
        ax = axes[i]
        perm_importance = importance_data['permutation_importance']

        colors = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'green', 'green', 'green']
        bars = ax.barh(FeatureImportanceCalculator.FEATURE_NAMES, perm_importance, color=colors, edgecolor='black')
        ax.set_xlabel('Importance (Accuracy Drop)')
        ax.set_title(f'Subject {sid} - Feature Importance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, perm_importance):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=8)

    for j in range(n_subjects, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_per_subject.png", dpi=150)
    plt.close()

    # 2. Sensor-level aggregation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Permutation importance by sensor
    sensor_data = {'Chest': [], 'Ankle': [], 'Hand': []}
    for r in all_importance_results:
        sensor_agg = FeatureImportanceCalculator.aggregate_by_sensor(r['permutation_importance'])
        for sensor in sensor_data:
            sensor_data[sensor].append(sensor_agg[sensor])

    x = np.arange(len(subject_ids))
    width = 0.25

    axes[0].bar(x - width, sensor_data['Chest'], width, label='Chest', color='red', alpha=0.7)
    axes[0].bar(x, sensor_data['Ankle'], width, label='Ankle', color='blue', alpha=0.7)
    axes[0].bar(x + width, sensor_data['Hand'], width, label='Hand', color='green', alpha=0.7)

    axes[0].set_xlabel('Subject ID')
    axes[0].set_ylabel('Permutation Importance')
    axes[0].set_title('Sensor Importance by Subject')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subject_ids)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Average sensor importance
    avg_sensors = {s: np.mean(v) for s, v in sensor_data.items()}
    axes[1].bar(avg_sensors.keys(), avg_sensors.values(),
               color=['red', 'blue', 'green'], edgecolor='black', alpha=0.7)

    for sensor, imp in avg_sensors.items():
        axes[1].text(sensor, imp, f'{imp:.3f}', ha='center', va='bottom')

    axes[1].set_ylabel('Average Permutation Importance')
    axes[1].set_title('Average Sensor Importance Across All Subjects')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensor_importance_comparison.png", dpi=150)
    plt.close()

    # 3. Feature importance heatmap
    perm_matrix = np.array([r['permutation_importance'] for r in all_importance_results])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(perm_matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(FeatureImportanceCalculator.FEATURE_NAMES, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(subject_ids)))
    ax.set_yticklabels(subject_ids)
    ax.set_title('Permutation Importance Heatmap (Feature × Subject)')

    plt.colorbar(im, ax=ax, label='Importance (Accuracy Drop)')

    # Add values
    for i in range(len(subject_ids)):
        for j in range(9):
            text = ax.text(j, i, f'{perm_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_heatmap.png", dpi=150)
    plt.close()

    print(f"  [SAVED] Feature importance plots to {output_dir}/")


def print_feature_importance_summary(all_importance_results: List[Dict]):
    """Print summary of feature importance findings."""
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE SUMMARY")
    print(f"{'='*80}")

    # Average importance across all subjects
    perm_importances = np.array([r['permutation_importance'] for r in all_importance_results])
    avg_perm = perm_importances.mean(axis=0)

    print("\nAverage Permutation Importance (all subjects):")
    print("-" * 50)
    for name, imp in zip(FeatureImportanceCalculator.FEATURE_NAMES, avg_perm):
        print(f"  {name:<10}: {imp:.4f}")

    # By sensor
    print("\nAverage by Sensor:")
    print("-" * 30)
    sensor_importance = {}
    for i, sensor in enumerate(FeatureImportanceCalculator.SENSOR_NAMES):
        start_idx = i * 3
        end_idx = start_idx + 3
        sensor_importance[sensor] = avg_perm[start_idx:end_idx].sum()
        print(f"  {sensor}: {sensor_importance[sensor]:.4f}")

    # Most important overall
    top_features = sorted(
        zip(FeatureImportanceCalculator.FEATURE_NAMES, avg_perm),
        key=lambda x: -x[1]
    )
    print("\nTop 5 Most Important Features:")
    print("-" * 40)
    for i, (name, imp) in enumerate(top_features[:5], 1):
        print(f"  {i}. {name}: {imp:.4f}")

    # Per-subject top features
    print(f"\n{'='*80}")
    print("TOP 3 FEATURES PER SUBJECT")
    print(f"{'='*80}")

    for r in all_importance_results:
        sid = r['subject_id']
        perm = r['permutation_importance']

        top_idx = np.argsort(perm)[-3:][::-1]
        top_features = [(FeatureImportanceCalculator.FEATURE_NAMES[i], perm[i]) for i in top_idx]

        print(f"\nSubject {sid}:")
        for i, (name, imp) in enumerate(top_features, 1):
            print(f"  {i}. {name}: {imp:.4f}")


def main():
    """Run comprehensive multi-subject LSTM analysis."""
    print("="*80)
    print("TEST 2 ENHANCED: LSTM MULTI-CLIENT ANALYSIS")
    print("="*80)

    # Configuration
    all_subjects = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    n_epochs = 15
    batch_size = 64

    np.random.seed(42)
    torch.manual_seed(42)

    all_results = []
    all_importance_results = []

    for subject_id in all_subjects:
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBJECT {subject_id}")
        print(f"{'='*80}")

        # Load data
        data = load_and_prepare_subject_data(subject_id, test_split=0.2)
        if data is None:
            continue

        print(f"  Train samples: {len(data['train_windows'])}")
        print(f"  Test samples: {len(data['test_windows'])}")
        print(f"  Activities: {len(data['label_names'])}")

        # Train model
        print(f"\n  Training LSTM model...")
        detector, metrics = train_subject_model(
            subject_id,
            data['train_windows'],
            data['train_labels'],
            n_epochs=n_epochs,
            batch_size=batch_size
        )
        metrics.n_test_samples = len(data['test_windows'])
        metrics.label_names = data['label_names']

        print(f"  Training time: {metrics.training_time:.2f}s")

        # Evaluate
        print(f"  Evaluating model...")
        eval_results = evaluate_model(
            detector,
            data['test_windows'],
            data['test_labels']
        )

        print(f"  Test Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  F1 Score: {eval_results['f1']:.4f}")

        # Feature importance
        print(f"  Calculating feature importance...")
        perm_importance = FeatureImportanceCalculator.permutation_importance(
            detector, data['test_windows'], data['test_labels'], n_samples=500
        )

        grad_importance = FeatureImportanceCalculator.gradient_importance(
            detector, data['test_windows'], data['test_labels'], n_samples=200
        )

        # Store results
        all_results.append({
            'subject_id': subject_id,
            'train_metrics': metrics.get_summary(),
            'eval_metrics': {
                'accuracy': eval_results['accuracy'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'f1': eval_results['f1']
            },
            'training_curves': {
                'train_losses': metrics.train_losses,
                'train_accuracies': metrics.train_accuracies
            }
        })

        all_importance_results.append({
            'subject_id': subject_id,
            'permutation_importance': perm_importance,
            'gradient_importance': grad_importance
        })

    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    generate_comparison_plots(all_results)
    generate_feature_importance_plots(all_importance_results)

    # Print summaries
    print_feature_importance_summary(all_importance_results)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    avg_acc = np.mean([r['eval_metrics']['accuracy'] for r in all_results])
    avg_f1 = np.mean([r['eval_metrics']['f1'] for r in all_results])
    avg_time = np.mean([r['train_metrics']['training_time'] for r in all_results])

    print(f"Subjects evaluated: {len(all_results)}")
    print(f"Average test accuracy: {avg_acc:.4f}")
    print(f"Average F1 score: {avg_f1:.4f}")
    print(f"Average training time: {avg_time:.2f}s")

    # Save results to JSON
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/multi_subject_results.json", 'w') as f:
        json.dump({
            'results': all_results,
            'feature_importance': [
                {
                    'subject_id': r['subject_id'],
                    'permutation_importance': r['permutation_importance'].tolist(),
                    'gradient_importance': r['gradient_importance'].tolist()
                }
                for r in all_importance_results
            ]
        }, f, indent=2)

    print(f"\n[SAVED] Results to {output_dir}/multi_subject_results.json")

    print("\n" + "="*80)
    print("TEST 2 ENHANCED COMPLETE")
    print("="*80)

    return all_results, all_importance_results


if __name__ == "__main__":
    main()
