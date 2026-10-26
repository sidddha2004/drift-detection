"""
Test 2: LSTM Single Client Accuracy

Tests LSTM model accuracy on a randomly selected client with comprehensive metrics:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix
- Loss curve
- Training time

For 1 randomly selected subject.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
from typing import Dict, List, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import random

from src.data_pipeline import PAMAP2Preprocessor
from src.data_pipeline.windowing import create_data_pipeline
from src.drift_detection.lstm_model import LSTMDriftDetector, LSTMConfig


class ClientMetrics:
    """Track training and evaluation metrics."""

    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.training_time = 0.0

    def add_train_metrics(self, loss: float, accuracy: float):
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)

    def add_eval_metrics(self, loss: float, accuracy: float):
        self.eval_losses.append(loss)
        self.eval_accuracies.append(accuracy)

    def get_summary(self) -> Dict:
        return {
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else None,
            'final_eval_loss': self.eval_losses[-1] if self.eval_losses else None,
            'final_eval_acc': self.eval_accuracies[-1] if self.eval_accuracies else None,
            'best_eval_acc': max(self.eval_accuracies) if self.eval_accuracies else None,
            'training_time': self.training_time
        }


def load_client_data(subject_id: int, test_split: float = 0.2) -> Tuple:
    """
    Load and prepare data for a single client.

    Returns:
        Tuple of (train_windows, train_labels, test_windows, test_labels, label_names)
    """
    print(f"\n{'='*60}")
    print(f"LOADING CLIENT DATA: Subject {subject_id}")
    print(f"{'='*60}")

    preprocessor = PAMAP2Preprocessor()
    filepath = f"PAMAP2_Dataset/Protocol/subject{subject_id}.dat"

    # Load and preprocess
    accel_data, activity_labels = preprocessor.preprocess_file(filepath)
    print(f"  Loaded {len(activity_labels)} samples")

    # Create pipeline for LSTM (raw windows)
    pipeline = create_data_pipeline(
        window_size=256,
        stride=128,
        return_raw_windows=True
    )

    # Process data
    windows, labels = pipeline.process(accel_data, activity_labels)

    # Filter out transient activity (ID=0)
    valid_mask = labels > 0
    windows = windows[valid_mask]
    labels = labels[valid_mask]

    # Map labels to contiguous range
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])

    # Get label names
    from src.data_pipeline.windowing import ActivityID
    label_names = [ActivityID(l).name for l in unique_labels]

    print(f"  Created {len(windows)} windows")
    print(f"  Unique activities: {len(label_names)}")

    # Shuffle and split
    indices = np.random.permutation(len(windows))
    n_test = int(len(windows) * test_split)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_windows = windows[train_indices]
    train_labels = remapped_labels[train_indices]
    test_windows = windows[test_indices]
    test_labels = remapped_labels[test_indices]

    print(f"  Train: {len(train_windows)} windows")
    print(f"  Test: {len(test_windows)} windows")

    return train_windows, train_labels, test_windows, test_labels, label_names


def train_client(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 64
) -> Tuple[LSTMDriftDetector, ClientMetrics]:
    """
    Train LSTM model on client data.

    Returns:
        Tuple of (trained_detector, metrics)
    """
    print(f"\n{'='*60}")
    print("TRAINING LSTM MODEL")
    print(f"{'='*60}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")

    lstm_config = LSTMConfig(
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001
    )

    detector = LSTMDriftDetector(lstm_config=lstm_config)
    metrics = ClientMetrics()

    n_samples = len(train_windows)
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_accuracies = []

        # Shuffle each epoch
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_windows = train_windows[batch_idx]
            batch_labels = train_labels[batch_idx]

            # Train batch
            loss = detector.train_batch(batch_windows, batch_labels)
            epoch_losses.append(loss)

        # Evaluate on training set
        train_acc, train_loss = detector.evaluate(train_windows, train_labels)
        metrics.add_train_metrics(train_loss, train_acc)

        print(f"  Epoch {epoch+1}/{n_epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}")

    metrics.training_time = time.time() - start_time
    print(f"  Training time: {metrics.training_time:.2f}s")

    return detector, metrics


def evaluate_client(
    detector: LSTMDriftDetector,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    label_names: List[str]
) -> Dict:
    """
    Evaluate trained model with comprehensive metrics.

    Returns:
        Dict with all evaluation metrics
    """
    print(f"\n{'='*60}")
    print("EVALUATING MODEL")
    print(f"{'='*60}")

    detector.model.eval()

    # Get predictions
    all_predictions = []
    test_losses = []

    # Batch inference
    batch_size = 64
    for i in range(0, len(test_windows), batch_size):
        batch_windows = test_windows[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]

        windows_tensor = torch.from_numpy(batch_windows).float()
        labels_tensor = torch.from_numpy(batch_labels).long()

        with torch.no_grad():
            logits = detector.model(windows_tensor)
            loss = detector.criterion(logits, labels_tensor)
            test_losses.append(loss.item())

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_predictions.append(predictions.numpy())

    predictions = np.concatenate(all_predictions)

    # Overall metrics
    accuracy = (predictions == test_labels).mean()
    avg_loss = np.mean(test_losses)

    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Loss: {avg_loss:.4f}")

    # Per-class metrics
    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)

    # Top confused pairs
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 5:
                confused_pairs.append((label_names[i], label_names[j], cm[i, j]))
    confused_pairs.sort(key=lambda x: -x[2])

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'confused_pairs': confused_pairs[:5],
        'predictions': predictions,
        'label_names': label_names
    }


def print_per_class_metrics(
    test_labels: np.ndarray,
    predictions: np.ndarray,
    label_names: List[str]
):
    """Print detailed per-class metrics."""
    print(f"\n{'='*60}")
    print("PER-CLASS METRICS")
    print(f"{'='*60}")
    print(f"{'Activity':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)

    for i, label in enumerate(label_names):
        mask = test_labels == i
        support = mask.sum()
        if support == 0:
            continue

        # Binary metrics for this class
        y_true_binary = (test_labels == i).astype(int)
        y_pred_binary = (predictions == i).astype(int)

        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        print(f"{label:<25} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")


def print_confusion_summary(eval_results: Dict):
    """Print confusion matrix summary."""
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX (Top Confusions)")
    print(f"{'='*60}")

    confused_pairs = eval_results['confused_pairs']
    if confused_pairs:
        print("  Top confused pairs (true -> predicted, count):")
        for true_label, pred_label, count in confused_pairs:
            print(f"    {true_label} -> {pred_label}: {count}")
    else:
        print("  No significant confusion found.")


def main():
    """Run Test 2: LSTM Single Client Accuracy."""
    print("="*60)
    print("TEST 2: LSTM SINGLE CLIENT ACCURACY")
    print("="*60)

    # Random subject selection
    all_subjects = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    np.random.seed(42)
    random.seed(42)

    selected_subject = random.choice(all_subjects)
    print(f"\nRandomly selected subject: {selected_subject}")

    # Load data
    train_windows, train_labels, test_windows, test_labels, label_names = load_client_data(
        selected_subject, test_split=0.2
    )

    # Train model
    detector, train_metrics = train_client(
        train_windows, train_labels,
        n_epochs=10,
        batch_size=64
    )

    # Evaluate
    eval_results = evaluate_client(detector, test_windows, test_labels, label_names)

    # Per-class metrics
    print_per_class_metrics(test_labels, eval_results['predictions'], label_names)

    # Confusion summary
    print_confusion_summary(eval_results)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Subject: {selected_subject}")
    print(f"Training time: {train_metrics.training_time:.2f}s")
    print(f"Final train accuracy: {train_metrics.train_accuracies[-1]:.4f}")
    print(f"Final test accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test precision: {eval_results['precision']:.4f}")
    print(f"Test recall: {eval_results['recall']:.4f}")
    print(f"Test F1: {eval_results['f1']:.4f}")

    # Model info
    total_params, _ = detector.model.count_parameters()
    memory_size = detector.model.get_memory_size()
    print(f"\nModel parameters: {total_params:,}")
    print(f"Model size (float32): {memory_size['float32']:.1f} KB")
    print(f"Model size (int8): {memory_size['int8']:.1f} KB")

    print("\n" + "="*60)
    print("TEST 2 COMPLETE")
    print("="*60)

    return {
        'subject_id': selected_subject,
        'train_metrics': train_metrics.get_summary(),
        'eval_metrics': {
            'accuracy': eval_results['accuracy'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1': eval_results['f1']
        }
    }


if __name__ == "__main__":
    main()
