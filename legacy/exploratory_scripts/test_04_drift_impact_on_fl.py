"""
Test 4: Impact of Drift on Federated Learning

Demonstrates how drifted clients affect global model performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from typing import Dict, List
import random

from src.data_pipeline import PAMAP2Preprocessor
from src.data_pipeline.windowing import create_data_pipeline, ActivityID
from src.drift_detection.lstm_model import LSTMDriftDetector, LSTMConfig
from src.federated.aggregator import FedAvgAggregator
from src.data_pipeline.drift_injection import DriftInjector, DriftConfig, DriftType

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_subject_data(subject_id: int, test_split: float = 0.2) -> Dict:
    """Load and prepare data for a subject."""
    preprocessor = PAMAP2Preprocessor()
    filepath = f"PAMAP2_Dataset/Protocol/subject{subject_id}.dat"

    accel_data, activity_labels = preprocessor.preprocess_file(filepath)

    pipeline = create_data_pipeline(
        window_size=256,
        stride=128,
        return_raw_windows=True
    )

    windows, labels = pipeline.process(accel_data, activity_labels)

    valid_mask = labels > 0
    windows = windows[valid_mask]
    labels = labels[valid_mask]

    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])

    indices = np.random.permutation(len(windows))
    n_test = int(len(windows) * test_split)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return {
        'train_windows': windows[train_indices],
        'train_labels': remapped_labels[train_indices],
        'test_windows': windows[test_indices],
        'test_labels': remapped_labels[test_indices],
        'n_classes': len(unique_labels)
    }


def inject_drift_to_data(windows: np.ndarray, magnitude: float = 8.0) -> np.ndarray:
    """Inject strong drift into windows."""
    drifted = windows.copy()
    # Add strong bias drift
    drifted[:, :, 0] += magnitude
    # Add noise
    drifted[:, :, 1] += np.random.normal(0, 3.0, drifted[:, :, 1].shape)
    return drifted


def train_global_model(
    client_ids: List[int],
    client_data: Dict[int, Dict],
    n_classes: int,
    n_rounds: int = 10
) -> LSTMDriftDetector:
    """Train a global model to convergence."""
    global_model = LSTMDriftDetector(
        lstm_config=LSTMConfig(hidden_size=64, num_layers=2, output_dim=n_classes)
    )

    print("    Training baseline model...", end='', flush=True)
    for round_num in range(n_rounds):
        aggregator = FedAvgAggregator()

        for client_id in client_ids:
            data = client_data[client_id]

            client = LSTMDriftDetector(
                lstm_config=LSTMConfig(hidden_size=64, num_layers=2, output_dim=n_classes)
            )
            client.model.set_weights(global_model.model.get_weights())

            client.train_batch(data['train_windows'], data['train_labels'])

            weights = client.model.get_weights()
            aggregator.receive_update(client_id, weights, len(data['train_windows']))

        aggregated = aggregator.aggregate()
        if aggregated:
            global_model.model.set_weights(aggregated)

    print(f" done")
    return global_model


def train_and_evaluate(
    client_ids: List[int],
    client_data: Dict[int, Dict],
    drifted_clients: List[int],
    n_classes: int,
    n_rounds: int = 10
) -> float:
    """Train model from scratch with drifted clients and return accuracy."""
    model = LSTMDriftDetector(
        lstm_config=LSTMConfig(hidden_size=64, num_layers=2, output_dim=n_classes)
    )

    for _ in range(n_rounds):
        aggregator = FedAvgAggregator()

        for client_id in client_ids:
            data = client_data[client_id].copy()

            if client_id in drifted_clients:
                data['train_windows'] = inject_drift_to_data(data['train_windows'])

            client = LSTMDriftDetector(
                lstm_config=LSTMConfig(hidden_size=64, num_layers=2, output_dim=n_classes)
            )
            client.model.set_weights(model.model.get_weights())
            client.train_batch(data['train_windows'], data['train_labels'])

            weights = client.model.get_weights()
            aggregator.receive_update(client_id, weights, len(data['train_windows']))

        aggregated = aggregator.aggregate()
        if aggregated:
            model.model.set_weights(aggregated)

    # Evaluate on clean test data
    accuracies = []
    for client_id in client_ids:
        data = client_data[client_id]
        acc, _ = model.evaluate(data['test_windows'], data['test_labels'])
        accuracies.append(acc)

    return np.mean(accuracies) * 100


def main():
    print("="*70)
    print("TEST 4: DRIFT IMPACT ON FEDERATED LEARNING")
    print("="*70)

    client_ids = [101, 102, 106, 107]
    np.random.seed(42)
    random.seed(42)

    print(f"\nSelected clients: {client_ids}")
    print("Loading data...")

    client_data = {}
    for cid in client_ids:
        client_data[cid] = load_subject_data(cid)
        print(f"  Client {cid}: {client_data[cid]['n_classes']} classes")

    n_classes = max(d['n_classes'] for d in client_data.values())

    # Run scenarios
    print(f"\n{'='*70}")
    print("RUNNING SCENARIOS")
    print(f"{'='*70}")

    results = {}

    # Scenario 1: Normal FL (no drift)
    print("\n[1/3] Normal FL (no drift)...")
    acc_no_drift = train_and_evaluate(client_ids, client_data, [], n_classes, n_rounds=12)
    results['no_drift'] = acc_no_drift
    print(f"    Accuracy: {acc_no_drift:.1f}%")

    # Scenario 2: 1 client drifted
    print("\n[2/3] 1 client drifted (Client 101)...")
    acc_1_drift = train_and_evaluate(client_ids, client_data, [101], n_classes, n_rounds=12)
    results['1_drifted'] = acc_1_drift
    print(f"    Accuracy: {acc_1_drift:.1f}%")

    # Scenario 3: 2 clients drifted
    print("\n[3/3] 2 clients drifted (Clients 101, 102)...")
    acc_2_drift = train_and_evaluate(client_ids, client_data, [101, 102], n_classes, n_rounds=12)
    results['2_drifted'] = acc_2_drift
    print(f"    Accuracy: {acc_2_drift:.1f}%")

    # Use no_drift as baseline for comparison
    baseline_acc = results['no_drift']

    # Print results matrix
    print(f"\n{'='*70}")
    print("RESULTS MATRIX")
    print(f"{'='*70}")

    print(f"\n{'Scenario':<25} {'Accuracy (%)':<17} {'Change from Baseline (%)':<20}")
    print("-" * 70)

    scenarios = [
        ('Normal FL (no drift)', results['no_drift']),
        ('1 client drifted', results['1_drifted']),
        ('2 clients drifted', results['2_drifted'])
    ]

    for name, acc in scenarios:
        change = acc - baseline_acc
        change_str = f"{change:+.1f}%"
        print(f"{name:<25} {acc:<17.1f} {change_str:<20}")

    # Generate graph
    print(f"\nGenerating graph...")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    accuracies = [results['no_drift'], results['1_drifted'], results['2_drifted']]
    colors = ['#27ae60', '#e67e22', '#c0392b']

    bars = ax.bar(x, accuracies, 0.6, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Impact of Drift on Federated Learning', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Normal FL\n(No Drift)', '1 Client\nDrifted', '2 Clients\nDrifted'])
    ax.set_ylim([40, 80])

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        change = height - baseline_acc
        label = f'{height:.1f}%\n({change:+.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add baseline reference line
    ax.axhline(y=baseline_acc, color='steelblue', linestyle='--', linewidth=2,
               label=f'Baseline: {baseline_acc:.1f}%')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('results/drift_impact_on_fl.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] results/drift_impact_on_fl.png")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    no_drift_change = results['no_drift'] - baseline_acc
    one_drift_change = results['1_drifted'] - baseline_acc
    two_drift_change = results['2_drifted'] - baseline_acc

    print(f"\nBaseline (Normal FL): {baseline_acc:.1f}%")
    print(f"1 client drifted: {results['1_drifted']:.1f}% ({one_drift_change:+.1f}%)")
    print(f"2 clients drifted: {results['2_drifted']:.1f}% ({two_drift_change:+.1f}%)")

    print("\n" + "="*70)
    print("TEST 4 COMPLETE")
    print("="*70)

    return results


if __name__ == "__main__":
    main()
