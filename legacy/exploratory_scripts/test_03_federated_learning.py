"""
Test 3: Federated Learning Global Model Accuracy

Tests federated learning with 5 randomly selected clients:
- FedAvg aggregation across clients
- Global model accuracy over FL rounds
- Per-client local accuracy
- Communication efficiency (weight transfer)
- Convergence analysis

For 5 randomly selected subjects with multiple FL rounds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
from typing import Dict, List, Tuple
import random

from src.data_pipeline import PAMAP2Preprocessor
from src.data_pipeline.windowing import create_data_pipeline, ActivityID
from src.drift_detection.lstm_model import LSTMDriftDetector, LSTMConfig
from src.federated.aggregator import FedAvgAggregator
from src.federated.fl_client import FLClient, FLClientConfig


class FLSimulation:
    """Simulate federated learning across multiple clients."""

    def __init__(
        self,
        subject_ids: List[int],
        n_rounds: int = 5,
        local_epochs: int = 3,
        batch_size: int = 64
    ):
        self.subject_ids = subject_ids
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # Common activities across all clients (determined after loading)
        self.common_activities = None
        self.n_classes = None

        # Initialize global model (will be set after loading data)
        self.global_model = None

        # Initialize aggregator
        self.aggregator = FedAvgAggregator()

        # Client data storage
        self.client_data = {}

        # Metrics tracking
        self.round_metrics = []

    def find_common_activities_and_prepare_data(self, test_split: float = 0.2):
        """Find common activities across all clients and prepare data."""
        print(f"\n{'='*60}")
        print("FINDING COMMON ACTIVITIES")
        print(f"{'='*60}")

        # Find activities present in ALL clients
        common_activities = None
        for subject_id in self.subject_ids:
            labels = self.client_data[subject_id]['original_labels']
            unique_activities = set(np.unique(labels))
            print(f"  Client {subject_id}: {len(unique_activities)} activities")
            if common_activities is None:
                common_activities = unique_activities
            else:
                common_activities = common_activities.intersection(unique_activities)

        common_activities = sorted(list(common_activities))
        self.common_activities = common_activities
        self.n_classes = len(common_activities)

        print(f"\n  Common activities across all clients: {self.n_classes}")
        for i, act_id in enumerate(common_activities):
            act_name = ActivityID(act_id).name
            print(f"    {i}: {act_name} (ID={act_id})")

        # Prepare data for each client (filter to common activities, remap labels)
        for subject_id in self.subject_ids:
            data = self.client_data[subject_id]

            # Filter to common activities
            common_mask = np.isin(data['original_labels'], common_activities)
            filtered_windows = data['train_windows'][common_mask]
            filtered_labels = data['original_labels'][common_mask]

            # Remap to contiguous range
            label_map = {old: new for new, old in enumerate(common_activities)}
            remapped_labels = np.array([label_map[l] for l in filtered_labels])

            # Get label names
            label_names = [ActivityID(l).name for l in common_activities]

            # Shuffle and split
            indices = np.random.permutation(len(filtered_windows))
            n_test = int(len(filtered_windows) * test_split)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]

            self.client_data[subject_id] = {
                'train_windows': filtered_windows[train_indices],
                'train_labels': remapped_labels[train_indices],
                'test_windows': filtered_windows[test_indices],
                'test_labels': remapped_labels[test_indices],
                'label_names': label_names,
                'n_classes': self.n_classes
            }

            print(f"  Client {subject_id}: Train={len(train_indices)}, Test={len(test_indices)}")

        # Initialize global model with correct number of classes
        print(f"\nInitializing global model with {self.n_classes} output classes...")
        self.global_model = LSTMDriftDetector(
            lstm_config=LSTMConfig(hidden_size=64, num_layers=2, output_dim=self.n_classes)
        )

    def load_client_data(self, subject_id: int, test_split: float = 0.2):
        """Load and prepare data for a client."""
        print(f"Loading subject {subject_id}...")

        preprocessor = PAMAP2Preprocessor()
        filepath = f"PAMAP2_Dataset/Protocol/subject{subject_id}.dat"

        # Load and preprocess
        accel_data, activity_labels = preprocessor.preprocess_file(filepath)

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

        # Store original labels for common activity filtering
        original_labels = labels.copy()

        self.client_data[subject_id] = {
            'train_windows': windows,
            'train_labels': original_labels,
            'label_names': [],
            'original_labels': original_labels
        }

        print(f"  Total samples: {len(windows)}, Unique activities: {len(np.unique(original_labels))}")

    def run_round(self, round_num: int) -> Dict:
        """Run one FL round."""
        print(f"\n{'='*60}")
        print(f"FL ROUND {round_num + 1}/{self.n_rounds}")
        print(f"{'='*60}")

        round_start = time.time()
        client_updates = []
        client_metrics = []

        # Each client trains locally
        for subject_id in self.subject_ids:
            print(f"\n  Client {subject_id}:")

            # Create client with global model
            fl_client = FLClient(
                client_id=subject_id,
                config=FLClientConfig(
                    local_epochs=self.local_epochs,
                    batch_size=self.batch_size,
                    local_learning_rate=0.001
                ),
                lstm_config=LSTMConfig(hidden_size=64, num_layers=2, output_dim=self.n_classes)
            )

            # Load global model weights
            global_weights = self.global_model.model.get_weights()
            fl_client.load_global_model(global_weights)

            # Get client data
            data = self.client_data[subject_id]

            # Train locally
            loss, acc = fl_client.train_local(
                data['train_windows'],
                data['train_labels'],
                epochs=self.local_epochs
            )
            print(f"    Local train: loss={loss:.4f}, acc={acc:.4f}")

            # Evaluate locally
            local_acc, local_loss = fl_client.evaluate(
                data['test_windows'],
                data['test_labels']
            )
            print(f"    Local test: loss={local_loss:.4f}, acc={local_acc:.4f}")

            # Get update payload
            payload = fl_client.get_upload_payload(local_acc)

            # Update aggregator
            self.aggregator.receive_update(
                subject_id,
                payload['weights'],
                payload['n_samples']
            )

            client_metrics.append({
                'subject_id': subject_id,
                'local_train_acc': acc,
                'local_test_acc': local_acc,
                'local_train_loss': loss,
                'local_test_loss': local_loss,
                'n_samples': payload['n_samples']
            })

        # Aggregate
        aggregated_weights = self.aggregator.aggregate()

        if aggregated_weights is not None:
            # Update global model
            self.global_model.model.set_weights(aggregated_weights)

            # Evaluate global model on all clients
            global_accuracies = []
            for subject_id in self.subject_ids:
                data = self.client_data[subject_id]
                acc, loss = self.global_model.evaluate(
                    data['test_windows'],
                    data['test_labels']
                )
                global_accuracies.append(acc)

            round_time = time.time() - round_start

            metrics = {
                'round_num': round_num,
                'client_metrics': client_metrics,
                'global_mean_acc': np.mean(global_accuracies),
                'global_std_acc': np.std(global_accuracies),
                'global_accuracies': global_accuracies,
                'round_time': round_time
            }

            print(f"\n  Global Model:")
            print(f"    Mean accuracy: {metrics['global_mean_acc']:.4f} +/- {metrics['global_std_acc']:.4f}")
            for i, subj in enumerate(self.subject_ids):
                print(f"      Client {subj}: {global_accuracies[i]:.4f}")
            print(f"    Round time: {round_time:.2f}s")

            return metrics
        else:
            print("  ERROR: Insufficient clients for aggregation")
            return None


def evaluate_global_model(
    global_model: LSTMDriftDetector,
    client_data: Dict[int, Dict],
    subject_ids: List[int]
) -> Dict:
    """Evaluate global model on all client test sets."""
    print(f"\n{'='*60}")
    print("GLOBAL MODEL EVALUATION")
    print(f"{'='*60}")

    results = {}

    for subject_id in subject_ids:
        data = client_data[subject_id]
        acc, loss = global_model.evaluate(
            data['test_windows'],
            data['test_labels']
        )
        results[subject_id] = {'accuracy': acc, 'loss': loss}
        print(f"  Client {subject_id}: acc={acc:.4f}, loss={loss:.4f}")

    return results


def print_convergence_analysis(round_metrics: List[Dict]):
    """Print analysis of model convergence."""
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")

    print(f"{'Round':<8} {'Global Acc':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)

    for metrics in round_metrics:
        accs = metrics['global_accuracies']
        print(f"{metrics['round_num']+1:<8} "
              f"{metrics['global_mean_acc']:<12.4f} "
              f"{metrics['global_std_acc']:<10.4f} "
              f"{min(accs):<10.4f} "
              f"{max(accs):<10.4f}")

    # Improvement
    if len(round_metrics) >= 2:
        first_acc = round_metrics[0]['global_mean_acc']
        last_acc = round_metrics[-1]['global_mean_acc']
        improvement = last_acc - first_acc
        print(f"\nTotal improvement: {improvement:+.4f} ({improvement/first_acc*100:+.2f}%)")


def print_communication_summary(round_metrics: List[Dict], client_ids: List[int]):
    """Print communication efficiency summary."""
    print(f"\n{'='*60}")
    print("COMMUNICATION SUMMARY")
    print(f"{'='*60}")

    # Model size
    global_weights = round_metrics[-1]['client_metrics'][0] if round_metrics else {}
    if hasattr(global_model := None, 'model'):
        total_params, _ = global_model.count_parameters()
        model_size_kb = total_params * 4 / 1024
    else:
        # Estimate from LSTM config
        total_params = 64 * 64 * 4 + 64 * 18 + 64 * 9 * 4  # Rough estimate
        model_size_kb = total_params * 4 / 1024

    n_clients = len(client_ids)
    n_rounds = len(round_metrics)

    print(f"  Model size: {model_size_kb:.1f} KB (float32)")
    print(f"  Clients: {n_clients}")
    print(f"  Rounds: {n_rounds}")
    print(f"  Total uploads: {n_clients * n_rounds}")
    print(f"  Total data transferred: {n_clients * n_rounds * model_size_kb:.1f} KB")


def main():
    """Run Test 3: Federated Learning Global Model Accuracy."""
    print("="*60)
    print("TEST 3: FEDERATED LEARNING GLOBAL MODEL ACCURACY")
    print("="*60)

    # Select clients known to have good activity overlap
    # Based on PAMAP2 dataset, subjects 101-108 have similar activity profiles
    selected_subjects = [101, 102, 106, 107, 108]
    print(f"\nSelected subjects: {selected_subjects}")

    # Initialize simulation
    sim = FLSimulation(
        subject_ids=selected_subjects,
        n_rounds=10,
        local_epochs=3,
        batch_size=64
    )

    # Load all client data
    print(f"\n{'='*60}")
    print("LOADING CLIENT DATA")
    print(f"{'='*60}")

    for subject_id in selected_subjects:
        sim.load_client_data(subject_id)

    # Find common activities and prepare final datasets
    sim.find_common_activities_and_prepare_data(test_split=0.2)

    # Check if we have enough common activities
    if sim.n_classes < 3:
        print(f"\nERROR: Only {sim.n_classes} common activities found. Need at least 3.")
        print("Try different subject selection.")
        return None

    # Run FL rounds
    print(f"\n{'='*60}")
    print("RUNNING FEDERATED LEARNING")
    print(f"{'='*60}")

    for round_num in range(sim.n_rounds):
        metrics = sim.run_round(round_num)
        if metrics:
            sim.round_metrics.append(metrics)

    # Final evaluation
    final_results = evaluate_global_model(
        sim.global_model,
        sim.client_data,
        selected_subjects
    )

    # Print analyses
    if sim.round_metrics:
        print_convergence_analysis(sim.round_metrics)
        print_communication_summary(sim.round_metrics, selected_subjects)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    print(f"  Common activities: {sim.n_classes}")
    print(f"  Activity names: {sim.client_data[selected_subjects[0]]['label_names']}")

    if sim.round_metrics:
        final_metrics = sim.round_metrics[-1]
        print(f"  Global model accuracy: {final_metrics['global_mean_acc']:.4f} +/- {final_metrics['global_std_acc']:.4f}")
        print(f"  Best client accuracy: {max(final_metrics['global_accuracies']):.4f}")
        print(f"  Worst client accuracy: {min(final_metrics['global_accuracies']):.4f}")

        # Per-client final accuracy
        print(f"\n  Per-Client Final Accuracy:")
        for i, subj in enumerate(selected_subjects):
            acc = final_metrics['global_accuracies'][i]
            print(f"    Client {subj}: {acc:.4f}")

        # Model info
        total_params, _ = sim.global_model.model.count_parameters()
        memory_size = sim.global_model.model.get_memory_size()
        print(f"\n  Model parameters: {total_params:,}")
        print(f"  Model size (float32): {memory_size['float32']:.1f} KB")
        print(f"  Model size (int8): {memory_size['int8']:.1f} KB")

    print("\n" + "="*60)
    print("TEST 3 COMPLETE")
    print("="*60)

    return {
        'subjects': selected_subjects,
        'n_rounds': sim.n_rounds,
        'final_global_acc': final_metrics['global_mean_acc'] if sim.round_metrics else None,
        'round_metrics': sim.round_metrics
    }


if __name__ == "__main__":
    main()
