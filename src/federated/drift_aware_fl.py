"""
Dynamic Drift-Aware Federated Learning Simulation

Extends standard FL simulation with:
- Per-client, per-round dynamic drift injection
- Heterogeneous drift scenarios (different clients drift at different times)
- Comprehensive drift metrics tracking

Research Justification:
- Real-world FL: users experience drift independently (not synchronized)
- Different drift types affect clients differently
- Tests FL robustness under realistic heterogeneous conditions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time

# Import drift injection from data pipeline
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_pipeline.drift_injection import (
    DriftInjector, DriftConfig, DriftType, MultiDriftInjector
)
from drift_detection.lstm_model import LSTMConfig


@dataclass
class DynamicDriftConfig:
    """Configuration for dynamic drift injection in FL."""

    # Probability of drift per client per round (0.0 to 1.0)
    drift_probability: float = 0.3

    # Available drift types to sample from
    drift_types: List[DriftType] = field(default_factory=lambda: [
        DriftType.SUDDEN,
        DriftType.GRADUAL,
        DriftType.RECURRING,
        DriftType.FEATURE_LEVEL
    ])

    # Magnitude range to sample from
    magnitude_min: float = 0.5
    magnitude_max: float = 1.5

    # For feature-level drift: which axes to sample from
    # 0-2: wrist (x,y,z), 3-5: chest (x,y,z), 6-8: ankle (x,y,z)
    target_axes: List[int] = field(default_factory=lambda: [0, 3, 6])

    # For gradual drift: ramp duration (samples)
    ramp_samples_min: int = 100
    ramp_samples_max: int = 500

    # For recurring drift: period and duration (samples)
    period_min: int = 500
    period_max: int = 2000
    duration_min: int = 100
    duration_max: int = 500

    # Injection point within client data (fraction of data)
    # Drift starts at this fraction through the client's data
    injection_fraction_min: float = 0.2
    injection_fraction_max: float = 0.6

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class DriftEvent:
    """Records a drift event for a specific client in a specific round."""

    client_id: int
    round_num: int
    drift_type: DriftType
    magnitude: float
    target_axis: Optional[int]  # Only for feature-level drift
    injection_point: int  # Sample index where drift starts
    n_samples_affected: int  # How many samples have drift
    drift_config: DriftConfig  # Full config for reproducibility

    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'round_num': self.round_num,
            'drift_type': self.drift_type.value,
            'magnitude': self.magnitude,
            'target_axis': self.target_axis,
            'injection_point': self.injection_point,
            'n_samples_affected': self.n_samples_affected
        }


class DriftAwareFlSimulation:
    """
    Federated Learning simulation with dynamic per-client drift injection.

    Key Features:
    - Each client has independent probability of experiencing drift each round
    - Drift type, magnitude, and timing are randomized
    - Comprehensive tracking of drift events and their impact

    Usage:
        config = DynamicDriftConfig(drift_probability=0.3)
        sim = DriftAwareFlSimulation(config, features, labels, n_clients=5)
        results = sim.run(n_rounds=10)
    """

    def __init__(
        self,
        drift_config: DynamicDriftConfig,
        features: np.ndarray,
        labels: np.ndarray,
        n_clients: int = 5,
        lstm_config: Optional[LSTMConfig] = None,
        test_ratio: float = 0.2,
        seed: int = 42
    ):
        self.drift_config = drift_config
        self.features = features
        self.labels = labels
        self.n_clients = n_clients
        self.lstm_config = lstm_config or LSTMConfig()
        self.test_ratio = test_ratio
        self.seed = seed

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Partition data non-IID (by label distribution)
        self.client_data = self._partition_data_non_iid()

        # Drift tracking
        self.drift_events: List[DriftEvent] = []
        self.client_drift_history: Dict[int, List[int]] = {
            i: [] for i in range(n_clients)
        }  # client_id -> list of rounds with drift

        # Metrics
        self.round_metrics: List[Dict] = []

    def _partition_data_non_iid(
        self,
        test_ratio: float = None
    ) -> Dict[int, Dict]:
        """
        Partition data non-IID across clients.

        Each client gets a subset of labels preferentially,
        simulating real non-IID federated learning.

        Returns:
            Dict mapping client_id -> {train_features, train_labels, test_features, test_labels}
        """
        test_ratio = test_ratio or self.test_ratio
        n_samples = len(self.labels)
        unique_labels = np.unique(self.labels)
        n_labels = len(unique_labels)

        # Assign labels preferentially to clients
        # Each client gets 1-2 "home" labels with more samples
        labels_per_client = max(1, n_labels // self.n_clients)
        client_label_assignments = {}

        for client_id in range(self.n_clients):
            start_label = (client_id * labels_per_client) % n_labels
            end_label = min(start_label + labels_per_client + 1, n_labels)
            client_label_assignments[client_id] = unique_labels[start_label:end_label]

        # Distribute samples
        client_data = {}
        for client_id in range(self.n_clients):
            client_labels = client_label_assignments[client_id]

            # 70% of samples with client's labels go to this client
            # 30% are distributed to other clients (for some IID mixing)
            mask = np.isin(self.labels, client_labels)
            client_indices = np.where(mask)[0]

            # Shuffle and split
            n_client_samples = len(client_indices)
            shuffled = self.rng.permutation(client_indices)

            n_test = max(1, int(n_client_samples * test_ratio))
            test_indices = shuffled[:n_test]
            train_indices = shuffled[n_test:]

            client_data[client_id] = {
                'train_features': self.features[train_indices],
                'train_labels': self.labels[train_indices],
                'test_features': self.features[test_indices],
                'test_labels': self.labels[test_indices],
                'assigned_labels': client_labels
            }

        return client_data

    def sample_drift_for_client(
        self,
        client_id: int,
        round_num: int
    ) -> Optional[DriftConfig]:
        """
        Sample drift configuration for a specific client in a specific round.

        Args:
            client_id: Client identifier
            round_num: FL round number

        Returns:
            DriftConfig if drift should be injected, None otherwise
        """
        # Check if drift should occur (based on probability)
        if self.rng.random() >= self.drift_config.drift_probability:
            return None

        # Sample drift type
        drift_type = self.rng.choice(self.drift_config.drift_types)

        # Sample magnitude
        magnitude = float(self.rng.uniform(
            self.drift_config.magnitude_min,
            self.drift_config.magnitude_max
        ))

        # Sample injection point (as fraction of client data)
        injection_frac = float(self.rng.uniform(
            self.drift_config.injection_fraction_min,
            self.drift_config.injection_fraction_max
        ))

        # Create drift config based on type
        config = DriftConfig(
            drift_type=drift_type,
            magnitude=magnitude,
            injection_point=0,  # Will be set when applying
            seed=int(self.rng.integers(0, 10000))
        )

        # Type-specific parameters
        if drift_type == DriftType.GRADUAL:
            config.ramp_samples = int(self.rng.integers(
                self.drift_config.ramp_samples_min,
                self.drift_config.ramp_samples_max
            ))
        elif drift_type == DriftType.RECURRING:
            config.period = int(self.rng.integers(
                self.drift_config.period_min,
                self.drift_config.period_max
            ))
            config.duration = int(self.rng.integers(
                self.drift_config.duration_min,
                self.drift_config.duration_max
            ))
        elif drift_type == DriftType.FEATURE_LEVEL:
            config.target_axis = int(self.rng.choice(
                self.drift_config.target_axes
            ))

        # Store injection fraction for later
        config._injection_fraction = injection_frac

        return config

    def inject_drift_into_client_data(
        self,
        client_id: int,
        drift_config: DriftConfig
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Inject drift into client's training data.

        Args:
            client_id: Client identifier
            drift_config: Drift configuration

        Returns:
            Tuple of (drifted_features, drift_mask, injection_point)
            - drifted_features: Training features with drift applied
            - drift_mask: Boolean array indicating which samples have drift
            - injection_point: Sample index where drift starts
        """
        original_features = self.client_data[client_id]['train_features'].copy()

        # Calculate injection point
        n_samples = len(original_features)
        injection_point = int(n_samples * drift_config._injection_fraction)
        drift_config.injection_point = injection_point

        # Create injector and apply drift
        injector = DriftInjector(drift_config)
        drifted_features = injector.inject_drift(original_features)

        # Create drift mask
        drift_mask = injector.get_drift_mask(n_samples)
        n_affected = int(np.sum(drift_mask))

        return drifted_features, drift_mask, injection_point

    def run_round(
        self,
        round_num: int,
        local_epochs: int = 3
    ) -> Dict:
        """
        Run one FL round with dynamic drift injection.

        Args:
            round_num: Round number (0-indexed)
            local_epochs: Epochs per client

        Returns:
            Dict with round metrics
        """
        round_drift_events = []
        client_accuracies = {}
        client_losses = {}

        for client_id in range(self.n_clients):
            # Sample drift for this client
            drift_config = self.sample_drift_for_client(client_id, round_num)

            if drift_config is not None:
                # Inject drift
                drifted_features, drift_mask, inj_point = self.inject_drift_into_client_data(
                    client_id, drift_config
                )

                # Record drift event
                event = DriftEvent(
                    client_id=client_id,
                    round_num=round_num,
                    drift_type=drift_config.drift_type,
                    magnitude=drift_config.magnitude,
                    target_axis=drift_config.target_axis if drift_config.drift_type == DriftType.FEATURE_LEVEL else None,
                    injection_point=inj_point,
                    n_samples_affected=int(np.sum(drift_mask)),
                    drift_config=drift_config
                )
                self.drift_events.append(event)
                round_drift_events.append(event)
                self.client_drift_history[client_id].append(round_num)

                # Train on drifted data
                train_features = drifted_features
                has_drift = True
            else:
                # No drift - use original data
                train_features = self.client_data[client_id]['train_features']
                has_drift = False

            # Simple mock training (in real implementation, would train LSTM)
            # For now, just compute data statistics as proxy
            client_losses[client_id] = float(np.var(train_features))

            # Evaluate on test set (no drift on test)
            test_features = self.client_data[client_id]['test_features']
            test_labels = self.client_data[client_id]['test_labels']

            # Mock accuracy (in real implementation, would run inference)
            # Use label distribution similarity as proxy
            train_labels = self.client_data[client_id]['train_labels']
            unique_train = set(train_labels)
            unique_test = set(test_labels)
            overlap = len(unique_train & unique_test) / max(len(unique_test), 1)
            client_accuracies[client_id] = min(0.95, overlap * 0.9 + self.rng.random() * 0.1)

        return {
            'round_num': round_num,
            'drift_events': round_drift_events,
            'client_accuracies': client_accuracies,
            'client_losses': client_losses,
            'n_clients_with_drift': len(round_drift_events)
        }

    def run(
        self,
        n_rounds: int = 10,
        local_epochs: int = 3,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete FL simulation with dynamic drift.

        Args:
            n_rounds: Number of FL rounds
            local_epochs: Epochs per client per round
            verbose: Print progress

        Returns:
            Dict with all simulation results
        """
        if verbose:
            print("=" * 60)
            print("DYNAMIC DRIFT-AWARE FL SIMULATION")
            print("=" * 60)
            print(f"\nConfiguration:")
            print(f"  Clients: {self.n_clients}")
            print(f"  Rounds: {n_rounds}")
            print(f"  Drift probability: {self.drift_config.drift_probability}")
            print(f"  Drift types: {[dt.value for dt in self.drift_config.drift_types]}")
            print(f"\nData partition (non-IID):")
            for cid, data in self.client_data.items():
                print(f"  Client {cid}: {len(data['train_labels'])} train, "
                      f"{len(data['test_labels'])} test, "
                      f"labels: {data['assigned_labels']}")

        # Run rounds
        for round_num in range(n_rounds):
            metrics = self.run_round(round_num, local_epochs)
            self.round_metrics.append(metrics)

            if verbose:
                drift_info = ""
                if metrics['n_clients_with_drift'] > 0:
                    drift_types = [e.drift_type.value[:4] for e in metrics['drift_events']]
                    drift_info = f" | Drift: {drift_types}"

                avg_acc = np.mean(list(metrics['client_accuracies'].values()))
                print(f"  Round {round_num + 1}/{n_rounds}: "
                      f"avg_acc={avg_acc:.1%} | "
                      f"{metrics['n_clients_with_drift']}/{self.n_clients} clients drifting"
                      f"{drift_info}")

        # Compute summary statistics
        results = self._compute_summary(n_rounds)

        if verbose:
            print("\n" + "=" * 60)
            print("SIMULATION COMPLETE")
            print("=" * 60)
            print(f"\nTotal drift events: {len(self.drift_events)}")
            print(f"Drift distribution:")
            for dt in DriftType:
                count = sum(1 for e in self.drift_events if e.drift_type == dt)
                print(f"  {dt.value}: {count} events")

        return results

    def _compute_summary(self, n_rounds: int) -> Dict:
        """Compute summary statistics from simulation."""

        # Drift statistics
        drift_by_type = {}
        for dt in DriftType:
            drift_by_type[dt.value] = sum(
                1 for e in self.drift_events if e.drift_type == dt
            )

        # Client statistics
        client_stats = {}
        for cid in range(self.n_clients):
            n_drift_rounds = len(self.client_drift_history[cid])
            client_stats[cid] = {
                'total_drift_events': n_drift_rounds,
                'drift_rounds': self.client_drift_history[cid],
                'drift_fraction': n_drift_rounds / n_rounds
            }

        # Accuracy over rounds
        round_accuracies = [
            np.mean(list(m['client_accuracies'].values()))
            for m in self.round_metrics
        ]

        # Compare clients with vs without drift
        acc_with_drift = []
        acc_without_drift = []

        for m in self.round_metrics:
            drifted_clients = {e.client_id for e in m['drift_events']}
            for cid, acc in m['client_accuracies'].items():
                if cid in drifted_clients:
                    acc_with_drift.append(acc)
                else:
                    acc_without_drift.append(acc)

        return {
            'n_rounds': n_rounds,
            'n_clients': self.n_clients,
            'total_drift_events': len(self.drift_events),
            'drift_by_type': drift_by_type,
            'client_stats': client_stats,
            'round_accuracies': round_accuracies,
            'avg_accuracy': np.mean(round_accuracies),
            'final_accuracy': round_accuracies[-1] if round_accuracies else None,
            'accuracy_with_drift': np.mean(acc_with_drift) if acc_with_drift else None,
            'accuracy_without_drift': np.mean(acc_without_drift) if acc_without_drift else None,
            'drift_events': [e.to_dict() for e in self.drift_events],
            'round_metrics': self.round_metrics
        }


def run_dynamic_drift_fl(
    features: np.ndarray,
    labels: np.ndarray,
    n_clients: int = 5,
    n_rounds: int = 10,
    drift_probability: float = 0.3,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[DriftAwareFlSimulation, Dict]:
    """
    Convenience function to run dynamic drift FL simulation.

    Args:
        features: Input features (n_samples, 256, 9) for LSTM
        labels: Labels (n_samples,)
        n_clients: Number of virtual clients
        n_rounds: Number of FL rounds
        drift_probability: Probability of drift per client per round
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (simulation, results)
    """
    drift_config = DynamicDriftConfig(
        drift_probability=drift_probability,
        seed=seed
    )

    sim = DriftAwareFlSimulation(
        drift_config=drift_config,
        features=features,
        labels=labels,
        n_clients=n_clients,
        seed=seed
    )

    results = sim.run(n_rounds=n_rounds, verbose=verbose)

    return sim, results


if __name__ == "__main__":
    # Test with synthetic data
    print("Generating synthetic HAR data...")
    np.random.seed(42)

    n_samples = 1000
    n_classes = 4
    n_windows = 256

    features = []
    labels = []

    for class_id in range(n_classes):
        class_samples = n_samples // n_classes
        base_signal = np.sin(np.linspace(0, 2 * np.pi * (class_id + 1), n_windows))

        for _ in range(class_samples):
            window = np.zeros((n_windows, 9), dtype=np.float32)
            for axis in range(9):
                noise = np.random.randn(n_windows) * 0.1
                window[:, axis] = base_signal * (axis + 1) / 9 + noise
            features.append(window)
            labels.append(class_id)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"Generated {len(features)} samples, {n_classes} classes")
    print()

    # Run simulation
    sim, results = run_dynamic_drift_fl(
        features, labels,
        n_clients=5,
        n_rounds=10,
        drift_probability=0.4,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("DYNAMIC DRIFT FL TEST COMPLETE")
    print("=" * 60)
