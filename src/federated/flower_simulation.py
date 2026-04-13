"""
Flower-based Federated Learning Simulation

Provides realistic FL simulation with:
- Multiple virtual clients training in parallel
- Server-side aggregation (FedAvg)
- Non-IID data distribution
- Comprehensive metrics tracking

Research Justification:
- Simulates real-world FL deployment without physical devices
- Enables controlled experiments with non-IID data
- Standard evaluation framework for FL research papers

Usage:
    python -m src.federated.flower_simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import flwr as fl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from federated.flower_client import FlowerClient, FlowerClientConfig, NonIIDDataPartitioner
from drift_detection.lstm_model import LSTMConfig


@dataclass
class FlowerSimulationConfig:
    """Configuration for Flower FL simulation."""
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 64
    lstm_config: Optional[LSTMConfig] = None
    partition_type: str = "label_skew"
    dirichlet_alpha: float = 0.5
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2


class FlowerSimulation:
    """Flower-based FL Simulation."""

    def __init__(
        self,
        config: FlowerSimulationConfig,
        features: np.ndarray,
        labels: np.ndarray,
        output_dir: Optional[str] = None
    ):
        self.config = config
        self.features = features
        self.labels = labels
        self.output_dir = output_dir
        self.round_accuracies = []

        if config.lstm_config is None:
            self.config.lstm_config = LSTMConfig(
                hidden_size=64,
                num_layers=2,
                learning_rate=0.001,
                batch_size=config.batch_size
            )

        self.partitioner = NonIIDDataPartitioner(
            features, labels,
            num_clients=config.num_clients,
            partition_type=config.partition_type,
            alpha=config.dirichlet_alpha
        )
        self.client_data = self.partitioner.partition(test_ratio=0.2)

    def create_clients(self) -> List[FlowerClient]:
        """Create Flower clients with partitioned data."""
        clients = []
        for client_id, data in self.client_data.items():
            client_config = FlowerClientConfig(
                client_id=client_id,
                num_clients=self.config.num_clients,
                lstm_config=self.config.lstm_config,
                local_epochs=self.config.local_epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.lstm_config.learning_rate
            )

            client = FlowerClient(
                client_id=client_id,
                config=client_config,
                train_data=(data["train_features"], data["train_labels"]),
                test_data=(data["test_features"], data["test_labels"])
            )
            clients.append(client)

        return clients

    def get_parameters(self, client: FlowerClient) -> List[np.ndarray]:
        """Get parameters from a client."""
        return client.get_parameters({})

    def set_parameters(self, client: FlowerClient, parameters: List[np.ndarray]):
        """Set parameters on a client."""
        client.set_parameters(parameters)

    def aggregate(self, parameters_list: List[List[np.ndarray]], weights: List[int]) -> List[np.ndarray]:
        """Aggregate model parameters using weighted average."""
        aggregated = []
        total_weight = sum(weights)

        for param_idx in range(len(parameters_list[0])):
            # Stack parameters from all clients
            stacked = np.stack([params[param_idx] * w for params, w in zip(parameters_list, weights)], axis=0)
            # Weighted average
            aggregated.append(np.sum(stacked, axis=0) / total_weight)

        return aggregated

    def run(self) -> Dict:
        """Run the FL simulation."""
        print("=" * 60)
        print("FLOWER FEDERATED LEARNING SIMULATION")
        print("=" * 60)

        clients = self.create_clients()
        print(f"\nCreated {len(clients)} clients")

        print(f"\nData partition type: {self.config.partition_type}")
        for client_id, data in self.client_data.items():
            n_train = len(data["train_labels"])
            n_test = len(data["test_labels"])
            n_classes = len(data["label_distribution"])
            print(f"  Client {client_id}: {n_train} train, {n_test} test, {n_classes} classes")

        print(f"\nStarting FL training for {self.config.num_rounds} rounds...")
        print("=" * 60)

        # Initialize global model (average of initial client models)
        initial_params = [self.get_parameters(c) for c in clients]
        global_params = self.aggregate(initial_params, [1] * len(clients))

        # FL rounds
        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n--- Round {round_num} ---")

            # Select clients for this round
            n_fit = max(self.config.min_fit_clients, int(len(clients) * self.config.fraction_fit))
            fit_indices = np.random.choice(len(clients), n_fit, replace=False)
            fit_clients = [clients[i] for i in fit_indices]

            # Set global parameters on selected clients
            for client in fit_clients:
                self.set_parameters(client, global_params)

            # Train clients
            print(f"  Training {len(fit_clients)} clients...")
            fit_results = []
            for i, client in enumerate(fit_clients):
                updated_params, num_samples, metrics = client.fit(
                    global_params,
                    {"epochs": str(self.config.local_epochs), "batch_size": str(self.config.batch_size)}
                )
                fit_results.append((updated_params, num_samples, metrics))
                print(f"    Client {fit_indices[i]}: loss={metrics.get('loss', 'N/A'):.4f}")

            # Aggregate updated parameters
            params_list = [r[0] for r in fit_results]
            sample_counts = [r[1] for r in fit_results]
            global_params = self.aggregate(params_list, sample_counts)

            # Evaluate on all clients
            print(f"  Evaluating...")
            round_accuracies = []
            for i, client in enumerate(clients):
                self.set_parameters(client, global_params)
                loss, num_samples, eval_metrics = client.evaluate(
                    global_params,
                    {"round": str(round_num)}
                )
                acc = eval_metrics.get('accuracy', 0.0)
                round_accuracies.append(acc)
                print(f"    Client {i}: accuracy={acc:.1%}")

            avg_acc = np.mean(round_accuracies)
            std_acc = np.std(round_accuracies)
            self.round_accuracies.append(avg_acc)
            print(f"  Round {round_num} - Avg Accuracy: {avg_acc:.1%} (+/- {std_acc:.1%})")

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"\nFinal accuracy: {self.round_accuracies[-1]:.1%}")
        print(f"Best accuracy: {max(self.round_accuracies):.1%}")
        print(f"Improvement: {self.round_accuracies[-1] - self.round_accuracies[0]:.1%}")

        return {
            "round_accuracies": self.round_accuracies,
            "final_accuracy": self.round_accuracies[-1],
            "best_accuracy": max(self.round_accuracies)
        }


def run_flower_simulation(
    features: np.ndarray,
    labels: np.ndarray,
    num_clients: int = 5,
    num_rounds: int = 10,
    partition_type: str = "label_skew",
    use_lstm: bool = True
) -> Tuple[FlowerSimulation, Dict]:
    """Convenience function to run FL simulation."""
    config = FlowerSimulationConfig(
        num_clients=num_clients,
        num_rounds=num_rounds,
        partition_type=partition_type,
        use_lstm=use_lstm,
        lstm_config=LSTMConfig(
            hidden_size=64,
            num_layers=2,
            learning_rate=0.001,
            batch_size=64
        )
    )

    sim = FlowerSimulation(config, features, labels)
    results = sim.run()

    return sim, results


if __name__ == "__main__":
    print("Generating synthetic HAR data...")
    np.random.seed(42)

    n_samples = 2000
    n_classes = 4
    n_windows = 256

    # Generate data with class-specific patterns
    features = []
    labels = []

    for class_id in range(n_classes):
        class_samples = n_samples // n_classes
        base_signal = np.sin(np.linspace(0, 2 * np.pi * (class_id + 1), n_windows))

        for _ in range(class_samples):
            window = np.zeros((n_windows, 9))
            for axis in range(9):
                noise = np.random.randn(n_windows) * 0.1
                window[:, axis] = base_signal * (axis + 1) / 9 + noise
            features.append(window)
            labels.append(class_id)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"Generated {len(features)} samples, {n_classes} classes")

    # Run simulation
    sim, results = run_flower_simulation(
        features, labels,
        num_clients=5,
        num_rounds=5,
        partition_type="iid",
        use_lstm=True
    )

    print("\n" + "=" * 60)
    print("FLOWER SIMULATION TEST COMPLETE")
    print("=" * 60)
