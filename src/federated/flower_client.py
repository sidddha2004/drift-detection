"""
Flower-based Federated Learning Client

Integrates with the Flower framework for realistic FL simulation:
- Implements flwr.client.NumPyClient interface
- Wraps LSTM/MLP models for federated training
- Supports quantized weight transmission

Research Justification:
- Flower is the standard FL framework (used in 100+ research papers)
- Provides realistic network simulation, straggler handling
- Enables comparison with state-of-the-art FL research
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import flwr as fl
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from drift_detection.lstm_model import ActivityLSTM, LSTMDriftDetector, LSTMConfig


@dataclass
class FlowerClientConfig:
    """Configuration for Flower FL client."""
    # Model
    lstm_config: Optional[LSTMConfig] = None

    # Training
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001

    # Data partition (for non-IID simulation)
    client_id: int = 0
    num_clients: int = 10


class FlowerClient(fl.client.NumPyClient):
    """
    Flower Client wrapping LSTM model.

    Implements the flwr.client.NumPyClient interface:
    - get_parameters(): Return model weights
    - fit(): Train on local data, return updated weights
    - evaluate(): Evaluate on local test data

    Usage:
        client = FlowerClient(client_id=0, config=config)

        # Start training (usually via fl.client.start_client)
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=client.to_client()
        )
    """

    def __init__(
        self,
        client_id: int,
        config: FlowerClientConfig,
        train_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        super().__init__()
        self.client_id = client_id
        self.config = config

        # Create LSTM model
        self.lstm_config = config.lstm_config or LSTMConfig()
        self.model = ActivityLSTM(self.lstm_config)
        self.use_lstm = True

        # Training data (can be set after initialization)
        self.train_features: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None

        if train_data:
            self.set_train_data(*train_data)
        if test_data:
            self.set_test_data(*test_data)

        # Optimizer and criterion
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0001
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        # Statistics
        self.samples_processed = 0
        self.rounds_participated = 0

    def set_train_data(self, windows: np.ndarray, labels: np.ndarray):
        """Set local training data."""
        self.train_features = windows
        self.train_labels = labels

    def set_test_data(self, windows: np.ndarray, labels: np.ndarray):
        """Set local test data."""
        self.test_features = windows
        self.test_labels = labels

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model weights for federated averaging."""
        weights_dict = self.model.get_weights()

        # Convert to list of arrays (Flower format)
        weight_list = []
        for name, arr in weights_dict.items():
            weight_list.append(arr.astype(np.float32))

        return weight_list

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load model weights from server."""
        weights_dict = self.model.get_weights()
        keys = list(weights_dict.keys())

        if len(parameters) != len(keys):
            raise ValueError(
                f"Parameter count mismatch: got {len(parameters)}, expected {len(keys)}"
            )

        # Map parameters back to weight dict
        for i, key in enumerate(keys):
            weights_dict[key] = parameters[i].astype(np.float32)

        self.model.set_weights(weights_dict)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train on local data and return updated weights.

        Args:
            parameters: Global model weights from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_weights, num_samples, metrics)
        """
        # Load global weights
        self.set_parameters(parameters)

        # Get training config
        epochs = int(config.get("epochs", self.config.local_epochs))
        batch_size = int(config.get("batch_size", self.config.batch_size))

        if self.train_features is None:
            raise ValueError("No training data available")

        # Train
        self.model.train()
        n_samples = len(self.train_features)
        total_loss = 0.0
        n_batches = 0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_epoch_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                features = self.train_features[batch_idx]
                labels = self.train_labels[batch_idx]

                # Forward pass
                if self.use_lstm:
                    x_tensor = torch.from_numpy(features).float()
                else:
                    x_tensor = torch.from_numpy(features).float()

                y_tensor = torch.from_numpy(labels).long()

                self.optimizer.zero_grad()
                outputs = self.model(x_tensor)
                loss = self.criterion(outputs, y_tensor)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_epoch_batches += 1

            total_loss += epoch_loss / n_epoch_batches

        self.samples_processed += n_samples * epochs
        self.rounds_participated += 1

        # Get updated weights
        updated_weights = self.get_parameters({})

        # Return metrics
        metrics = {
            "client_id": self.client_id,
            "loss": float(total_loss / epochs),
            "samples_processed": self.samples_processed,
            "rounds_participated": self.rounds_participated
        }

        return updated_weights, n_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data.

        Args:
            parameters: Model weights to evaluate
            config: Configuration from server

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Load weights
        self.set_parameters(parameters)

        if self.test_features is None:
            raise ValueError("No test data available")

        self.model.eval()

        with torch.no_grad():
            x_tensor = torch.from_numpy(self.test_features).float()
            y_tensor = torch.from_numpy(self.test_labels).long()

            outputs = self.model(x_tensor)
            loss = self.criterion(outputs, y_tensor)

            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            accuracy = (predictions == y_tensor).float().mean().item()

        metrics = {
            "client_id": self.client_id,
            "accuracy": accuracy,
            "rounds_participated": self.rounds_participated
        }

        return float(loss), len(self.test_features), metrics

    def to_client(self) -> fl.client.Client:
        """Convert to Flower client for start_simulation."""
        # Return self as NumPyClient - Flower will wrap it
        return self

    def get_statistics(self) -> Dict:
        """Get client statistics."""
        return {
            "client_id": self.client_id,
            "use_lstm": self.use_lstm,
            "samples_processed": self.samples_processed,
            "rounds_participated": self.rounds_participated
        }


class NonIIDDataPartitioner:
    """
    Partition data across clients in non-IID fashion.

    Simulates realistic FL scenario where each client has:
    - Different activity distribution (label skew)
    - Different number of samples (quantity skew)

    Research Justification:
    - Real HAR data is inherently non-IID (users have different habits)
    - Label skew: Some users walk more, others sit more
    - Quantity skew: Different usage patterns
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_clients: int,
        partition_type: str = "label_skew",
        alpha: float = 0.5  # Dirichlet concentration parameter
    ):
        self.features = features
        self.labels = labels
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.alpha = alpha

        self.client_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def partition(self, test_ratio: float = 0.2) -> Dict[int, Dict]:
        """
        Partition data across clients.

        Args:
            test_ratio: Fraction for test data

        Returns:
            Dict mapping client_id to {train_features, train_labels, test_features, test_labels}
        """
        if self.partition_type == "label_skew":
            return self._partition_label_skew(test_ratio)
        elif self.partition_type == "quantity_skew":
            return self._partition_quantity_skew(test_ratio)
        elif self.partition_type == "iid":
            return self._partition_iid(test_ratio)
        else:
            raise ValueError(f"Unknown partition type: {self.partition_type}")

    def _partition_label_skew(self, test_ratio: float) -> Dict[int, Dict]:
        """Label skew: each client gets subset of classes."""
        # Assign primary classes to each client using Dirichlet
        unique_labels = np.unique(self.labels)
        n_classes = len(unique_labels)

        # Dirichlet sampling for class distribution
        label_dist = np.random.dirichlet([self.alpha] * self.num_clients, size=n_classes)

        # Assign samples to clients
        client_train_features = [[] for _ in range(self.num_clients)]
        client_train_labels = [[] for _ in range(self.num_clients)]

        for class_idx, label in enumerate(unique_labels):
            class_mask = self.labels == label
            class_features = self.features[class_mask]
            class_indices = np.where(class_mask)[0]

            # Split according to Dirichlet distribution
            proportions = label_dist[class_idx]
            n_samples = len(class_indices)

            # Shuffle and split
            shuffled = np.random.permutation(n_samples)
            split_points = (np.cumsum(proportions) * n_samples).astype(int)

            for client_id in range(self.num_clients):
                start = 0 if client_id == 0 else split_points[client_id - 1]
                end = split_points[client_id] if client_id < self.num_clients - 1 else n_samples

                indices = class_indices[shuffled[start:end]]
                client_train_features[client_id].append(self.features[indices])
                client_train_labels[client_id].append(self.labels[indices])

        # Concatenate and create train/test splits
        result = {}
        for client_id in range(self.num_clients):
            train_features = np.concatenate(client_train_features[client_id])
            train_labels = np.concatenate(client_train_labels[client_id])

            # Test split (use global distribution for fair evaluation)
            test_mask = np.random.rand(len(train_features)) < test_ratio
            test_features = train_features[test_mask]
            test_labels = train_labels[test_mask]
            train_features = train_features[~test_mask]
            train_labels = train_labels[~test_mask]

            result[client_id] = {
                "train_features": train_features,
                "train_labels": train_labels,
                "test_features": test_features,
                "test_labels": test_labels,
                "label_distribution": dict(zip(*np.unique(train_labels, return_counts=True)))
            }

        return result

    def _partition_quantity_skew(self, test_ratio: float) -> Dict[int, Dict]:
        """Quantity skew: clients have different amounts of data."""
        n_samples = len(self.features)

        # Create quantity distribution (some clients have 10x more data)
        quantities = np.random.power(2, self.num_clients)
        quantities = (quantities / quantities.sum() * n_samples).astype(int)

        # Shuffle and split
        indices = np.random.permutation(n_samples)
        result = {}

        start = 0
        for client_id in range(self.num_clients):
            end = start + quantities[client_id]
            client_indices = indices[start:end]

            client_features = self.features[client_indices]
            client_labels = self.labels[client_indices]

            # Train/test split
            test_mask = np.random.rand(len(client_features)) < test_ratio
            test_features = client_features[test_mask]
            test_labels = client_labels[test_mask]
            train_features = client_features[~test_mask]
            train_labels = client_labels[~test_mask]

            result[client_id] = {
                "train_features": train_features,
                "train_labels": train_labels,
                "test_features": test_features,
                "test_labels": test_labels,
                "n_train_samples": len(train_features),
                "label_distribution": dict(zip(*np.unique(train_labels, return_counts=True)))
            }

            start = end

        return result

    def _partition_iid(self, test_ratio: float) -> Dict[int, Dict]:
        """IID partition: equal distribution across clients."""
        n_samples = len(self.features)
        samples_per_client = n_samples // self.num_clients

        # Shuffle and split equally
        indices = np.random.permutation(n_samples)
        result = {}

        for client_id in range(self.num_clients):
            start = client_id * samples_per_client
            end = start + samples_per_client if client_id < self.num_clients - 1 else n_samples

            client_indices = indices[start:end]
            client_features = self.features[client_indices]
            client_labels = self.labels[client_indices]

            # Train/test split
            test_mask = np.random.rand(len(client_features)) < test_ratio
            test_features = client_features[test_mask]
            test_labels = client_labels[test_mask]
            train_features = client_features[~test_mask]
            train_labels = client_labels[~test_mask]

            result[client_id] = {
                "train_features": train_features,
                "train_labels": train_labels,
                "test_features": test_features,
                "test_labels": test_labels,
                "n_train_samples": len(train_features),
                "label_distribution": dict(zip(*np.unique(train_labels, return_counts=True)))
            }

        return result
