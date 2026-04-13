"""
Phase 5: Federated Learning - Server

FL Server coordinates federated learning rounds:
- Collect model updates from clients
- Aggregate using FedAvg
- Broadcast global model back to clients

Research Justification:
- Centralized coordination simplifies synchronization
- Aggregation server is stateless (can be restarted)
- Supports both synchronous and asynchronous updates
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from federated.aggregator import FedAvgAggregator, AggregationConfig, QuantizationHelper
from federated.fl_client import FLClient, FLClientConfig
from drift_detection.lstm_model import LSTMConfig

# Import integrated detector (optional - for integrated FL experiments)
try:
    from drift_detection.integrated_detector import (
        FederatedIntegratedDriftDetector,
        FederatedIntegratedDriftDetectorConfig
    )
    INTEGRATED_DETECTOR_AVAILABLE = True
except ImportError:
    INTEGRATED_DETECTOR_AVAILABLE = False
    FederatedIntegratedDriftDetector = None
    FederatedIntegratedDriftDetectorConfig = None


@dataclass
class FLServerConfig:
    """
    Configuration for FL server.

    Research Parameters:
    - min_clients: Minimum clients required for aggregation
      More clients = better model diversity but slower rounds
    - sync_interval_adaptations: Trigger sync every N local adaptations
      Balances communication cost vs. model freshness
    - sync_interval_minutes: Maximum time between syncs (fail-safe)
      Prevents stale global models
    - timeout_seconds: Wait time for client responses
      Handles stragglers in networked deployment
    """
    # Aggregation
    min_clients: int = 3

    # Sync triggers
    sync_interval_adaptations: int = 5
    sync_interval_minutes: int = 10

    # Network timeouts (for real deployment)
    timeout_seconds: int = 300  # 5 minutes

    # Simulation mode
    single_process: bool = True

    # Logging
    log_aggregation_stats: bool = True


@dataclass
class ClientState:
    """Tracks state of a connected client."""
    client_id: int
    last_update_time: float = 0.0
    updates_sent: int = 0
    last_accuracy: float = 0.0
    is_active: bool = True


class FLServer:
    """
    Federated Learning Server.

    Coordinates FL rounds:
    1. Collect updates from clients
    2. Aggregate using FedAvg
    3. Broadcast global model

    Supports two modes:
    - Single-process simulation (for research/development)
    - Networked deployment (for real edge devices)

    Usage (Simulation Mode):
        server = FLServer(config)

        # Register simulated clients
        server.register_client(client1)
        server.register_client(client2)

        # Run FL round
        global_weights = server.run_round()

    Usage (Networked Mode):
        server = FLServer(config)

        # HTTP/gRPC endpoints would call:
        server.receive_update(client_id, weights, n_samples)
        server.aggregate()
        server.broadcast()
    """

    def __init__(
        self,
        config: Optional[FLServerConfig] = None,
        aggregation_config: Optional[AggregationConfig] = None
    ):
        self.config = config or FLServerConfig()
        self.aggregation_config = aggregation_config or AggregationConfig()

        # Core aggregator
        self.aggregator = FedAvgAggregator(self.aggregation_config)

        # Client management
        self.clients: Dict[int, FLClient] = {}
        self.client_states: Dict[int, ClientState] = {}

        # Global model
        self.global_weights: Optional[Dict[str, np.ndarray]] = None
        self.global_model_version: int = 0

        # FL round tracking
        self.completed_rounds: int = 0
        self.last_sync_time: float = time.time()
        self.adaptations_since_sync: int = 0

        # Statistics
        self.round_history: List[Dict] = []
        self.quantizer = QuantizationHelper()

    def register_client(self, client: FLClient):
        """
        Register a client for federated learning.

        Args:
            client: FLClient instance
        """
        self.clients[client.client_id] = client
        self.client_states[client.client_id] = ClientState(client_id=client.client_id)

    def register_detector(self, detector: 'FederatedIntegratedDriftDetector'):
        """
        Register a FederatedIntegratedDriftDetector for federated learning.

        This method enables detectors to participate in FL rounds.
        The detector's internal FL client is used for weight exchange.

        Args:
            detector: FederatedIntegratedDriftDetector instance
        """
        if not INTEGRATED_DETECTOR_AVAILABLE:
            raise RuntimeError(
                "FederatedIntegratedDriftDetector not available. "
                "Ensure drift_detection module is properly configured."
            )

        self.clients[detector.config.client_id] = detector.fl_client
        self.client_states[detector.config.client_id] = ClientState(
            client_id=detector.config.client_id
        )

        # Set server reference in detector for sync operations
        detector.set_fl_server(self)

    def unregister_client(self, client_id: int):
        """Remove a client."""
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.client_states:
            del self.client_states[client_id]

    def receive_update(
        self,
        client_id: int,
        weights: Dict[str, np.ndarray],
        n_samples: int,
        accuracy: float = 0.0
    ):
        """
        Receive model update from client.

        For networked deployment, this is called by HTTP/gRPC endpoint.

        Args:
            client_id: Client identifier
            weights: Model weights
            n_samples: Number of samples client trained on
            accuracy: Client's local accuracy
        """
        self.aggregator.receive_update(client_id, weights, n_samples)

        # Update client state
        if client_id in self.client_states:
            state = self.client_states[client_id]
            state.last_update_time = time.time()
            state.updates_sent += 1
            state.last_accuracy = accuracy

        self.adaptations_since_sync += 1

    def aggregate(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Aggregate pending updates.

        Returns:
            Aggregated global weights, or None if insufficient clients
        """
        aggregated = self.aggregator.aggregate()

        if aggregated is not None:
            self.global_weights = aggregated
            self.global_model_version += 1
            self.completed_rounds += 1
            self.last_sync_time = time.time()
            self.adaptations_since_sync = 0

            # Log stats
            if self.config.log_aggregation_stats:
                self._log_round_stats()

        return aggregated

    def broadcast(self, detectors: Optional[List['FederatedIntegratedDriftDetector']] = None) -> Dict[int, bool]:
        """
        Broadcast global model to all registered clients.

        For single-process simulation, directly loads weights into clients.
        For networked deployment, sends via HTTP/gRPC.

        Args:
            detectors: Optional list of FederatedIntegratedDriftDetector instances.
                       If provided, also updates detector's internal model.

        Returns:
            Dict of client_id → success status
        """
        if self.global_weights is None:
            return {cid: False for cid in self.clients}

        results = {}

        for client_id, client in self.clients.items():
            try:
                if self.config.single_process:
                    # Direct load (simulation)
                    client.load_global_model(self.global_weights)
                    results[client_id] = True
                else:
                    # Networked: would send via HTTP/gRPC
                    # Placeholder for network code
                    results[client_id] = False
            except Exception as e:
                results[client_id] = False

        # Also update detectors if provided
        if detectors:
            for detector in detectors:
                try:
                    detector.load_model_weights(self.global_weights)
                    results[detector.config.client_id] = True
                except Exception as e:
                    results[detector.config.client_id] = False

        return results

    def run_round(
        self,
        detectors: Optional[List['FederatedIntegratedDriftDetector']] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Run a complete FL round: collect → aggregate → broadcast.

        For single-process simulation with registered clients.

        Args:
            detectors: Optional list of FederatedIntegratedDriftDetector instances.
                       If provided, also updates detectors after broadcast.

        Returns:
            Global model weights after aggregation
        """
        if not self.clients:
            return None

        # Collect updates from all active clients
        for client_id, client in self.clients.items():
            state = self.client_states.get(client_id)
            if state and not state.is_active:
                continue

            # Get client's current weights
            weights, scale_params = client.get_compressed_weights()

            # Evaluate client accuracy
            # In real scenario, client would report this
            accuracy = 0.0  # Placeholder

            self.receive_update(client_id, weights, client.local_sample_count, accuracy)

        # Aggregate
        aggregated = self.aggregate()

        if aggregated is not None:
            # Broadcast to all clients
            if detectors:
                self.broadcast(detectors)
            else:
                self.broadcast()

        return aggregated

    def should_trigger_sync(self) -> bool:
        """
        Check if sync should be triggered.

        Returns:
            True if sync conditions met
        """
        # Check adaptations threshold
        if self.adaptations_since_sync >= self.config.sync_interval_adaptations:
            return True

        # Check time threshold
        elapsed_minutes = (time.time() - self.last_sync_time) / 60
        if elapsed_minutes >= self.config.sync_interval_minutes:
            return True

        return False

    def _log_round_stats(self):
        """Log statistics for completed round."""
        stats = {
            'round': self.completed_rounds,
            'timestamp': time.time(),
            'participating_clients': len(self.client_states),
            'model_version': self.global_model_version
        }
        self.round_history.append(stats)

    def get_server_state(self) -> Dict:
        """Get current server state."""
        return {
            'global_model_version': self.global_model_version,
            'completed_rounds': self.completed_rounds,
            'registered_clients': len(self.clients),
            'active_clients': sum(1 for s in self.client_states.values() if s.is_active),
            'adaptations_since_sync': self.adaptations_since_sync,
            'last_sync_time': self.last_sync_time,
            'aggregation_state': self.aggregator.get_aggregation_state()
        }

    def get_client_statistics(self) -> Dict[int, Dict]:
        """Get statistics for all clients."""
        return {cid: client.get_statistics() for cid, client in self.clients.items()}


class FLSimulator:
    """
    Simulator for federated learning experiments.

    Creates multiple FL clients from the same data source,
    simulates non-IID data distribution, and runs FL rounds.

    Usage:
        simulator = FLSimulator(n_clients=9)

        # Distribute data non-IID (each client gets different subjects)
        simulator.distribute_data_non_iid(all_features, all_labels, subject_ids)

        # Run FL experiment
        for round in range(n_rounds):
            # Each client trains locally
            for client in simulator.clients:
                client.train_local(client_features, client_labels)

            # Server aggregates
            simulator.server.run_round()
    """

    def __init__(
        self,
        n_clients: int = 9,
        server_config: Optional[FLServerConfig] = None,
        client_config: Optional[FLClientConfig] = None,
        lstm_config: Optional[LSTMConfig] = None
    ):
        self.n_clients = n_clients
        self.server = FLServer(server_config)
        self.client_config = client_config or FLClientConfig()
        self.lstm_config = lstm_config or LSTMConfig()

        # Create clients
        self.clients: List[FLClient] = []
        for i in range(n_clients):
            client = FLClient(
                client_id=i + 1,
                config=self.client_config,
                lstm_config=self.lstm_config
            )
            self.clients.append(client)
            self.server.register_client(client)

        # Data distribution tracking
        self.client_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def distribute_data_iid(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ):
        """
        Distribute data IID across clients (random split).

        Args:
            features: All features (n_samples, 45)
            labels: All labels (n_samples,)
        """
        n_samples = len(features)
        indices = np.random.permutation(n_samples)

        chunk_size = n_samples // self.n_clients

        for i, client in enumerate(self.clients):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.n_clients - 1 else n_samples

            client_indices = indices[start_idx:end_idx]
            client_features = features[client_indices]
            client_labels = labels[client_indices]

            self.client_data[client.client_id] = (client_features, client_labels)

    def distribute_data_non_iid(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray
    ):
        """
        Distribute data non-IID (by subject).

        Each client gets data from different subjects,
        simulating real federated learning scenario.

        Args:
            features: All features
            labels: All labels
            subject_ids: Subject ID for each sample
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        subjects_per_client = n_subjects // self.n_clients

        for i, client in enumerate(self.clients):
            start_idx = i * subjects_per_client
            end_idx = start_idx + subjects_per_client if i < self.n_clients - 1 else n_subjects

            client_subjects = unique_subjects[start_idx:end_idx]
            mask = np.isin(subject_ids, client_subjects)

            client_features = features[mask]
            client_labels = labels[mask]

            self.client_data[client.client_id] = (client_features, client_labels)

    def run_experiment(
        self,
        n_rounds: int = 10,
        local_epochs: int = 5,
        eval_features: Optional[np.ndarray] = None,
        eval_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run FL experiment for multiple rounds.

        Args:
            n_rounds: Number of FL rounds
            local_epochs: Epochs per client per round
            eval_features: Holdout evaluation features (optional)
            eval_labels: Holdout evaluation labels (optional)

        Returns:
            Experiment results (accuracies per round)
        """
        results = {
            'round': [],
            'global_accuracy': [],
            'client_accuracies': []
        }

        for round_num in range(n_rounds):
            # Each client trains locally
            for client in self.clients:
                if client.client_id in self.client_data:
                    features, labels = self.client_data[client.client_id]
                    client.train_local(features, labels, epochs=local_epochs)

            # Server runs round
            self.server.run_round()

            # Evaluate
            if eval_features is not None and eval_labels is not None:
                # Evaluate global model on holdout set
                # Use first client as proxy for global model
                acc, _ = self.clients[0].evaluate(eval_features, eval_labels)
                results['global_accuracy'].append(acc)
            else:
                # Track average client accuracy
                accuracies = []
                for client in self.clients:
                    if client.client_id in self.client_data:
                        features, labels = self.client_data[client.client_id]
                        acc, _ = client.evaluate(features, labels)
                        accuracies.append(acc)
                results['global_accuracy'].append(np.mean(accuracies))

            results['round'].append(round_num + 1)
            results['client_accuracies'].append(
                {c.client_id: c.get_statistics() for c in self.clients}
            )

        return results


class FederatedIntegratedSimulator:
    """
    Simulator for federated learning experiments with drift detection + adaptation.

    Extends FLSimulator to support the full pipeline:
        1. Data streaming → windowing → features
        2. Local drift detection + classification + adaptation
        3. FL sync after N adaptations or T minutes
        4. Global model aggregation and broadcast

    Research Usage:
        simulator = FederatedIntegratedSimulator(
            n_clients=4,
            detector_config=FederatedIntegratedDriftDetectorConfig(
                sync_interval_adaptations=5,
                quantization_bits=16
            )
        )

    Metrics Tracked:
        - Per-round global accuracy
        - Local drift detection rate
        - Adaptation success rate
        - Communication rounds (FL syncs)
    """

    def __init__(
        self,
        n_clients: int = 4,
        server_config: Optional[FLServerConfig] = None,
        detector_config: Optional[FederatedIntegratedDriftDetectorConfig] = None,
        lstm_config: Optional[LSTMConfig] = None
    ):
        if not INTEGRATED_DETECTOR_AVAILABLE:
            raise RuntimeError(
                "FederatedIntegratedDriftDetector not available."
            )

        self.n_clients = n_clients
        self.server = FLServer(server_config)
        self.lstm_config = lstm_config or LSTMConfig()

        # Create detector config with lstm_config properly set
        base_detector_config = detector_config or FederatedIntegratedDriftDetectorConfig()
        self.detector_config = FederatedIntegratedDriftDetectorConfig(
            lstm_config=self.lstm_config,
            sync_interval_adaptations=base_detector_config.sync_interval_adaptations,
            sync_interval_minutes=base_detector_config.sync_interval_minutes,
            quantization_bits=base_detector_config.quantization_bits,
            adwin_delta=base_detector_config.adwin_delta,
            min_samples_for_adaptation=base_detector_config.min_samples_for_adaptation
        )

        self.detectors: List[FederatedIntegratedDriftDetector] = []
        for i in range(n_clients):
            client_config = FederatedIntegratedDriftDetectorConfig(
                lstm_config=self.lstm_config,
                client_id=i + 1,
                sync_interval_adaptations=self.detector_config.sync_interval_adaptations,
                sync_interval_minutes=self.detector_config.sync_interval_minutes,
                quantization_bits=self.detector_config.quantization_bits
            )
            detector = FederatedIntegratedDriftDetector(config=client_config)
            self.detectors.append(detector)
            self.server.register_detector(detector)

        self.client_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def distribute_data_iid(self, features: np.ndarray, labels: np.ndarray):
        """Distribute data IID across clients (random split)."""
        n_samples = len(features)
        indices = np.random.permutation(n_samples)
        chunk_size = n_samples // self.n_clients

        for i, detector in enumerate(self.detectors):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.n_clients - 1 else n_samples
            client_indices = indices[start_idx:end_idx]
            client_features = features[client_indices]
            client_labels = labels[client_indices]
            self.client_data[detector.config.client_id] = (client_features, client_labels)

    def distribute_data_non_iid(
        self, features: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray
    ):
        """Distribute data non-IID (by subject)."""
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        subjects_per_client = n_subjects // self.n_clients

        for i, detector in enumerate(self.detectors):
            start_idx = i * subjects_per_client
            end_idx = start_idx + subjects_per_client if i < self.n_clients - 1 else n_subjects
            client_subjects = unique_subjects[start_idx:end_idx]
            mask = np.isin(subject_ids, client_subjects)
            client_features = features[mask]
            client_labels = labels[mask]
            self.client_data[detector.config.client_id] = (client_features, client_labels)

    def run_integrated_experiment(
        self, n_stream_samples: int = 500, eval_features: Optional[np.ndarray] = None,
        eval_labels: Optional[np.ndarray] = None, seed: Optional[int] = None
    ) -> Dict:
        """Run integrated FL experiment with drift detection + adaptation."""
        if seed is not None:
            np.random.seed(seed)

        results = {
            'round': [], 'global_accuracy': [], 'drift_events': [],
            'sync_events': [], 'adaptation_stats': [], 'client_stats': []
        }

        sample_indices = {
            cid: np.random.choice(len(data[0]), min(n_stream_samples, len(data[0])), replace=False)
            for cid, data in self.client_data.items()
        }

        batch_size = 50
        total_batches = n_stream_samples // batch_size

        for batch_num in range(total_batches):
            for detector in self.detectors:
                cid = detector.config.client_id
                if cid not in self.client_data:
                    continue

                features, labels = self.client_data[cid]
                indices = sample_indices[cid]
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                for idx in batch_indices:
                    feat = features[idx]
                    label = labels[idx]
                    pred, drift = detector.process(feat, label)
                    if drift:
                        results['drift_events'].append({
                            'batch': batch_num, 'client_id': cid, 'sample': idx,
                            'drift_class': detector.last_drift_class.name if detector.last_drift_class else None
                        })

                sync_result = detector.maybe_sync()
                if sync_result and sync_result.synced:
                    results['sync_events'].append({
                        'batch': batch_num, 'client_id': cid,
                        'round': sync_result.round_number, 'sync_time_ms': sync_result.sync_time_ms
                    })

            if eval_features is not None and eval_labels is not None:
                acc = self._evaluate_detector(self.detectors[0], eval_features, eval_labels)
                results['global_accuracy'].append(acc)
                results['round'].append(batch_num + 1)

            results['client_stats'].append({
                'batch': batch_num,
                'clients': {d.config.client_id: d.get_statistics() for d in self.detectors}
            })

        if eval_features is not None and eval_labels is not None:
            results['final_accuracies'] = [
                self._evaluate_detector(d, eval_features, eval_labels) for d in self.detectors
            ]

        results['adaptation_stats'] = [d.get_statistics() for d in self.detectors]
        return results

    def _evaluate_detector(self, detector, features: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate detector on holdout set."""
        correct = sum(1 for i in range(len(features)) if detector.predict(features[i]) == labels[i])
        return correct / len(features)

    def get_experiment_summary(self, results: Dict) -> str:
        """Generate human-readable summary of experiment results."""
        lines = ["=" * 60, "FEDERATED INTEGRATED EXPERIMENT SUMMARY", "=" * 60]
        lines.append(f"\nTotal FL Syncs: {len(results.get('sync_events', []))}")
        lines.append(f"Total Drift Events: {len(results.get('drift_events', []))}")

        drift_events = results.get('drift_events', [])
        if drift_events:
            class_counts = {}
            for event in drift_events:
                dc = event.get('drift_class', 'Unknown')
                class_counts[dc] = class_counts.get(dc, 0) + 1
            lines.append("Drift Class Distribution:")
            for cls, count in sorted(class_counts.items()):
                lines.append(f"  - {cls}: {count}")

        if 'final_accuracies' in results:
            lines.append("\nFinal Client Accuracies:")
            for i, acc in enumerate(results['final_accuracies']):
                lines.append(f"  - Client {i+1}: {acc:.1%}")
            lines.append(f"  - Average: {np.mean(results['final_accuracies']):.1%}")

        if results.get('global_accuracy'):
            accs = results['global_accuracy']
            lines.append("\nAccuracy Trend (per batch):")
            lines.append(f"  - Start: {accs[0]:.1%}")
            lines.append(f"  - End: {accs[-1]:.1%}")
            if len(accs) > 1:
                lines.append(f"  - Delta: {accs[-1] - accs[0]:+.1%}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
