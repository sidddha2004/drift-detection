"""
Integrated Multiclass Drift Detector with Adaptation

Combines all phases into a single end-to-end pipeline:
1. Stage A: Binary drift detection (ADWIN)
2. Stage B: Drift classification (4 types)
3. Stage C: Type-specific adaptation (mitigation)
4. Stage D: Federated learning sync (optional)

Usage:
    detector = IntegratedDriftDetector()

    for window, label in data_stream:
        pred = detector.predict(window)
        detector.update(label)  # Checks drift, classifies, adapts if needed

        if detector.drift_detected:
            print(f"Drift adapted: {detector.last_drift_class}")
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from drift_detection.lstm_model import LSTMDriftDetector, LSTMConfig, ActivityLSTM
from drift_detection.drift_classifier import DriftClassifier, DriftClass, DriftClassifierConfig
from adaptation.adapter import TypeSpecificAdapter, AdapterConfig, AdaptationResult

# Import FL client if available (optional dependency)
try:
    from federated.fl_client import FLClient, FLClientConfig
    FL_AVAILABLE = True
except ImportError:
    FL_AVAILABLE = False
    FLClient = None
    FLClientConfig = None


@dataclass
class IntegratedDriftDetectorConfig:
    """Configuration for integrated drift detector."""
    # Model
    lstm_config: Optional[LSTMConfig] = None

    # Drift detection
    adwin_delta: float = 0.002

    # Drift classification
    drift_classifier_config: Optional[DriftClassifierConfig] = None

    # Adaptation
    adapter_config: Optional[AdapterConfig] = None

    # Data buffering for adaptation
    max_buffer_size: int = 500
    min_samples_for_adaptation: int = 50


class IntegratedDriftDetector:
    """
    End-to-end drift detection with automatic adaptation.

    Pipeline:
        Input window → Model prediction → Error tracking → ADWIN check
                                           ↓
                            Drift detected? → YES → Drift classification
                                           ↓
                            Drift type → Adaptation strategy → Model update
    """

    def __init__(self, config: Optional[IntegratedDriftDetectorConfig] = None):
        self.config = config or IntegratedDriftDetectorConfig()

        # Initialize LSTM model
        lstm_config = self.config.lstm_config or LSTMConfig()
        self.detector = LSTMDriftDetector(lstm_config=lstm_config)

        # Initialize drift classifier (Stage B)
        classifier_config = self.config.drift_classifier_config or DriftClassifierConfig()
        self.drift_classifier = DriftClassifier(classifier_config)

        # Initialize adapter (Stage C)
        adapter_config = self.config.adapter_config or AdapterConfig()
        self.adapter = TypeSpecificAdapter(adapter_config)

        # Data buffers for adaptation
        self.recent_data: List[np.ndarray] = []
        self.recent_labels: List[int] = []
        self.max_buffer_size = self.config.max_buffer_size

        # State tracking
        self.drift_detected = False
        self.last_drift_class: Optional[DriftClass] = None
        self.last_adaptation_result: Optional[AdaptationResult] = None
        self.total_drifts = 0
        self.total_adaptations = 0
        self.samples_processed = 0

    def predict(self, features: np.ndarray) -> int:
        """
        Predict activity class for input window.

        Args:
            features: Input window of shape (256, 9)

        Returns:
            Predicted activity class (0-17)
        """
        return self.detector.predict(features)

    def update(self, prediction: int, true_label: int,
               features: Optional[np.ndarray] = None) -> bool:
        """
        Update detector with prediction result and check for drift.

        If drift is detected:
        1. Classify drift type (Stage B)
        2. Apply adaptation strategy (Stage C)
        3. Reset drift detector

        Args:
            prediction: Predicted activity class
            true_label: Ground truth activity class
            features: Current window features (needed for adaptation)

        Returns:
            True if drift was detected and adapted
        """
        self.samples_processed += 1

        # Store recent data for potential adaptation
        if features is not None:
            self.recent_data.append(features)
            self.recent_labels.append(true_label)

            # Trim buffer
            if len(self.recent_data) > self.max_buffer_size:
                self.recent_data.pop(0)
                self.recent_labels.pop(0)

        # Stage A: Binary drift detection
        drift_flag = self.detector.update_and_check_drift(prediction, true_label)

        if not drift_flag:
            self.drift_detected = False
            return False

        # Drift detected!
        self.drift_detected = True
        self.total_drifts += 1

        # Need features for adaptation
        if features is None or len(self.recent_data) < self.config.min_samples_for_adaptation:
            return False

        # Stage B: Classify drift type
        error_buffer = np.array(self.detector.error_buffer[-100:])
        drift_class, drift_details = self.drift_classifier.classify(
            errors=error_buffer,
            recent_data=np.array(self.recent_data[-50:]),
            baseline_data=np.array(self.recent_data[:50]) if len(self.recent_data) >= 100 else None
        )
        self.last_drift_class = drift_class

        # Stage C: Apply adaptation strategy
        recent_data = np.array(self.recent_data)
        recent_labels = np.array(self.recent_labels)
        baseline_data = np.array(self.recent_data[:50]) if len(self.recent_data) >= 100 else None
        baseline_labels = np.array(self.recent_labels[:50]) if len(self.recent_labels) >= 100 else None

        self.last_adaptation_result = self.adapter.adapt(
            drift_class=self.last_drift_class,
            detector=self.detector,
            recent_data=recent_data,
            recent_labels=recent_labels,
            baseline_data=baseline_data,
            baseline_labels=baseline_labels if baseline_labels is not None else None,
            timestamp=self.samples_processed
        )

        self.total_adaptations += 1

        # Reset drift detector after adaptation
        self.detector.reset_drift_detector()

        # Clear recent buffer (keep only post-adaptation samples)
        self.recent_data = []
        self.recent_labels = []

        return True

    def process(self, features: np.ndarray, true_label: int) -> Tuple[int, bool]:
        """
        Full processing pipeline: predict, update, and adapt if needed.

        Args:
            features: Input window features
            true_label: Ground truth activity class

        Returns:
            Tuple of (prediction, drift_detected)
        """
        prediction = self.predict(features)
        drift = self.update(prediction, true_label, features)
        return prediction, drift

    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        return {
            'samples_processed': self.samples_processed,
            'total_drifts': self.total_drifts,
            'total_adaptations': self.total_adaptations,
            'drift_rate': self.total_drifts / max(1, self.samples_processed),
            'last_drift_class': self.last_drift_class.name if self.last_drift_class else None,
            'last_adaptation_success': self.last_adaptation_result.success if self.last_adaptation_result else None,
            'recent_error_rate': self.detector.get_error_rate(),
            'current_accuracy': 1.0 - self.detector.get_error_rate()
        }

    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current model weights for FL."""
        return self.detector.model.get_weights()

    def load_model_weights(self, weights: Dict[str, np.ndarray]):
        """Load model weights from FL server."""
        self.detector.model.set_weights(weights)
        self.detector.reset_drift_detector()


@dataclass
class FederatedIntegratedDriftDetectorConfig:
    """
    Configuration for Federated Integrated Drift Detector.

    Extends IntegratedDriftDetectorConfig with FL-specific parameters.

    Research Parameters:
    - sync_interval_adaptations: Number of local adaptations before FL sync
      Lower = more frequent sync (better model freshness) but higher communication cost
      Higher = less communication but potentially stale global model
    - sync_interval_minutes: Time-based sync trigger (fail-safe)
      Prevents never syncing if few adaptations occur
    - quantization_bits: Weight compression for upload (8/16/32)
      Research trade-off: bandwidth vs. accuracy
    - client_id: Unique identifier for this client
      Used for tracking and non-IID analysis
    """
    # Base detector config
    lstm_config: Optional[LSTMConfig] = None
    adwin_delta: float = 0.002
    drift_classifier_config: Optional[DriftClassifierConfig] = None
    adapter_config: Optional[AdapterConfig] = None

    # Data buffering
    max_buffer_size: int = 500
    min_samples_for_adaptation: int = 50

    # FL-specific config
    sync_interval_adaptations: int = 5
    sync_interval_minutes: float = 10.0
    quantization_bits: int = 32
    client_id: int = 0

    # FL client config
    fl_client_config: Optional['FLClientConfig'] = None


@dataclass
class FLSyncResult:
    """Result from a federated learning sync operation."""
    synced: bool
    round_number: int
    global_model_version: int
    sync_time_ms: float
    was_quantized: bool
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'synced': self.synced,
            'round_number': self.round_number,
            'global_model_version': self.global_model_version,
            'sync_time_ms': self.sync_time_ms,
            'was_quantized': self.was_quantized,
            'details': self.details
        }


class FederatedIntegratedDriftDetector:
    """
    Federated Learning wrapper around IntegratedDriftDetector.

    Architecture:
        ┌─────────────────────────────────────────────────────────────┐
        │  FederatedIntegratedDriftDetector                           │
        │  ┌───────────────────────────────────────────────────────┐  │
        │  │  IntegratedDriftDetector (detection + classification  │  │
        │  │   + adaptation)                                        │  │
        │  └───────────────────────────────────────────────────────┘  │
        │         ↑                                                   │
        │  FL integration layer (sync tracking, weight exchange)      │
        └─────────────────────────────────────────────────────────────┘

    Pipeline Flow:
        1. predict(features) → delegate to internal detector
        2. update(prediction, label, features) → detect, classify, adapt
        3. maybe_sync() → if adaptations >= threshold, trigger FL sync

    Research Usage:
        # Create detector with FL config
        config = FederatedIntegratedDriftDetectorConfig(
            client_id=0,
            sync_interval_adaptations=5,  # Sync every 5 adaptations
            quantization_bits=16  # Compress weights to 16-bit
        )
        detector = FederatedIntegratedDriftDetector(config)

        # Process data stream
        for features, label in data_stream:
            pred = detector.predict(features)
            detector.update(pred, label, features)

            # Check if FL sync should be triggered
            sync_result = detector.maybe_sync()
            if sync_result and sync_result.synced:
                print(f"Synced at round {sync_result.round_number}")

    When to use:
        - Multi-client FL experiments with non-IID data
        - Research on drift adaptation in federated settings
        - Privacy-preserving HAR with distributed learning

    When NOT to use:
        - Single-device deployment (use IntegratedDriftDetector)
        - Ablation studies without FL (use IntegratedDriftDetector)
    """

    def __init__(
        self,
        config: Optional[FederatedIntegratedDriftDetectorConfig] = None,
        fl_client: Optional['FLClient'] = None
    ):
        """
        Initialize federated drift detector.

        Args:
            config: Configuration for detector and FL
            fl_client: Optional pre-configured FL client
                       If None, creates internal client
        """
        self.config = config or FederatedIntegratedDriftDetectorConfig()

        # Create internal detector (composition, not inheritance)
        detector_config = IntegratedDriftDetectorConfig(
            lstm_config=self.config.lstm_config,
            adwin_delta=self.config.adwin_delta,
            drift_classifier_config=self.config.drift_classifier_config,
            adapter_config=self.config.adapter_config,
            max_buffer_size=self.config.max_buffer_size,
            min_samples_for_adaptation=self.config.min_samples_for_adaptation
        )
        self.detector = IntegratedDriftDetector(detector_config)

        # FL client - either provided or created internally
        if fl_client is not None:
            self.fl_client = fl_client
        elif FL_AVAILABLE:
            if self.config.fl_client_config is None:
                self.config.fl_client_config = FLClientConfig(
                    quantization_bits=self.config.quantization_bits
                )
            self.fl_client = FLClient(
                client_id=self.config.client_id,
                config=self.config.fl_client_config,
                lstm_config=self.config.lstm_config
            )
        else:
            self.fl_client = None

        # FL sync tracking
        self.adaptations_since_sync: int = 0
        self.last_sync_time: float = time.time()
        self.last_sync_result: Optional[FLSyncResult] = None
        self.total_syncs: int = 0

        # Reference to FL server (set externally by simulator)
        self._fl_server = None

    def predict(self, features: np.ndarray) -> int:
        """
        Predict activity class for input window.

        Delegates to internal IntegratedDriftDetector.

        Args:
            features: Input window of shape (256, 9) or (45,)

        Returns:
            Predicted activity class (0-17)
        """
        return self.detector.predict(features)

    def update(self, prediction: int, true_label: int,
               features: Optional[np.ndarray] = None) -> bool:
        """
        Update detector with prediction result and check for drift.

        If drift is detected and adapted, increments adaptations_since_sync counter.

        Args:
            prediction: Predicted activity class
            true_label: Ground truth activity class
            features: Current window features (needed for adaptation)

        Returns:
            True if drift was detected and adapted
        """
        drift_detected = self.detector.update(prediction, true_label, features)

        if drift_detected:
            self.adaptations_since_sync += 1

        return drift_detected

    def process(self, features: np.ndarray, true_label: int) -> Tuple[int, bool]:
        """
        Full processing pipeline: predict, update, and optionally sync.

        Args:
            features: Input window features
            true_label: Ground truth activity class

        Returns:
            Tuple of (prediction, drift_detected)
        """
        prediction, drift = self.detector.process(features, true_label)
        return prediction, drift

    def maybe_sync(self) -> Optional[FLSyncResult]:
        """
        Check if FL sync should be triggered and perform if needed.

        Sync triggers:
        1. adaptations_since_sync >= sync_interval_adaptations
        2. elapsed time since last sync >= sync_interval_minutes

        Returns:
            FLSyncResult if sync was triggered, None otherwise
        """
        if self.fl_client is None:
            return None

        # Check if server is set (required for sync)
        if self._fl_server is None:
            return None

        # Check adaptations threshold
        should_sync = (
            self.adaptations_since_sync >= self.config.sync_interval_adaptations
        )

        # Check time threshold
        elapsed_minutes = (time.time() - self.last_sync_time) / 60.0
        if elapsed_minutes >= self.config.sync_interval_minutes:
            should_sync = True

        if not should_sync:
            return None

        # Perform sync
        start_time = time.time()

        # Get current weights (optionally quantized)
        weights, scale_params = self.fl_client.get_compressed_weights(
            bits=self.config.quantization_bits
        )

        # Get accuracy for upload decision
        # In real deployment, this would use held-out test set
        current_accuracy = 1.0 - self.detector.detector.get_error_rate()

        # Prepare upload payload
        payload = self.fl_client.get_upload_payload(current_accuracy)

        if payload is None:
            # Threshold not met for upload
            self.adaptations_since_sync = 0  # Reset counter anyway
            return FLSyncResult(
                synced=False,
                round_number=self._fl_server.aggregator.round_number,
                global_model_version=self._fl_server.global_model_version or 0,
                sync_time_ms=(time.time() - start_time) * 1000,
                was_quantized=self.config.quantization_bits < 32,
                details={'reason': 'upload_threshold_not_met'}
            )

        # Send to server
        self._fl_server.receive_update(
            client_id=self.config.client_id,
            weights=weights,
            n_samples=self.fl_client.local_sample_count,
            accuracy=current_accuracy
        )

        # Trigger aggregation and broadcast
        global_weights = self._fl_server.aggregate()

        if global_weights is not None:
            # Dequantize if needed
            if scale_params and len(scale_params) > 0:
                from federated.aggregator import QuantizationHelper
                global_weights = QuantizationHelper.dequantize(
                    global_weights, scale_params, self.config.quantization_bits
                )

            # Load global model
            self.fl_client.load_global_model(global_weights, scale_params={})

            # Also update internal detector
            self.detector.load_model_weights(global_weights)

            # Broadcast to other clients
            self._fl_server.broadcast()

        # Update tracking
        sync_time_ms = (time.time() - start_time) * 1000
        self.adaptations_since_sync = 0
        self.last_sync_time = time.time()
        self.total_syncs += 1

        self.last_sync_result = FLSyncResult(
            synced=True,
            round_number=self._fl_server.completed_rounds,
            global_model_version=self._fl_server.global_model_version or 0,
            sync_time_ms=sync_time_ms,
            was_quantized=self.config.quantization_bits < 32,
            details={
                'client_id': self.config.client_id,
                'adaptations_triggered': self.adaptations_since_sync,
                'accuracy': current_accuracy
            }
        )

        return self.last_sync_result

    def set_fl_server(self, server):
        """
        Set reference to FL server for sync operations.

        Called by FLSimulator when registering clients.

        Args:
            server: FLServer instance
        """
        self._fl_server = server

    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current model weights for FL."""
        return self.detector.get_model_weights()

    def load_model_weights(self, weights: Dict[str, np.ndarray]):
        """Load model weights from FL server."""
        self.detector.load_model_weights(weights)

    def get_statistics(self) -> Dict:
        """Get combined local + FL statistics."""
        local_stats = self.detector.get_statistics()

        fl_stats = {
            'client_id': self.config.client_id,
            'adaptations_since_sync': self.adaptations_since_sync,
            'total_syncs': self.total_syncs,
            'last_sync_time': self.last_sync_time,
            'fl_client_connected': self.fl_client is not None
        }

        if self.last_sync_result:
            fl_stats['last_sync_round'] = self.last_sync_result.round_number
            fl_stats['last_sync_time_ms'] = self.last_sync_result.sync_time_ms

        return {**local_stats, **fl_stats}

    def get_full_statistics(self) -> Dict:
        """Get comprehensive statistics including internal detector."""
        return {
            'federated_stats': self.get_statistics(),
            'internal_detector_stats': self.detector.get_statistics(),
            'fl_client_stats': (
                self.fl_client.get_statistics()
                if self.fl_client else None
            )
        }


def test_integrated_detector():
    """Test the integrated drift detector with synthetic data."""
    print("=" * 60)
    print("INTEGRATED DRIFT DETECTOR - TEST")
    print("=" * 60)

    # Create detector with LSTM
    print("\n1. Creating integrated detector...")
    config = IntegratedDriftDetectorConfig(
        lstm_config=LSTMConfig(
            input_size=9,
            hidden_size=32,  # Smaller for faster testing
            num_layers=1,
            output_dim=4  # Match n_classes
        ),
        adwin_delta=0.01,  # More sensitive for testing
        max_buffer_size=200,
        min_samples_for_adaptation=30
    )
    detector = IntegratedDriftDetector(config)
    print("   PASS")

    # Generate synthetic window data for LSTM
    print("\n2. Generating synthetic window data...")
    np.random.seed(42)
    n_samples = 600
    n_classes = 4
    seq_len = 256

    # Create learnable patterns (raw windows for LSTM)
    windows = []
    labels = []

    for class_id in range(n_classes):
        class_samples = n_samples // n_classes
        for _ in range(class_samples):
            # Create class-specific temporal pattern
            window = np.zeros((seq_len, 9), dtype=np.float32)
            for axis in range(9):
                base_signal = np.sin(np.linspace(0, 2 * np.pi * (class_id + 1), seq_len))
                noise = np.random.randn(seq_len) * 0.1
                window[:, axis] = base_signal * (axis + 1) / 9 + noise
            windows.append(window)
            labels.append(class_id)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # First 300 samples: normal distribution
    # Next 300 samples: drift (labels shifted by 1)
    drift_point = 300
    labels_after_drift = labels.copy()
    labels_after_drift[drift_point:] = (labels[drift_point:] + 1) % n_classes

    print(f"   Generated {n_samples} windows, {n_classes} classes")
    print(f"   Drift injected at sample {drift_point}")
    print("   PASS")

    # Pre-train on first 200 samples (pre-drift data)
    print("\n3. Pre-training on pre-drift data...")
    train_windows = windows[:200]
    train_labels = labels[:200]

    for epoch in range(3):  # Fewer epochs for faster testing
        indices = np.random.permutation(len(train_windows))
        for i in range(0, len(train_windows), 32):
            batch_idx = indices[i:i+32]
            detector.detector.train_batch(train_windows[batch_idx], train_labels[batch_idx])

    # Evaluate pre-training accuracy
    pre_preds = [detector.predict(w) for w in windows[200:drift_point]]
    pre_acc = sum(1 for p, l in zip(pre_preds, labels[200:drift_point]) if p == l) / len(pre_preds)
    print(f"   Pre-drift accuracy: {pre_acc:.1%}")
    print("   PASS")

    # Process stream with drift
    print("\n4. Processing data stream with drift...")
    predictions = []
    drift_events = []

    for i in range(200, n_samples):
        label = labels_after_drift[i] if i >= drift_point else labels[i]
        pred, drift = detector.process(windows[i], label)
        predictions.append(pred)
        if drift:
            drift_events.append({
                'sample': i,
                'drift_class': detector.last_drift_class,
                'adaptation_success': detector.last_adaptation_result.success if detector.last_adaptation_result else False
            })

    print(f"   Total drifts detected: {detector.total_drifts}")
    print(f"   Total adaptations: {detector.total_adaptations}")

    for event in drift_events[:5]:  # Show first 5
        print(f"   - Sample {event['sample']}: {event['drift_class']}, success={event['adaptation_success']}")
    print("   PASS")

    # Print statistics
    print("\n5. Final statistics:")
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("INTEGRATED DETECTOR TEST COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_integrated_detector()
