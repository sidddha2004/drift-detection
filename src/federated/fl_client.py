"""
Phase 5: Federated Learning - Client

FL Client wraps the edge detector with federated learning capabilities:
- Local training on private data
- Weight compression for upload
- Global model download and integration

Research Justification:
- Local training preserves data privacy (raw data never leaves device)
- Weight compression reduces communication cost (important for edge networks)
- Conditional upload avoids wasting bandwidth on insignificant updates
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from drift_detection.lstm_model import LSTMDriftDetector, LSTMConfig, ActivityLSTM
from federated.aggregator import QuantizationHelper


@dataclass
class FLClientConfig:
    """
    Configuration for FL client.

    Research Parameters:
    - local_epochs: Number of epochs for local training
      More epochs = better local fit but more computation
    - batch_size: Mini-batch size for local SGD
    - quantization_bits: Weight compression (8, 16, or 32)
      Lower = less bandwidth but potential accuracy loss
    - upload_threshold: Minimum accuracy improvement to upload
      Prevents wasting bandwidth on insignificant updates
    """
    # Local training
    local_epochs: int = 5
    batch_size: int = 64
    local_learning_rate: float = 0.001

    # Compression
    quantization_bits: int = 32  # 8, 16, or 32

    # Upload policy
    upload_threshold: float = 0.0  # Always upload
    min_samples_for_upload: int = 10


class FLClient:
    """
    Federated Learning Client.

    Wraps LSTMDriftDetector with FL capabilities:
    - Local training on private data
    - Weight compression/decompression
    - Conditional upload logic

    Usage:
        client = FLClient(client_id=1, config=fl_config)

        # Local training
        client.train_local(windows, labels, epochs=5)

        # Get compressed weights for upload
        compressed_weights, scale_params = client.get_compressed_weights()

        # Load global model from server
        client.load_global_model(global_weights, scale_params)
    """

    def __init__(
        self,
        client_id: int,
        config: Optional[FLClientConfig] = None,
        lstm_config: Optional[LSTMConfig] = None
    ):
        self.client_id = client_id
        self.config = config or FLClientConfig()

        # Internal drift detector with LSTM
        self.detector = LSTMDriftDetector(lstm_config=lstm_config)

        # FL state
        self.last_upload_accuracy: float = 0.0
        self.local_sample_count: int = 0
        self.rounds_participated: int = 0

        # Quantization helper
        self.quantizer = QuantizationHelper()

    def train_local(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        epochs: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Train on local data.

        Args:
            windows: Local window data (n_samples, 256, 9)
            labels: Local labels (n_samples,)
            epochs: Override default epochs

        Returns:
            Tuple of (final_loss, final_accuracy)
        """
        epochs = epochs or self.config.local_epochs
        n_samples = len(windows)
        self.local_sample_count = n_samples

        # Training loop
        losses = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_losses = []

            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                loss = self.detector.train_batch(
                    windows[batch_idx],
                    labels[batch_idx]
                )
                epoch_losses.append(loss)

            losses.append(np.mean(epoch_losses))

        # Evaluate
        accuracy, final_loss = self.detector.evaluate(windows, labels)

        return final_loss, accuracy

    def get_compressed_weights(
        self,
        bits: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
        """
        Get model weights, optionally compressed.

        Args:
            bits: Override default quantization bits

        Returns:
            Tuple of (weights, scale_params)
            If bits >= 32, scale_params is empty dict
        """
        bits = bits or self.config.quantization_bits
        weights = self.detector.model.get_weights()

        if bits >= 32:
            return weights, {}

        return self.quantizer.quantize(weights, bits)

    def load_global_model(
        self,
        weights: Dict[str, np.ndarray],
        scale_params: Optional[Dict[str, Tuple[float, float]]] = None,
        bits: Optional[int] = None
    ):
        """
        Load global model from server.

        Args:
            weights: Model weights (possibly quantized)
            scale_params: Dequantization parameters
            bits: Quantization bits used
        """
        bits = bits or self.config.quantization_bits

        if scale_params and len(scale_params) > 0:
            # Dequantize
            weights = self.quantizer.dequantize(weights, scale_params, bits)

        # Load into detector
        self.detector.model.set_weights(weights)

        # Reset drift detector for fresh monitoring
        self.detector.reset_drift_detector()

        self.rounds_participated += 1

    def should_upload(
        self,
        current_accuracy: float,
        baseline_accuracy: Optional[float] = None
    ) -> bool:
        """
        Decide whether to upload model update.

        Research Justification:
        - Avoids wasting bandwidth on insignificant updates
        - Conditional upload reduces communication overhead
        - Trade-off: may delay propagation of small improvements

        Args:
            current_accuracy: Current model accuracy
            baseline_accuracy: Accuracy at last upload (optional)

        Returns:
            True if should upload
        """
        # Check minimum samples
        if self.local_sample_count < self.config.min_samples_for_upload:
            return False

        # Always upload if no baseline
        if baseline_accuracy is None:
            return True

        # Check improvement threshold
        accuracy_delta = current_accuracy - baseline_accuracy
        return accuracy_delta >= self.config.upload_threshold

    def get_upload_payload(
        self,
        current_accuracy: float
    ) -> Optional[Dict]:
        """
        Prepare upload payload if conditions met.

        Args:
            current_accuracy: Current model accuracy

        Returns:
            Payload dict if should upload, None otherwise
        """
        if not self.should_upload(current_accuracy):
            return None

        weights, scale_params = self.get_compressed_weights()

        return {
            'client_id': self.client_id,
            'weights': weights,
            'scale_params': scale_params,
            'n_samples': self.local_sample_count,
            'accuracy': current_accuracy,
            'quantization_bits': self.config.quantization_bits,
            'rounds_participated': self.rounds_participated
        }

    def get_statistics(self) -> Dict:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'local_samples': self.local_sample_count,
            'rounds_participated': self.rounds_participated,
            'last_upload_accuracy': self.last_upload_accuracy,
            'quantization_bits': self.config.quantization_bits
        }

    # Delegate detector methods for convenience

    def predict(self, features: np.ndarray) -> int:
        """Predict using local model."""
        return self.detector.predict(features)

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate local model."""
        return self.detector.evaluate(features, labels)

    def update_and_check_drift(
        self,
        prediction: int,
        true_label: int
    ) -> bool:
        """Check for drift."""
        return self.detector.update_and_check_drift(prediction, true_label)
