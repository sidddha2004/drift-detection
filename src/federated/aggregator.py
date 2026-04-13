"""
Phase 5: Federated Learning - Aggregator

Implements Federated Averaging (FedAvg) algorithm for aggregating
model updates from multiple edge clients.

FedAvg Algorithm:
    Global model w_g is maintained by server.
    Each client k trains locally on dataset D_k.
    Server aggregates: w_g = Σ (n_k / n_total) × w_k

Research Justification:
- Weighted averaging by sample count accounts for non-IID data
- Simple, communication-efficient (only weights transferred)
- Proven in federated learning literature (McMahan et al., 2017)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AggregationConfig:
    """Configuration for aggregation."""
    # Minimum clients required for aggregation
    min_clients: int = 3

    # Aggregation algorithm
    algorithm: str = "fedavg"  # fedavg, fedprox (future)

    # Weight clipping for stability
    clip_weights: bool = False
    clip_norm: float = 1.0


class FedAvgAggregator:
    """
    Federated Averaging aggregator.

    Combines model weights from multiple clients using
    sample-count weighted averaging.

    Mathematical Formulation:
        w_global = Σ_{k=1}^{K} (n_k / n_total) × w_k

        where:
        - K = number of clients
        - n_k = number of samples on client k
        - n_total = Σ n_k
        - w_k = model weights from client k

    Research Justification:
    - Clients with more data contribute more to global model
    - Unbiased estimator of centralized training
    - Handles non-IID data better than simple averaging
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config or AggregationConfig()

        # Buffer for collecting updates
        self.pending_updates: Dict[int, Tuple[Dict[str, np.ndarray], int]] = {}
        self.round_number: int = 0

    def receive_update(
        self,
        client_id: int,
        weights: Dict[str, np.ndarray],
        n_samples: int
    ):
        """
        Receive model update from a client.

        Args:
            client_id: Unique client identifier
            weights: Model weights (layer_name → numpy array)
            n_samples: Number of samples client trained on
        """
        # Deep copy weights to prevent mutation
        copied_weights = {k: v.copy() for k, v in weights.items()}
        self.pending_updates[client_id] = (copied_weights, n_samples)

    def aggregate(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Aggregate all pending updates using FedAvg.

        Returns:
            Aggregated global model weights, or None if insufficient clients
        """
        if len(self.pending_updates) < self.config.min_clients:
            return None

        # Compute total samples
        n_total = sum(n for _, n in self.pending_updates.values())

        # Initialize aggregated weights
        aggregated: Dict[str, np.ndarray] = {}

        # First pass: initialize structure from first client
        first_weights = next(iter(self.pending_updates.values()))[0]
        for key, value in first_weights.items():
            aggregated[key] = np.zeros_like(value)

        # Weighted sum: Σ (n_k / n_total) × w_k
        for client_id, (weights, n_samples) in self.pending_updates.items():
            weight_factor = n_samples / n_total

            for key, value in weights.items():
                aggregated[key] += weight_factor * value

        # Optional: weight clipping for stability
        if self.config.clip_weights:
            aggregated = self._clip_weights(aggregated)

        self.round_number += 1
        self.pending_updates.clear()

        return aggregated

    def _clip_weights(
        self,
        weights: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Clip weight norms for stability.

        Prevents any single client from dominating aggregation.
        """
        # Compute global norm
        total_norm = 0.0
        for value in weights.values():
            total_norm += np.sum(value ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if exceeds threshold
        if total_norm > self.config.clip_norm:
            scale = self.config.clip_norm / total_norm
            weights = {k: v * scale for k, v in weights.items()}

        return weights

    def get_aggregation_state(self) -> Dict:
        """Get current aggregation state."""
        return {
            'round_number': self.round_number,
            'pending_clients': list(self.pending_updates.keys()),
            'pending_count': len(self.pending_updates),
            'min_clients_required': self.config.min_clients
        }

    def reset(self):
        """Reset aggregator state."""
        self.pending_updates.clear()
        self.round_number = 0


class QuantizationHelper:
    """
    Helper for weight quantization (compression).

    Research Justification:
    - Full 32-bit weights may be excessive for edge networks
    - 8-bit quantization reduces bandwidth 4×
    - Research question: Is 8-bit sufficient for HAR models?

    Quantization Formula:
        q = round((w - w_min) / (w_max - w_min) × (2^bits - 1))

    Dequantization:
        w' = q / (2^bits - 1) × (w_max - w_min) + w_min
    """

    @staticmethod
    def quantize(
        weights: Dict[str, np.ndarray],
        bits: int = 8
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
        """
        Quantize weights to reduced bit width.

        Args:
            weights: Layer weights (float32)
            bits: Target bit width (8, 16, or 32)

        Returns:
            Tuple of (quantized_weights, scale_params)
            scale_params: Dict of (min, max) per layer for dequantization
        """
        if bits >= 32:
            # No compression
            return weights, {}

        quantized = {}
        scale_params = {}

        max_value = (1 << bits) - 1  # 2^bits - 1

        for key, value in weights.items():
            w_min = value.min()
            w_max = value.max()

            if w_max - w_min < 1e-8:
                # Near-constant weights, no need to quantize
                quantized[key] = value.copy()
                scale_params[key] = (w_min, w_max)
                continue

            # Normalize to [0, 1]
            normalized = (value - w_min) / (w_max - w_min)

            # Quantize to [0, 2^bits - 1]
            q = np.round(normalized * max_value).astype(np.uint8 if bits == 8 else np.uint16)

            quantized[key] = q
            scale_params[key] = (float(w_min), float(w_max))

        return quantized, scale_params

    @staticmethod
    def dequantize(
        quantized: Dict[str, np.ndarray],
        scale_params: Dict[str, Tuple[float, float]],
        bits: int = 8
    ) -> Dict[str, np.ndarray]:
        """
        Dequantize weights to float32.

        Args:
            quantized: Quantized weights
            scale_params: Scale parameters from quantization
            bits: Original quantization bits

        Returns:
            Dequantized float32 weights
        """
        if len(scale_params) == 0:
            # Was not quantized
            return quantized

        dequantized = {}
        max_value = (1 << bits) - 1

        for key, q in quantized.items():
            if key not in scale_params:
                dequantized[key] = q.astype(np.float32)
                continue

            w_min, w_max = scale_params[key]

            # Dequantize to [0, 1]
            normalized = q.astype(np.float32) / max_value

            # Scale back to original range
            dequantized[key] = normalized * (w_max - w_min) + w_min

        return dequantized

    @staticmethod
    def compute_compression_ratio(
        original: Dict[str, np.ndarray],
        quantized: Dict[str, np.ndarray],
        bits: int = 8
    ) -> float:
        """
        Compute compression ratio achieved.

        Returns:
            Ratio of original size to compressed size (>1 means compression)
        """
        # Original size in bytes (float32 = 4 bytes)
        original_size = sum(v.size * 4 for v in original.values())

        # Quantized size
        if bits == 8:
            quantized_size = sum(v.size * 1 for v in quantized.values())
        elif bits == 16:
            quantized_size = sum(v.size * 2 for v in quantized.values())
        else:
            quantized_size = original_size

        # Include scale params overhead (2 floats per layer)
        scale_overhead = len(quantized) * 8  # 2 × float32 per layer
        quantized_size += scale_overhead

        return original_size / quantized_size if quantized_size > 0 else 1.0
