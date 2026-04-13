"""
LSTM Model for Human Activity Recognition

Provides a recurrent neural network architecture for capturing
temporal patterns in accelerometer data.

Architecture:
    Input: (batch, seq_len=256, input_size=9)
        ↓
    LSTM(input_size=9, hidden_size=64, num_layers=2)
        ↓
    Last hidden state (batch, 64)
        ↓
    Linear(64, 18)
        ↓
    Output: Activity logits (batch, 18)

Research Justification:
- HAR is sequential: activities have characteristic motion patterns over time
- LSTM captures long-range dependencies within 2.56s window
- Proven in literature: 85-95% accuracy on PAMAP2 (vs 40-60% for MLP)
- Still edge-friendly: 77K params = 308KB (fits in 512MB Pi RAM)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """
    Configuration for LSTM activity recognition model.

    Research Parameters:
    - input_size: 9 accelerometer axes (3 sensors × 3 axes)
    - hidden_size: Larger = more capacity but more compute
    - num_layers: 2 layers captures hierarchical temporal patterns
    - dropout: Regularization to prevent overfitting
    - learning_rate: Adam optimizer LR
    """
    # Input dimensions
    input_size: int = 9       # 9 accelerometer axes
    seq_len: int = 256        # 2.56s @ 100Hz

    # LSTM architecture
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False  # True = 2x params, better accuracy

    # Output
    output_dim: int = 18  # PAMAP2 activity classes

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64


class ActivityLSTM(nn.Module):
    """
    LSTM-based activity classifier for edge HAR.

    Takes raw windowed accelerometer data and outputs activity predictions.
    Designed for Raspberry Pi deployment (moderate compute, fits in RAM).
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        super(ActivityLSTM, self).__init__()
        self.config = config or LSTMConfig()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional
        )

        # Output classifier
        lstm_output_dim = (
            self.config.hidden_size * 2
            if self.config.bidirectional
            else self.config.hidden_size
        )
        self.fc = nn.Linear(lstm_output_dim, self.config.output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and FC weights for stable training."""
        # LSTM weight initialization (orthogonal for recurrent, uniform for input)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # FC initialization
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Args:
            x: Input of shape (batch, seq_len, input_size)
               e.g., (batch, 256, 9) for raw accelerometer windows

        Returns:
            Logits of shape (batch, output_dim)
        """
        # LSTM forward
        # output: (batch, seq_len, hidden_size * num_directions)
        # (h_n, c_n): (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state from last layer
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        if self.config.bidirectional:
            # Concatenate forward and backward last states
            h_forward = h_n[-2, :, :]  # Last layer, forward
            h_backward = h_n[-1, :, :]  # Last layer, backward
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_last = h_n[-1, :, :]  # Last layer

        # Classify
        logits = self.fc(h_last)
        return logits

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict activity labels.

        Args:
            x: Input of shape (batch, seq_len, input_size) or (seq_len, input_size)

        Returns:
            Predicted class labels (0-17)
        """
        # Handle single sample
        if x.dim() == 2:
            x = x.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions.numpy()

    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract all model weights for federated learning.

        Returns:
            Dict mapping layer names to numpy arrays
        """
        weights = {}
        sd = self.lstm.state_dict()

        # LSTM weights (store with lstm. prefix for consistent loading)
        for i in range(self.config.num_layers):
            weights[f'lstm.weight_ih_l{i}'] = sd[f'weight_ih_l{i}'].detach().numpy().copy()
            weights[f'lstm.weight_hh_l{i}'] = sd[f'weight_hh_l{i}'].detach().numpy().copy()
            weights[f'lstm.bias_ih_l{i}'] = sd[f'bias_ih_l{i}'].detach().numpy().copy()
            weights[f'lstm.bias_hh_l{i}'] = sd[f'bias_hh_l{i}'].detach().numpy().copy()

        # FC weights
        weights['fc.weight'] = self.fc.weight.detach().numpy().copy()
        weights['fc.bias'] = self.fc.bias.detach().numpy().copy()

        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """
        Load model weights for federated learning.

        Args:
            weights: Dict from get_weights()
        """
        lstm_state_dict = {}
        fc_state_dict = {}

        # Reconstruct LSTM state dict
        for i in range(self.config.num_layers):
            key_ih = f'lstm.weight_ih_l{i}'
            key_hh = f'lstm.weight_hh_l{i}'
            key_bi = f'lstm.bias_ih_l{i}'
            key_bh = f'lstm.bias_hh_l{i}'
            if key_ih in weights:
                lstm_state_dict[f'weight_ih_l{i}'] = torch.from_numpy(weights[key_ih])
                lstm_state_dict[f'weight_hh_l{i}'] = torch.from_numpy(weights[key_hh])
                lstm_state_dict[f'bias_ih_l{i}'] = torch.from_numpy(weights[key_bi])
                lstm_state_dict[f'bias_hh_l{i}'] = torch.from_numpy(weights[key_bh])

        # Load LSTM weights
        if lstm_state_dict:
            self.lstm.load_state_dict(lstm_state_dict)

        # FC weights (separate module)
        if 'fc.weight' in weights:
            fc_state_dict['weight'] = torch.from_numpy(weights['fc.weight'])
            fc_state_dict['bias'] = torch.from_numpy(weights['fc.bias'])
            self.fc.load_state_dict(fc_state_dict)

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count model parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_memory_size(self) -> Dict[str, float]:
        """
        Calculate memory footprint.

        Returns:
            Dict with sizes in KB for different precisions
        """
        total_params = self.count_parameters()[0]
        return {
            'float32': total_params * 4 / 1024,  # KB
            'float16': total_params * 2 / 1024,
            'int8': total_params * 1 / 1024
        }


class LSTMDriftDetector:
    """
    Drift detector using LSTM model.

    Same interface as BinaryDriftDetector but uses LSTM instead of MLP.
    """

    def __init__(
        self,
        lstm_config: Optional[LSTMConfig] = None,
        adwin_config: Optional[Dict] = None
    ):
        from river import drift

        self.lstm_config = lstm_config or LSTMConfig()
        self.adwin_config = adwin_config or {'delta': 0.002}

        # Initialize LSTM classifier
        self.model = ActivityLSTM(self.lstm_config)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lstm_config.learning_rate,
            weight_decay=self.lstm_config.weight_decay
        )

        # ADWIN drift detector
        self.adwin = drift.ADWIN(delta=self.adwin_config['delta'])

        # State tracking
        self.error_buffer = []
        self.total_samples = 0
        self.drift_detected = False
        self.drift_count = 0

    def predict(self, features: np.ndarray) -> int:
        """
        Predict activity class.

        Args:
            features: Either:
                - (seq_len, 9) raw window for single sample
                - (45,) feature vector (not supported for LSTM)

        Returns:
            Predicted activity class
        """
        # Ensure 2D input (seq_len, 9)
        if features.ndim == 1:
            raise ValueError("LSTM requires 2D input (seq_len, 9), not flattened features")

        features_tensor = torch.from_numpy(features).float()
        prediction = self.model.predict(features_tensor)
        return prediction[0]

    def update_and_check_drift(
        self,
        prediction: int,
        true_label: int
    ) -> bool:
        """
        Update error tracker and check for drift.

        Args:
            prediction: Predicted class
            true_label: Ground truth class

        Returns:
            True if drift detected
        """
        error = 1.0 if prediction != true_label else 0.0
        self.adwin.update(error)
        self.error_buffer.append(error)

        if len(self.error_buffer) > 1000:
            self.error_buffer.pop(0)

        self.total_samples += 1

        self.drift_detected = False
        if self.total_samples >= 30 and self.adwin.drift_detected:
            self.drift_detected = True
            self.drift_count += 1

        return self.drift_detected

    def train_batch(
        self,
        windows: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Train on batch of windows.

        Args:
            windows: Shape (batch_size, seq_len, 9)
            labels: Shape (batch_size,)

        Returns:
            Training loss
        """
        self.model.train()

        windows_tensor = torch.from_numpy(windows).float()
        labels_tensor = torch.from_numpy(labels).long()

        # Forward
        logits = self.model(windows_tensor)
        loss = self.criterion(logits, labels_tensor)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        windows: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate on batch of windows.

        Args:
            windows: Shape (batch_size, seq_len, 9)
            labels: Shape (batch_size,)

        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()

        windows_tensor = torch.from_numpy(windows).float()
        labels_tensor = torch.from_numpy(labels).long()

        with torch.no_grad():
            logits = self.model(windows_tensor)
            loss = self.criterion(logits, labels_tensor)

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            accuracy = (predictions == labels_tensor).float().mean().item()

        return accuracy, loss.item()

    def get_error_rate(self, window_size: int = 100) -> float:
        """Get recent error rate."""
        if len(self.error_buffer) == 0:
            return 0.0
        return np.mean(self.error_buffer[-window_size:])

    def reset_drift_detector(self):
        """Reset ADWIN after drift adaptation."""
        from river import drift
        self.adwin = drift.ADWIN(delta=self.adwin_config['delta'])
        self.error_buffer = []
        self.drift_detected = False

    def get_state(self) -> Dict:
        """Get detector state for serialization."""
        return {
            'model_weights': self.model.get_weights(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_samples': self.total_samples,
            'drift_count': self.drift_count,
            'recent_error_rate': self.get_error_rate()
        }

    def load_state(self, state: Dict):
        """Load detector state."""
        self.model.set_weights(state['model_weights'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.total_samples = state['total_samples']
        self.drift_count = state['drift_count']
