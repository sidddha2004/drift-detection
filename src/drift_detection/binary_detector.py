"""
Phase 2: Binary Drift Detection (Stage A)

Implements:
- Lightweight MLP classifier for activity recognition
- ADWIN-based error tracking for drift detection
- Binary drift flag generation

Architecture:
    Window Features (45) → MLP (45→32→18) → Activity Prediction
                                        ↓
                            Binary Error: error_t = 1[ŷ ≠ y]
                                        ↓
                            ADWIN Monitor → DriftFlag (bool)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from river import drift


@dataclass
class MLPConfig:
    """Configuration for the activity classification MLP."""
    input_dim: int = 45
    hidden_dim: int = 32
    output_dim: int = 18  # PAMAP2 activity classes
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    activation: str = "relu"


@dataclass
class ADWINConfig:
    """Configuration for ADWIN drift detector."""
    delta: float = 0.002  # Confidence parameter
    max_buffer: int = 1000  # Maximum error buffer size
    min_window: int = 30  # Minimum samples before detection


class ActivityMLP(nn.Module):
    """
    Multi-layer Perceptron for human activity recognition.

    Architecture: Input → Hidden (ReLU) → Output (Softmax)
    """

    def __init__(self, config: Optional[MLPConfig] = None):
        super(ActivityMLP, self).__init__()
        self.config = config or MLPConfig()

        # Network layers
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.config.hidden_dim, self.config.output_dim)

        # Initialize weights (Xavier initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, 45)

        Returns:
            Logits of shape (batch_size, 18)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict class labels.

        Args:
            x: Input features of shape (batch_size, 45) or (45,)

        Returns:
            Predicted class labels
        """
        # Handle single sample
        if x.dim() == 1:
            x = x.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions.numpy()

    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract model weights as numpy arrays.

        Returns:
            Dict mapping layer names to weight arrays
        """
        return {
            'fc1.weight': self.fc1.weight.detach().numpy().copy(),
            'fc1.bias': self.fc1.bias.detach().numpy().copy(),
            'fc2.weight': self.fc2.weight.detach().numpy().copy(),
            'fc2.bias': self.fc2.bias.detach().numpy().copy()
        }

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """
        Load weights from numpy arrays.

        Args:
            weights: Dict mapping layer names to weight arrays
        """
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(weights['fc1.weight']))
            self.fc1.bias.copy_(torch.from_numpy(weights['fc1.bias']))
            self.fc2.weight.copy_(torch.from_numpy(weights['fc2.weight']))
            self.fc2.bias.copy_(torch.from_numpy(weights['fc2.bias']))


class BinaryDriftDetector:
    """
    Stage A: Binary drift detection using ADWIN on classification errors.

    Continuously monitors the error stream from the activity classifier
    and raises a drift flag when the error rate distribution changes.
    """

    def __init__(
        self,
        mlp_config: Optional[MLPConfig] = None,
        adwin_config: Optional[ADWINConfig] = None
    ):
        self.mlp_config = mlp_config or MLPConfig()
        self.adwin_config = adwin_config or ADWINConfig()

        # Initialize MLP classifier
        self.model = ActivityMLP(self.mlp_config)

        # Initialize loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.mlp_config.learning_rate,
            momentum=self.mlp_config.momentum,
            weight_decay=self.mlp_config.weight_decay
        )

        # Initialize ADWIN drift detector
        self.adwin = drift.ADWIN(delta=self.adwin_config.delta)

        # State tracking
        self.error_buffer = []
        self.total_samples = 0
        self.drift_detected = False
        self.drift_count = 0

    def predict(self, features: np.ndarray) -> int:
        """
        Predict activity class for a single window.

        Args:
            features: Feature vector of shape (45,)

        Returns:
            Predicted activity class (0-17)
        """
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
            True if drift detected, False otherwise
        """
        # Compute binary error
        error = 1.0 if prediction != true_label else 0.0

        # Update ADWIN
        self.adwin.update(error)

        # Track error buffer
        self.error_buffer.append(error)
        if len(self.error_buffer) > self.adwin_config.max_buffer:
            self.error_buffer.pop(0)

        self.total_samples += 1

        # Check for drift (only after minimum samples)
        self.drift_detected = False
        if self.total_samples >= self.adwin_config.min_window:
            if self.adwin.drift_detected:
                self.drift_detected = True
                self.drift_count += 1

        return self.drift_detected

    def train_batch(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Train model on a batch of data.

        Args:
            features: Shape (batch_size, 45)
            labels: Shape (batch_size,)

        Returns:
            Training loss
        """
        self.model.train()

        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels).long()

        # Forward pass
        logits = self.model(features_tensor)
        loss = self.criterion(logits, labels_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate model on a batch of data.

        Args:
            features: Shape (batch_size, 45)
            labels: Shape (batch_size,)

        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()

        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels).long()

        with torch.no_grad():
            logits = self.model(features_tensor)
            loss = self.criterion(logits, labels_tensor)

            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            accuracy = (predictions == labels_tensor).float().mean().item()

        return accuracy, loss.item()

    def get_error_rate(self, window_size: int = 100) -> float:
        """
        Get recent error rate.

        Args:
            window_size: Number of recent samples to consider

        Returns:
            Error rate (0.0 to 1.0)
        """
        if len(self.error_buffer) == 0:
            return 0.0

        recent_errors = self.error_buffer[-window_size:]
        return np.mean(recent_errors)

    def reset_drift_detector(self):
        """
        Reset ADWIN after drift adaptation.

        Called after successful adaptation to resume monitoring.
        """
        self.adwin = drift.ADWIN(delta=self.adwin_config.delta)
        self.error_buffer = []
        self.drift_detected = False

    def get_state(self) -> Dict:
        """
        Get detector state for serialization.

        Returns:
            Dict containing model weights and detector state
        """
        return {
            'model_weights': self.model.get_weights(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_samples': self.total_samples,
            'drift_count': self.drift_count,
            'recent_error_rate': self.get_error_rate()
        }

    def load_state(self, state: Dict):
        """
        Load detector state from serialization.

        Args:
            state: Dict from get_state()
        """
        self.model.set_weights(state['model_weights'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.total_samples = state['total_samples']
        self.drift_count = state['drift_count']


class EdgeClient:
    """
    Simulates an edge device running continuous drift detection.

    Combines the MLP classifier with ADWIN monitoring for
    end-to-end Stage A evaluation.
    """

    def __init__(
        self,
        client_id: int,
        mlp_config: Optional[MLPConfig] = None,
        adwin_config: Optional[ADWINConfig] = None
    ):
        self.client_id = client_id
        self.detector = BinaryDriftDetector(mlp_config, adwin_config)

        # Statistics
        self.predictions = []
        self.true_labels = []
        self.drift_flags = []
        self.error_history = []

    def process_stream(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        initial_train_epochs: int = 5
    ) -> Dict:
        """
        Process a data stream with continuous drift detection.

        Args:
            features: Shape (n_windows, 45)
            labels: Shape (n_windows,)
            initial_train_epochs: Initial training epochs

        Returns:
            Dict with detection statistics
        """
        # Initial training on first 10% of data
        initial_size = max(10, len(features) // 10)
        for epoch in range(initial_train_epochs):
            indices = np.random.permutation(initial_size)
            for i in range(0, initial_size, 32):
                batch_idx = indices[i:i+32]
                self.detector.train_batch(
                    features[batch_idx],
                    labels[batch_idx]
                )

        # Process stream
        for i in range(len(features)):
            # Predict
            prediction = self.detector.predict(features[i])
            self.predictions.append(prediction)

            # Check for drift
            drift_flag = self.detector.update_and_check_drift(
                prediction, labels[i]
            )
            self.drift_flags.append(1 if drift_flag else 0)

            # Track statistics
            self.true_labels.append(labels[i])
            self.error_history.append(1 if prediction != labels[i] else 0)

        # Return statistics
        predictions_arr = np.array(self.predictions)
        accuracy = (predictions_arr == labels).mean()

        return {
            'client_id': self.client_id,
            'total_samples': len(features),
            'accuracy': accuracy,
            'drift_count': self.detector.drift_count,
            'final_error_rate': self.detector.get_error_rate(),
            'drift_flags': self.drift_flags
        }

    def get_model_weights(self) -> Dict:
        """Extract model weights for federated learning."""
        return self.detector.model.get_weights()

    def load_model_weights(self, weights: Dict):
        """Load new model weights from federated aggregation."""
        self.detector.model.set_weights(weights)
        self.detector.reset_drift_detector()


if __name__ == "__main__":
    # Test binary drift detection
    import sys
    sys.path.append('..')
    from data_pipeline import (
        PAMAP2Preprocessor,
        create_data_pipeline,
        DriftConfig,
        DriftType,
        DriftInjector
    )

    # Load test data
    test_file = "../../PAMAP2_Dataset/Protocol/subject101.dat"

    print("Loading and preprocessing data...")
    preprocessor = PAMAP2Preprocessor()
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)

    print("Generating windows and features...")
    pipeline = create_data_pipeline(window_size=256, stride=128)
    features, labels = pipeline.process(accel_data, activity_labels)

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Inject drift for testing
    print("\nInjecting sudden drift at sample 500...")
    drift_config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        magnitude=1.0,
        injection_point=500
    )
    injector = DriftInjector(drift_config)

    # Re-preprocess with drift
    accel_data_drifted = preprocessor.preprocess_file(test_file)[0]
    accel_data_drifted = injector.inject_drift(accel_data_drifted)

    # Re-extract features for drifted data
    features_drifted, _ = pipeline.process(accel_data_drifted, activity_labels)

    # Test drift detection
    print("\n=== Testing Binary Drift Detection ===")
    detector = BinaryDriftDetector()

    # Initial training
    print("Initial training on first 50 samples...")
    for epoch in range(3):
        for i in range(0, 50, 16):
            detector.train_batch(features[i:i+16], labels[i:i+16])

    # Evaluate before drift
    acc_before, _ = detector.evaluate(features[:500], labels[:500])
    print(f"Accuracy before drift: {acc_before:.3f}")

    # Process drifted stream
    print("\nProcessing drifted stream...")
    drift_flags = []
    errors = []

    for i in range(len(features_drifted)):
        pred = detector.predict(features_drifted[i])
        drift_flag = detector.update_and_check_drift(pred, labels[i])
        drift_flags.append(drift_flag)
        errors.append(1 if pred != labels[i] else 0)

    print(f"Total drift flags: {sum(drift_flags)}")
    print(f"Drift events detected: {detector.drift_count}")
    print(f"Final error rate: {detector.get_error_rate():.3f}")

    # Show when drift was detected
    drift_indices = [i for i, flag in enumerate(drift_flags) if flag]
    if drift_indices:
        print(f"First drift detection at sample: {drift_indices[0]}")
        print(f"Drift detection latency: {drift_indices[0] - 500} samples")
