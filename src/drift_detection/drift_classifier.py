"""
Phase 3: Drift Classification (Stage B)

Classifies the type of drift detected by Stage A into 4 categories:
- Sudden: Abrupt error increase
- Gradual: Slow error ramp
- Recurring: Matches historical pattern
- Feature-Level: Single-axis variance shift

Activated only when DriftFlag=True from Stage A (energy-aware).
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import IntEnum
from collections import deque


class DriftClass(IntEnum):
    """Drift type classification output."""
    SUDDEN = 0
    GRADUAL = 1
    RECURRING = 2
    FEATURE_LEVEL = 3


@dataclass
class DriftClassifierConfig:
    """Configuration for drift classifier."""
    # Error buffer size for analysis
    error_buffer_size: int = 100

    # Classification thresholds
    sudden_slope_threshold: float = 0.05
    sudden_variance_threshold: float = 0.3
    gradual_slope_min: float = 0.01
    gradual_slope_max: float = 0.05
    recurring_match_threshold: float = 0.85
    feature_variance_threshold: float = 2.0

    # Memory for recurring detection
    max_saved_patterns: int = 10


class DriftFeatureExtractor:
    """
    Extract features from error buffer and raw data for drift classification.
    """

    def __init__(self, config: Optional[DriftClassifierConfig] = None):
        self.config = config or DriftClassifierConfig()

    def compute_error_slope(self, errors: np.ndarray) -> float:
        """
        Compute slope of error trend using linear regression.

        Args:
            errors: Array of error values (0 or 1)

        Returns:
            Slope coefficient (positive = increasing errors)
        """
        if len(errors) < 10:
            return 0.0

        x = np.arange(len(errors))
        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = errors.mean()

        numerator = np.sum((x - x_mean) * (errors - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def compute_error_variance(self, errors: np.ndarray) -> float:
        """
        Compute variance of error stream.

        Args:
            errors: Array of error values

        Returns:
            Variance (0.0 to 0.25 for binary errors)
        """
        if len(errors) < 10:
            return 0.0
        return np.var(errors)

    def compute_error_kurtosis(self, errors: np.ndarray) -> float:
        """
        Compute kurtosis of error distribution.

        High kurtosis = impulsive/peaked errors (sudden drift)

        Args:
            errors: Array of error values

        Returns:
            Kurtosis value
        """
        if len(errors) < 10:
            return 0.0

        mean = np.mean(errors)
        std = np.std(errors)

        if std < 1e-8:
            return 0.0

        kurtosis = np.mean(((errors - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis

    def compute_autocorrelation(self, errors: np.ndarray, lag: int = 1) -> float:
        """
        Compute lag-1 autocorrelation of error stream.

        High autocorrelation = temporal structure (gradual drift)

        Args:
            errors: Array of error values
            lag: Lag for autocorrelation

        Returns:
            Autocorrelation coefficient (-1 to 1)
        """
        if len(errors) < 20:
            return 0.0

        n = len(errors)
        mean = np.mean(errors)
        var = np.var(errors)

        if var < 1e-8:
            return 0.0

        autocov = np.sum((errors[:n-lag] - mean) * (errors[lag:] - mean)) / n
        return autocov / var

    def compute_raw_variance_shift(
        self,
        recent_data: np.ndarray,
        baseline_data: np.ndarray
    ) -> float:
        """
        Compute change in raw data variance.

        Args:
            recent_data: Recent window data (n_windows, 9)
            baseline_data: Baseline window data (n_windows, 9)

        Returns:
            Variance shift ratio
        """
        recent_var = np.var(recent_data)
        baseline_var = np.var(baseline_data)

        if baseline_var < 1e-8:
            return 0.0

        return recent_var / baseline_var

    def compute_sensor_asymmetry(
        self,
        recent_data: np.ndarray
    ) -> np.ndarray:
        """
        Compute variance per sensor axis.

        Args:
            recent_data: Recent window data (n_windows, 9)

        Returns:
            Variance per axis (9,)
        """
        return np.var(recent_data, axis=0)

    def extract_all_features(
        self,
        errors: np.ndarray,
        recent_data: Optional[np.ndarray] = None,
        baseline_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract all features for drift classification.

        Args:
            errors: Error buffer (n_samples,)
            recent_data: Recent raw windows (optional)
            baseline_data: Baseline raw windows (optional)

        Returns:
            Feature vector (10,)
        """
        features = []

        # Error-based features
        features.append(self.compute_error_slope(errors))
        features.append(self.compute_error_variance(errors))
        features.append(self.compute_error_kurtosis(errors))
        features.append(self.compute_autocorrelation(errors))

        # Baseline deviation
        baseline_error_rate = 0.1  # Assumed baseline ~10% error
        current_error_rate = np.mean(errors)
        features.append(abs(current_error_rate - baseline_error_rate))

        # Raw data features (if available)
        if recent_data is not None:
            if baseline_data is not None:
                features.append(
                    self.compute_raw_variance_shift(recent_data, baseline_data)
                )
            else:
                features.append(1.0)

            # Sensor asymmetry (max variance ratio)
            axis_vars = self.compute_sensor_asymmetry(recent_data)
            features.append(axis_vars.max() / (axis_vars.mean() + 1e-8))
        else:
            features.append(1.0)
            features.append(1.0)

        # Padding to ensure 10 features
        while len(features) < 10:
            features.append(0.0)

        return np.array(features[:10])


class DriftClassifier:
    """
    Rule-based drift type classifier.

    Uses interpretable rules to classify drift into 4 categories.
    """

    def __init__(self, config: Optional[DriftClassifierConfig] = None):
        self.config = config or DriftClassifierConfig()
        self.feature_extractor = DriftFeatureExtractor(self.config)

        # Pattern memory for recurring drift detection
        self.saved_patterns: List[Dict] = []

        # State
        self.last_drift_class: Optional[DriftClass] = None
        self.last_drift_time: int = 0
        self.classification_history: List[DriftClass] = []

    def compute_memory_match_score(
        self,
        current_features: np.ndarray
    ) -> float:
        """
        Compute similarity to saved drift patterns.

        Args:
            current_features: Current feature vector

        Returns:
            Cosine similarity to best-matching pattern (0 to 1)
        """
        if len(self.saved_patterns) == 0:
            return 0.0

        best_score = 0.0

        for pattern in self.saved_patterns:
            stored_features = pattern['features']

            # Cosine similarity
            dot_product = np.dot(current_features, stored_features)
            norm_current = np.linalg.norm(current_features)
            norm_stored = np.linalg.norm(stored_features)

            if norm_current < 1e-8 or norm_stored < 1e-8:
                continue

            similarity = dot_product / (norm_current * norm_stored)
            best_score = max(best_score, similarity)

        return best_score

    def save_pattern(
        self,
        features: np.ndarray,
        drift_class: DriftClass,
        timestamp: int
    ):
        """
        Save drift pattern to memory.

        Args:
            features: Feature vector at drift time
            drift_class: Classified drift type
            timestamp: Sample index
        """
        pattern = {
            'features': features.copy(),
            'drift_class': drift_class,
            'timestamp': timestamp
        }

        self.saved_patterns.append(pattern)

        # Evict oldest if at capacity
        if len(self.saved_patterns) > self.config.max_saved_patterns:
            self.saved_patterns.pop(0)

    def classify(
        self,
        errors: np.ndarray,
        recent_data: Optional[np.ndarray] = None,
        baseline_data: Optional[np.ndarray] = None,
        time_since_last_drift: int = 0
    ) -> Tuple[DriftClass, Dict]:
        """
        Classify drift type from error features.

        Args:
            errors: Error buffer (n_samples,)
            recent_data: Recent raw windows (optional)
            baseline_data: Baseline raw windows (optional)
            time_since_last_drift: Samples since last drift event

        Returns:
            Tuple of (drift_class, details)
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(
            errors, recent_data, baseline_data
        )

        # Compute memory match for recurring detection
        memory_match = self.compute_memory_match_score(features)

        # Extract individual features for rules
        error_slope = features[0]
        error_variance = features[1]
        raw_variance_shift = features[5]

        # Apply classification rules
        drift_class: DriftClass
        confidence: float
        reason: str

        # Rule 1: Recurring (check first - pattern match)
        if memory_match > self.config.recurring_match_threshold:
            drift_class = DriftClass.RECURRING
            confidence = memory_match
            reason = f"Memory match: {memory_match:.3f}"

        # Rule 2: Sudden (steep slope, high variance)
        elif (error_slope > self.config.sudden_slope_threshold and
              error_variance > self.config.sudden_variance_threshold):
            drift_class = DriftClass.SUDDEN
            confidence = min(1.0, error_slope / 0.1)
            reason = f"Steep slope: {error_slope:.3f}, variance: {error_variance:.3f}"

        # Rule 3: Gradual (moderate slope)
        elif (error_slope > self.config.gradual_slope_min and
              error_slope <= self.config.gradual_slope_max):
            drift_class = DriftClass.GRADUAL
            confidence = min(1.0, error_slope / self.config.gradual_slope_max)
            reason = f"Gradual slope: {error_slope:.3f}"

        # Rule 4: Feature-Level (high raw variance, low error slope)
        elif (raw_variance_shift > self.config.feature_variance_threshold and
              error_slope < self.config.gradual_slope_min):
            drift_class = DriftClass.FEATURE_LEVEL
            confidence = min(1.0, raw_variance_shift / 3.0)
            reason = f"Variance shift: {raw_variance_shift:.3f}"

        # Default: Sudden (conservative)
        else:
            drift_class = DriftClass.SUDDEN
            confidence = 0.5
            reason = "Default classification"

        # Store state
        self.last_drift_class = drift_class
        self.classification_history.append(drift_class)

        details = {
            'features': features,
            'confidence': confidence,
            'reason': reason,
            'memory_match': memory_match,
            'error_slope': error_slope,
            'error_variance': error_variance,
            'raw_variance_shift': raw_variance_shift
        }

        return drift_class, details

    def update_memory(
        self,
        features: np.ndarray,
        drift_class: DriftClass,
        timestamp: int
    ):
        """
        Update pattern memory after drift adaptation.

        Args:
            features: Feature vector
            drift_class: Classified drift type
            timestamp: Current sample index
        """
        # Save all patterns for potential recurring detection
        self.save_pattern(features, drift_class, timestamp)
        self.last_drift_time = timestamp

    def get_statistics(self) -> Dict:
        """
        Get classifier statistics.

        Returns:
            Dict with classification history summary
        """
        if len(self.classification_history) == 0:
            return {
                'total_classifications': 0,
                'class_distribution': {}
            }

        classes = np.array(self.classification_history)
        unique, counts = np.unique(classes, return_counts=True)

        return {
            'total_classifications': len(classes),
            'class_distribution': {
                DriftClass(u).name: int(c)
                for u, c in zip(unique, counts)
            },
            'last_classification': (
                DriftClass(self.last_drift_class).name
                if self.last_drift_class is not None else None
            ),
            'saved_patterns': len(self.saved_patterns)
        }


class StageBDetector:
    """
    Complete Stage B: Drift Classification Module.

    Wraps DriftClassifier with buffer management and state tracking.
    """

    def __init__(self, config: Optional[DriftClassifierConfig] = None):
        self.config = config or DriftClassifierConfig()
        self.classifier = DriftClassifier(self.config)

        # Error buffer (filled by Stage A)
        self.error_buffer: deque = deque(maxlen=self.config.error_buffer_size)

        # State
        self.active = False  # Activated on DriftFlag=True
        self.current_drift_class: Optional[DriftClass] = None
        self.current_details: Optional[Dict] = None

        # Data buffers for raw analysis
        self.recent_windows: deque = deque(maxlen=50)
        self.baseline_windows: Optional[np.ndarray] = None

    def add_error(self, error: float):
        """
        Add error sample to buffer.

        Called continuously during Stage A monitoring.
        """
        self.error_buffer.append(error)

    def add_window(self, window: np.ndarray, is_baseline: bool = False):
        """
        Add raw window data for variance analysis.

        Args:
            window: Raw window data (window_size, 9)
            is_baseline: Whether this is baseline (pre-drift) data
        """
        if is_baseline:
            if self.baseline_windows is None:
                self.baseline_windows = []
            self.baseline_windows.append(window)
        else:
            self.recent_windows.append(window)

    def activate(self) -> bool:
        """
        Activate Stage B for drift classification.

        Returns:
            True if enough data for classification, False otherwise
        """
        self.active = True

        # Check if we have enough data
        if len(self.error_buffer) < 30:
            return False

        return True

    def classify_current_drift(
        self,
        timestamp: int
    ) -> Tuple[DriftClass, Dict]:
        """
        Perform drift classification.

        Args:
            timestamp: Current sample index

        Returns:
            Tuple of (drift_class, details)
        """
        errors = np.array(self.error_buffer)

        recent_data = None
        baseline_data = None

        if len(self.recent_windows) > 0:
            recent_data = np.array(self.recent_windows)

        if self.baseline_windows is not None and len(self.baseline_windows) > 0:
            baseline_data = np.array(self.baseline_windows)

        drift_class, details = self.classifier.classify(
            errors, recent_data, baseline_data
        )

        self.current_drift_class = drift_class
        self.current_details = details

        # Update memory
        self.classifier.update_memory(
            details['features'], drift_class, timestamp
        )

        return drift_class, details

    def deactivate(self):
        """
        Deactivate Stage B after classification and adaptation.
        """
        self.active = False
        self.recent_windows.clear()
        self.baseline_windows = None

    def get_state(self) -> Dict:
        """Get classifier state for serialization."""
        return {
            'error_buffer_length': len(self.error_buffer),
            'active': self.active,
            'current_drift_class': (
                DriftClass(self.current_drift_class).name
                if self.current_drift_class is not None else None
            ),
            'statistics': self.classifier.get_statistics()
        }


if __name__ == "__main__":
    # Test drift classifier
    print("=== Testing Drift Classifier ===\n")

    # Create synthetic error streams for different drift types
    np.random.seed(42)

    # Sudden drift: abrupt error increase
    sudden_errors = np.concatenate([
        np.random.binomial(1, 0.1, 50),  # 10% baseline error
        np.random.binomial(1, 0.6, 50)   # 60% after drift
    ])

    # Gradual drift: slow error ramp
    gradual_errors = np.random.binomial(
        1,
        np.linspace(0.1, 0.5, 100),
        100
    )

    # Recurring drift: periodic high error
    recurring_errors = np.zeros(100)
    for i in range(100):
        if (i % 20) < 10:
            recurring_errors[i] = np.random.binomial(1, 0.6)
        else:
            recurring_errors[i] = np.random.binomial(1, 0.1)

    # Test classification
    classifier = DriftClassifier()

    print("Testing Sudden Drift Classification:")
    drift_class, details = classifier.classify(sudden_errors)
    print(f"  Class: {DriftClass(drift_class).name}")
    print(f"  Confidence: {details['confidence']:.3f}")
    print(f"  Reason: {details['reason']}")

    print("\nTesting Gradual Drift Classification:")
    drift_class, details = classifier.classify(gradual_errors)
    print(f"  Class: {DriftClass(drift_class).name}")
    print(f"  Confidence: {details['confidence']:.3f}")
    print(f"  Reason: {details['reason']}")

    print("\nTesting Recurring Drift Classification:")
    # First, save a pattern
    classifier.save_pattern(
        classifier.feature_extractor.extract_all_features(recurring_errors),
        DriftClass.RECURRING,
        timestamp=0
    )
    drift_class, details = classifier.classify(recurring_errors)
    print(f"  Class: {DriftClass(drift_class).name}")
    print(f"  Confidence: {details['confidence']:.3f}")
    print(f"  Reason: {details['reason']}")

    print("\nClassifier Statistics:")
    stats = classifier.get_statistics()
    print(f"  Total classifications: {stats['total_classifications']}")
    print(f"  Class distribution: {stats['class_distribution']}")
