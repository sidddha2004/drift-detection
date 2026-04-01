"""
Phase 1: Windowing and Feature Extraction

Handles:
- Segmenting continuous accelerometer stream into fixed windows
- Computing statistical features per window
- Generating labeled training samples
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import IntEnum


class ActivityID(IntEnum):
    """PAMAP2 Activity IDs."""
    TRANSIENT = 0
    LYING = 1
    SITTING = 2
    STANDING = 3
    WALKING = 4
    RUNNING = 5
    CYCLING = 6
    NORDIC_WALKING = 7
    WATCHING_TV = 9
    COMPUTER_WORK = 10
    CAR_DRIVING = 11
    ASCENDING_STAIRS = 12
    DESCENDING_STAIRS = 13
    VACUUM_CLEANING = 16
    IRONING = 17
    FOLDING_LAUNDRY = 18
    HOUSE_CLEANING = 19
    PLAYING_SOCCER = 20
    ROPE_JUMPING = 24


@dataclass
class WindowingConfig:
    """Configuration for windowing parameters."""
    # Window size: 2.56 seconds @ 100Hz = 256 samples
    window_size: int = 256
    # Stride: 50% overlap = 128 samples
    stride: int = 128
    # Sampling frequency
    sampling_freq: int = 100


class FeatureExtractor:
    """
    Extract statistical features from sliding windows.

    Computes 5 statistics per axis per sensor:
    - Mean, Standard Deviation, Min, Max, 25th/75th Percentile

    Output: 3 axes × 3 sensors × 5 statistics = 45 features
    """

    def __init__(self, config: Optional[WindowingConfig] = None):
        self.config = config or WindowingConfig()

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract features from a single window.

        Args:
            window: Shape (window_size, 9) - 9 axes from 3 sensors

        Returns:
            Feature vector of shape (45,)
            Order: [hand_x_stats, hand_y_stats, hand_z_stats,
                    chest_x_stats, chest_y_stats, chest_z_stats,
                    ankle_x_stats, ankle_y_stats, ankle_z_stats]
            where each *_stats = [mean, std, min, max, p25]
        """
        features = []

        # Extract features for each of the 9 axes
        for axis_idx in range(9):
            axis_data = window[:, axis_idx]

            # Compute 5 statistics per axis
            mean_val = np.mean(axis_data)
            std_val = np.std(axis_data)
            min_val = np.min(axis_data)
            max_val = np.max(axis_data)
            p25_val = np.percentile(axis_data, 25)

            features.extend([mean_val, std_val, min_val, max_val, p25_val])

        return np.array(features, dtype=np.float32)

    def extract_features_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of windows.

        Args:
            windows: Shape (n_windows, window_size, 9)

        Returns:
            Feature matrix of shape (n_windows, 45)
        """
        n_windows = windows.shape[0]
        features = np.zeros((n_windows, 45), dtype=np.float32)

        for i in range(n_windows):
            features[i] = self.extract_features(windows[i])

        return features


class SlidingWindowGenerator:
    """
    Generate sliding windows from continuous time series data.
    """

    def __init__(self, config: Optional[WindowingConfig] = None):
        self.config = config or WindowingConfig()

    def generate_windows(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate sliding windows from continuous data.

        Args:
            data: Shape (n_samples, 9) - accelerometer data
            labels: Shape (n_samples,) - activity labels (optional)

        Returns:
            Tuple of (windows, window_labels)
            - windows: Shape (n_windows, window_size, 9)
            - window_labels: Shape (n_windows,) - most frequent label per window
        """
        n_samples = data.shape[0]
        n_windows = (n_samples - self.config.window_size) // self.config.stride + 1

        # Pre-allocate output arrays
        windows = np.zeros(
            (n_windows, self.config.window_size, 9),
            dtype=np.float32
        )

        # Generate windows
        for i in range(n_windows):
            start_idx = i * self.config.stride
            end_idx = start_idx + self.config.window_size
            windows[i] = data[start_idx:end_idx]

        # Generate window labels (most frequent activity in window)
        window_labels = None
        if labels is not None:
            window_labels = np.zeros(n_windows, dtype=np.int32)
            for i in range(n_windows):
                start_idx = i * self.config.stride
                end_idx = start_idx + self.config.window_size
                # Most frequent label in window
                window_labels[i] = np.bincount(labels[start_idx:end_idx]).argmax()

        return windows, window_labels

    def generate_windows_streaming(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ):
        """
        Generator that yields windows one at a time (memory efficient).

        Args:
            data: Shape (n_samples, 9)
            labels: Shape (n_samples,) (optional)

        Yields:
            Tuple of (window, label) or just window if labels is None
        """
        n_samples = data.shape[0]
        n_windows = (n_samples - self.config.window_size) // self.config.stride + 1

        for i in range(n_windows):
            start_idx = i * self.config.stride
            end_idx = start_idx + self.config.window_size

            window = data[start_idx:end_idx]

            if labels is not None:
                window_label = np.bincount(labels[start_idx:end_idx]).argmax()
                yield window, window_label
            else:
                yield window


class DataStreamPipeline:
    """
    Complete data pipeline: raw data → windows → features → labels.

    Combines preprocessing, windowing, and feature extraction.
    """

    def __init__(
        self,
        windowing_config: Optional[WindowingConfig] = None,
        return_features: bool = True
    ):
        self.window_generator = SlidingWindowGenerator(windowing_config)
        self.feature_extractor = FeatureExtractor(windowing_config)
        self.return_features = return_features

    def process(
        self,
        accel_data: np.ndarray,
        activity_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process raw accelerometer data into features and labels.

        Args:
            accel_data: Shape (n_samples, 9) - preprocessed accelerometer
            activity_labels: Shape (n_samples,) - activity IDs

        Returns:
            Tuple of (features, labels)
            - features: Shape (n_windows, 45)
            - labels: Shape (n_windows,)
        """
        # Generate sliding windows
        windows, window_labels = self.window_generator.generate_windows(
            accel_data, activity_labels
        )

        # Extract features
        if self.return_features:
            features = self.feature_extractor.extract_features_batch(windows)
        else:
            features = windows.reshape(windows.shape[0], -1)  # Flatten windows

        return features, window_labels

    def process_streaming(
        self,
        accel_data: np.ndarray,
        activity_labels: np.ndarray
    ):
        """
        Generator that processes data stream window-by-window.

        Args:
            accel_data: Shape (n_samples, 9)
            activity_labels: Shape (n_samples,)

        Yields:
            Tuple of (features, label)
        """
        for window, label in self.window_generator.generate_windows_streaming(
            accel_data, activity_labels
        ):
            if self.return_features:
                features = self.feature_extractor.extract_features(window)
            else:
                features = window.flatten()
            yield features, label


# Convenience functions
def create_data_pipeline(
    window_size: int = 256,
    stride: int = 128,
    return_features: bool = True
) -> DataStreamPipeline:
    """
    Factory function to create a data pipeline with custom parameters.

    Args:
        window_size: Window size in samples
        stride: Stride in samples
        return_features: Whether to extract statistical features

    Returns:
        Configured DataStreamPipeline
    """
    config = WindowingConfig(
        window_size=window_size,
        stride=stride,
        sampling_freq=100
    )
    return DataStreamPipeline(config, return_features)


if __name__ == "__main__":
    # Test windowing and feature extraction
    import sys
    sys.path.append('..')
    from preprocessing import PAMAP2Preprocessor

    # Load test data
    test_file = "../../PAMAP2_Dataset/Protocol/subject101.dat"

    print("Loading and preprocessing data...")
    preprocessor = PAMAP2Preprocessor()
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)

    print(f"Preprocessed data shape: {accel_data.shape}")

    # Create pipeline
    pipeline = create_data_pipeline(window_size=256, stride=128)

    print("Generating windows and features...")
    features, labels = pipeline.process(accel_data, activity_labels)

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

    # Show class distribution
    print("\nClass distribution:")
    for label_id in np.unique(labels):
        count = np.sum(labels == label_id)
        print(f"  Activity {label_id}: {count} windows ({100*count/len(labels):.1f}%)")
