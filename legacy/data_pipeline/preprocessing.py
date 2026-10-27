"""
Phase 1: Data Preprocessing for PAMAP2 Dataset

Handles:
- Loading .dat files from PAMAP2 dataset
- Filtering transient activities (activityID = 0)
- Interpolating missing values (NaN from wireless dropout)
- Running Z-score normalization per subject
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Columns for IMU sensors (1-indexed in paper, 0-indexed here)
    # IMU hand: columns 3-19 (0-indexed: 3-19)
    # IMU chest: columns 20-36
    # IMU ankle: columns 37-53

    # Accelerometer columns (±16g scale - recommended over ±6g)
    accel_columns_hand = [3, 4, 5]    # 3D acceleration, ±16g
    accel_columns_chest = [20, 21, 22]
    accel_columns_ankle = [37, 38, 39]

    # Activity ID and heart rate
    timestamp_col = 0
    activity_col = 1
    heart_rate_col = 2

    # Normalization parameters
    decay: float = 0.95  # Running mean/variance decay
    epsilon: float = 1e-8  # Numerical stability


class PAMAP2Preprocessor:
    """
    Preprocessor for PAMAP2 dataset.

    Handles loading, filtering, interpolation, and normalization
    of raw .dat files from the PAMAP2 dataset.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None
        self.is_fitted = False

    def load_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a PAMAP2 .dat file into a DataFrame.

        Args:
            filepath: Path to subjectXXX.dat file

        Returns:
            DataFrame with 54 columns as per PAMAP2 spec
        """
        # Load space-delimited file
        df = pd.read_csv(
            filepath,
            sep='\\s+',
            header=None,
            na_values=['NaN']  # Convert 'NaN' strings to actual NaN
        )
        return df

    def filter_transient_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with activityID = 0 (transient activities).

        From PAMAP2 readme: "data labeled with activityID=0 should be
        discarded in any kind of analysis. This data mainly covers
        transient activities between performing different activities."

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with activityID != 0 only
        """
        return df[df[self.config.activity_col] != 0].copy()

    def interpolate_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing IMU values using linear interpolation.

        Missing data occurs due to:
        - Wireless sensor dropout (~0.4% of samples)
        - Heart rate monitor sync issues (9Hz vs 100Hz)

        Args:
            df: DataFrame with NaN values

        Returns:
            DataFrame with interpolated values
        """
        # Create a copy to avoid SettingWithCopy warnings
        df = df.copy()

        # Get all IMU columns (3-53) - excludes heart rate (col 2) which is sparse
        imu_cols = list(range(3, 54))

        # Linear interpolation for IMU data - interpolate in-place on the slice
        df[imu_cols] = df[imu_cols].interpolate(
            method='linear',
            limit_direction='both'  # Handle NaN at edges
        )

        # Fill any remaining NaN at edges with forward/backward fill
        df[imu_cols] = df[imu_cols].ffill().bfill()

        # Also interpolate heart rate column (col 2) - sparse 9Hz data
        df[2] = df[2].interpolate(method='linear', limit_direction='both').ffill().bfill()

        return df

    def extract_accelerometer_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract only accelerometer data (±16g scale) from DataFrame.

        Returns 9 columns: 3 axes × 3 sensors (hand, chest, ankle)

        Args:
            df: Preprocessed DataFrame

        Returns:
            numpy array of shape (n_samples, 9)
        """
        accel_data = df[
            self.config.accel_columns_hand +
            self.config.accel_columns_chest +
            self.config.accel_columns_ankle
        ].values
        return accel_data

    def fit_running_normalization(self, accel_data: np.ndarray):
        """
        Initialize running normalization statistics on initial data.

        Uses exponential moving average for mean and variance:
        μ(t) = α·μ(t-1) + (1-α)·x(t)
        σ²(t) = α·σ²(t-1) + (1-α)·(x(t) - μ(t))²

        Args:
            accel_data: Initial accelerometer data for warmstart
        """
        # Initialize with first batch statistics
        self.running_mean = np.mean(accel_data, axis=0)
        self.running_var = np.var(accel_data, axis=0)
        self.is_fitted = True

    def normalize_batch(self, accel_data: np.ndarray) -> np.ndarray:
        """
        Apply running Z-score normalization to a batch of data.

        Updates running statistics incrementally.

        Args:
            accel_data: Batch of shape (n_samples, 9)

        Returns:
            Normalized data of same shape
        """
        if not self.is_fitted:
            self.fit_running_normalization(accel_data)

        # Normalize using current running stats
        normalized = (accel_data - self.running_mean) / np.sqrt(
            self.running_var + self.config.epsilon
        )

        # Update running statistics with new batch
        batch_mean = np.mean(accel_data, axis=0)
        batch_var = np.var(accel_data, axis=0)

        # Exponential moving average update
        self.running_mean = (
            self.config.decay * self.running_mean +
            (1 - self.config.decay) * batch_mean
        )
        self.running_var = (
            self.config.decay * self.running_var +
            (1 - self.config.decay) * batch_var
        )

        return normalized

    def preprocess_file(
        self,
        filepath: str,
        return_activity_labels: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full preprocessing pipeline for a single subject file.

        Args:
            filepath: Path to .dat file
            return_activity_labels: Whether to return activity IDs

        Returns:
            Tuple of (accelerometer_data, activity_labels)
            - accelerometer_data: shape (n_samples, 9) normalized
            - activity_labels: shape (n_samples,) or None
        """
        # Load raw data
        df = self.load_file(filepath)

        # Filter out transient activities
        df = self.filter_transient_activities(df)

        # Interpolate missing values
        df = self.interpolate_missing_values(df)

        # Extract accelerometer data
        accel_data = self.extract_accelerometer_data(df)

        # Apply running normalization
        accel_data = self.normalize_batch(accel_data)

        # Get activity labels if requested
        activity_labels = None
        if return_activity_labels:
            activity_labels = df[self.config.activity_col].values

        return accel_data, activity_labels


def load_subject_data(
    data_dir: str,
    subject_ids: list,
    protocol: bool = True
) -> dict:
    """
    Load and preprocess data for multiple subjects.

    Args:
        data_dir: Base directory containing PAMAP2_Dataset/
        subject_ids: List of subject IDs to load (e.g., [101, 105, 106])
        protocol: Load Protocol data (True) or Optional data (False)

    Returns:
        Dict mapping subject_id -> (accel_data, activity_labels)
    """
    dataset_subdir = "Protocol" if protocol else "Optional"
    base_path = f"{data_dir}/PAMAP2_Dataset/{dataset_subdir}"

    preprocessor = PAMAP2Preprocessor()
    subject_data = {}

    for subject_id in subject_ids:
        filepath = f"{base_path}/subject{subject_id}.dat"
        try:
            accel_data, activity_labels = preprocessor.preprocess_file(filepath)
            subject_data[subject_id] = (accel_data, activity_labels)
            print(f"Loaded subject {subject_id}: {accel_data.shape[0]} samples")
        except FileNotFoundError:
            print(f"Warning: Subject {subject_id} not found at {filepath}")

    return subject_data


# Convenience function for testing
if __name__ == "__main__":
    # Test preprocessing on a single file
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Default test path
        test_file = "PAMAP2_Dataset/Protocol/subject101.dat"

    print(f"Loading: {test_file}")

    preprocessor = PAMAP2Preprocessor()
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)

    print(f"Accelerometer data shape: {accel_data.shape}")
    print(f"Activity labels shape: {activity_labels.shape if activity_labels is not None else None}")
    print(f"Unique activities: {np.unique(activity_labels)}")
    print(f"Data range: [{accel_data.min():.3f}, {accel_data.max():.3f}]")
    print(f"Mean (should be ~0): {accel_data.mean():.6f}")
    print(f"Std (should be ~1): {accel_data.std():.6f}")
