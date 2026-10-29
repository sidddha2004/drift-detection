"""
Phase 1: Drift Injection Module

Injects artificial drift into data streams to simulate real-world
sensor or behavioral changes for evaluation purposes.

Supports 4 drift types:
- Sudden: Abrupt distribution shift
- Gradual: Slow linear ramp
- Recurring: Periodic drift pattern
- Feature-Level: Axis-specific drift
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum


class DriftType(Enum):
    """Types of drift that can be injected."""
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    RECURRING = "recurring"
    FEATURE_LEVEL = "feature_level"


@dataclass
class DriftConfig:
    """Configuration for drift injection."""
    # Drift magnitude in g-units (standardized accelerometer units)
    magnitude: float = 1.0

    # Drift type
    drift_type: DriftType = DriftType.SUDDEN

    # Injection point (sample index)
    injection_point: int = 5000

    # For gradual drift: ramp duration in samples
    ramp_samples: int = 5000

    # For recurring drift: period and duration in samples
    period: int = 10000
    duration: int = 2000

    # For feature-level drift: which axis to affect (0-8)
    # 0-2: wrist (x,y,z), 3-5: chest (x,y,z), 6-8: ankle (x,y,z)
    target_axis: int = 0  # Default: wrist-x

    # Random seed for reproducibility
    seed: int = 42


class DriftInjector:
    """
    Injects various types of drift into accelerometer data streams.

    All drift is additive: x' = x + drift(t)
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.drift_applied = np.zeros(9)  # Track drift per axis

    def reset(self):
        """Reset drift state."""
        self.drift_applied = np.zeros(9)
        self.rng = np.random.default_rng(self.config.seed)

    def inject_sudden_drift(
        self,
        data: np.ndarray,
        injection_point: int
    ) -> np.ndarray:
        """
        Inject sudden drift: abrupt distribution shift.

        x'(t) = x(t) + δ for t >= injection_point

        Args:
            data: Shape (n_samples, 9) or (n_samples, 256, 9) for LSTM windows
            injection_point: Sample index to start drift

        Returns:
            Data with drift applied
        """
        data = data.copy()
        n_samples = data.shape[0]

        if injection_point >= n_samples:
            return data

        # Apply constant offset to all axes
        drift_offset = self.config.magnitude
        self.drift_applied = np.full(9, drift_offset)

        # Handle both 2D (n_samples, 9) and 3D (n_samples, 256, 9) data
        if len(data.shape) == 3:
            # LSTM windows: (n_samples, 256, 9)
            data[injection_point:] += self.drift_applied[np.newaxis, np.newaxis, :]
        else:
            # 2D features: (n_samples, 9)
            data[injection_point:] += self.drift_applied[np.newaxis, :]

        return data

    def inject_gradual_drift(
        self,
        data: np.ndarray,
        injection_point: int
    ) -> np.ndarray:
        """
        Inject gradual drift: linear ramp over time.

        x'(t) = x(t) + α(t) · δ
        where α(t) = min(1, (t - injection_point) / ramp_samples)

        Args:
            data: Shape (n_samples, 9) or (n_samples, 256, 9) for LSTM windows
            injection_point: Sample index to start drift

        Returns:
            Data with drift applied
        """
        data = data.copy()
        n_samples = data.shape[0]

        if injection_point >= n_samples:
            return data

        # Create ramp function
        ramp_length = min(self.config.ramp_samples, n_samples - injection_point)
        alpha = np.linspace(0, 1, ramp_length)

        # Apply gradually increasing drift
        drift_offset = self.config.magnitude
        self.drift_applied = np.full(9, drift_offset)

        # Handle both 2D (n_samples, 9) and 3D (n_samples, 256, 9) data
        if len(data.shape) == 3:
            # LSTM windows: (n_samples, 256, 9)
            # Reshape alpha for proper broadcasting
            alpha_3d = alpha[:, np.newaxis, np.newaxis]  # (ramp_length, 1, 1)
            data[injection_point:injection_point + ramp_length] += (
                alpha_3d * self.drift_applied[np.newaxis, np.newaxis, :]
            )

            # Full drift after ramp
            if injection_point + ramp_length < n_samples:
                data[injection_point + ramp_length:] += self.drift_applied[np.newaxis, np.newaxis, :]
        else:
            # 2D features: (n_samples, 9)
            data[injection_point:injection_point + ramp_length] += (
                alpha[:, np.newaxis] * self.drift_applied[np.newaxis, :]
            )

            # Full drift after ramp
            if injection_point + ramp_length < n_samples:
                data[injection_point + ramp_length:] += self.drift_applied[np.newaxis, :]

        return data

    def inject_recurring_drift(
        self,
        data: np.ndarray,
        injection_point: int
    ) -> np.ndarray:
        """
        Inject recurring drift: periodic distribution shift.

        x'(t) = x(t) + δ if (t - injection_point) mod period < duration
        x'(t) = x(t) otherwise

        Args:
            data: Shape (n_samples, 9) or (n_samples, 256, 9) for LSTM windows
            injection_point: Sample index to start pattern

        Returns:
            Data with drift applied
        """
        data = data.copy()
        n_samples = data.shape[0]

        if injection_point >= n_samples:
            return data

        # Create time indices relative to injection point
        t = np.arange(injection_point, n_samples)
        relative_t = t - injection_point

        # Determine which samples are in "drift active" phase
        in_drift_phase = (relative_t % self.config.period) < self.config.duration

        # Apply drift during active phases
        drift_offset = self.config.magnitude
        self.drift_applied = np.full(9, drift_offset)

        # Handle both 2D (n_samples, 9) and 3D (n_samples, 256, 9) data
        if len(data.shape) == 3:
            # LSTM windows: (n_samples, 256, 9)
            data[injection_point:][in_drift_phase] += self.drift_applied[np.newaxis, np.newaxis, :]
        else:
            # 2D features: (n_samples, 9)
            data[injection_point:][in_drift_phase] += self.drift_applied[np.newaxis, :]

        return data

    def inject_feature_level_drift(
        self,
        data: np.ndarray,
        injection_point: int
    ) -> np.ndarray:
        """
        Inject feature-level drift: axis-specific distribution shift.

        x'(t) = x(t) + δ for target_axis only
        x'(t) = x(t) for other axes

        Args:
            data: Shape (n_samples, 9) or (n_samples, 256, 9) for LSTM windows
            injection_point: Sample index to start drift

        Returns:
            Data with drift applied
        """
        data = data.copy()
        n_samples = data.shape[0]

        if injection_point >= n_samples:
            return data

        # Apply drift to target axis only
        drift_offset = self.config.magnitude

        # Track drift for this axis
        self.drift_applied = np.zeros(9)
        self.drift_applied[self.config.target_axis] = drift_offset

        # Handle both 2D (n_samples, 9) and 3D (n_samples, 256, 9) data
        if len(data.shape) == 3:
            # LSTM windows: (n_samples, 256, 9)
            data[injection_point:, :, self.config.target_axis] += drift_offset
        else:
            # 2D features: (n_samples, 9)
            data[injection_point:, self.config.target_axis] += drift_offset

        return data

    def inject_drift(
        self,
        data: np.ndarray,
        injection_point: Optional[int] = None
    ) -> np.ndarray:
        """
        Inject drift based on configured type.

        Args:
            data: Shape (n_samples, 9)
            injection_point: Override config injection point (optional)

        Returns:
            Data with drift applied

        Raises:
            ValueError: If drift_type is unknown
        """
        inj_point = injection_point or self.config.injection_point

        if self.config.drift_type == DriftType.SUDDEN:
            return self.inject_sudden_drift(data, inj_point)
        elif self.config.drift_type == DriftType.GRADUAL:
            return self.inject_gradual_drift(data, inj_point)
        elif self.config.drift_type == DriftType.RECURRING:
            return self.inject_recurring_drift(data, inj_point)
        elif self.config.drift_type == DriftType.FEATURE_LEVEL:
            return self.inject_feature_level_drift(data, inj_point)
        else:
            raise ValueError(f"Unknown drift type: {self.config.drift_type}")

    def get_drift_mask(
        self,
        n_samples: int,
        injection_point: Optional[int] = None
    ) -> np.ndarray:
        """
        Get boolean mask indicating which samples have drift applied.

        Useful for evaluation (knowing ground truth drift periods).

        Args:
            n_samples: Total number of samples
            injection_point: Override config injection point

        Returns:
            Boolean array of shape (n_samples,)
        """
        inj_point = injection_point or self.config.injection_point
        mask = np.zeros(n_samples, dtype=bool)

        if inj_point >= n_samples:
            return mask

        if self.config.drift_type == DriftType.SUDDEN:
            mask[inj_point:] = True

        elif self.config.drift_type == DriftType.GRADUAL:
            mask[inj_point:] = True  # All samples after injection affected

        elif self.config.drift_type == DriftType.RECURRING:
            t = np.arange(inj_point, n_samples)
            relative_t = t - inj_point
            mask[inj_point:] = (relative_t % self.config.period) < self.config.duration

        elif self.config.drift_type == DriftType.FEATURE_LEVEL:
            mask[inj_point:] = True

        return mask


class MultiDriftInjector:
    """
    Inject multiple drift events into a single data stream.

    Supports sequential drift events of different types.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.drift_events: List[Dict] = []
        self.ground_truth: Optional[np.ndarray] = None

    def add_drift_event(
        self,
        drift_type: DriftType,
        magnitude: float,
        injection_point: int,
        target_axis: Optional[int] = None,
        duration: Optional[int] = None,
        period: Optional[int] = None,
        ramp_samples: Optional[int] = None
    ):
        """
        Schedule a drift event.

        Args:
            drift_type: Type of drift
            magnitude: Drift magnitude in g-units
            injection_point: Sample index to start drift
            target_axis: For feature-level drift (0-8)
            duration: For recurring drift (active duration)
            period: For recurring drift (period length)
            ramp_samples: For gradual drift (ramp duration)
        """
        self.drift_events.append({
            'drift_type': drift_type,
            'magnitude': magnitude,
            'injection_point': injection_point,
            'target_axis': target_axis,
            'duration': duration,
            'period': period,
            'ramp_samples': ramp_samples
        })

    def inject_all(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject all scheduled drift events into data.

        Args:
            data: Shape (n_samples, 9)

        Returns:
            Tuple of (drifted_data, drift_labels)
            - drifted_data: Data with all drift applied
            - drift_labels: Shape (n_samples,) - 0=no drift, 1+=drift_type_id
        """
        drifted_data = data.copy()
        n_samples = data.shape[0]

        # Initialize drift labels (0 = no drift)
        drift_labels = np.zeros(n_samples, dtype=np.int32)

        for i, event in enumerate(self.drift_events):
            config = DriftConfig(
                drift_type=event['drift_type'],
                magnitude=event['magnitude'],
                injection_point=event['injection_point'],
                target_axis=event.get('target_axis', 0),
                duration=event.get('duration', 2000),
                period=event.get('period', 10000),
                ramp_samples=event.get('ramp_samples', 5000)
            )

            injector = DriftInjector(config)
            drifted_data = injector.inject_drift(drifted_data)

            # Update ground truth labels
            mask = injector.get_drift_mask(n_samples)
            drift_labels[mask] = i + 1  # 1-indexed drift event ID

        self.ground_truth = drift_labels
        return drifted_data, drift_labels

    def reset(self):
        """Reset all scheduled drift events."""
        self.drift_events = []
        self.ground_truth = None


def create_standard_drift_scenario(
    seed: int = 42
) -> MultiDriftInjector:
    """
    Create a standard drift scenario for reproducible experiments.

    Scenario:
    - Sample 5000: Sudden drift (1.0g)
    - Sample 15000: Gradual drift (0.5g, 5000 sample ramp)
    - Sample 25000: Recurring drift (1.0g, 10000 period, 2000 duration)

    Args:
        seed: Random seed

    Returns:
        Configured MultiDriftInjector
    """
    injector = MultiDriftInjector(seed=seed)

    injector.add_drift_event(
        drift_type=DriftType.SUDDEN,
        magnitude=1.0,
        injection_point=5000
    )

    injector.add_drift_event(
        drift_type=DriftType.GRADUAL,
        magnitude=0.5,
        injection_point=15000,
        ramp_samples=5000
    )

    injector.add_drift_event(
        drift_type=DriftType.RECURRING,
        magnitude=1.0,
        injection_point=25000,
        duration=2000,
        period=10000
    )

    return injector


if __name__ == "__main__":
    # Test drift injection
    import sys
    sys.path.append('..')
    from preprocessing import PAMAP2Preprocessor
    from windowing import create_data_pipeline

    # Load test data
    test_file = "../../PAMAP2_Dataset/Protocol/subject101.dat"

    print("Loading data...")
    preprocessor = PAMAP2Preprocessor()
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)

    print(f"Original data shape: {accel_data.shape}")
    print(f"Original data range: [{accel_data.min():.3f}, {accel_data.max():.3f}]")

    # Test sudden drift
    print("\n=== Testing Sudden Drift ===")
    config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        magnitude=1.0,
        injection_point=5000
    )
    injector = DriftInjector(config)
    drifted_data = injector.inject_drift(accel_data.copy())

    print(f"Drifted data range: [{drifted_data.min():.3f}, {drifted_data.max():.3f}]")
    print(f"Mean before injection: {acceler_data[:5000].mean():.3f}")
    print(f"Mean after injection: {drifted_data[5000:].mean():.3f}")
    print(f"Drift applied per axis: {injector.drift_applied}")

    # Test standard scenario
    print("\n=== Testing Standard Multi-Drift Scenario ===")
    multi_injector = create_standard_drift_scenario()
    multi_drifted, drift_labels = multi_injector.inject_all(accel_data.copy())

    print(f"Number of drift events: {len(multi_injector.drift_events)}")
    print(f"Drift labels distribution: {np.bincount(drift_labels)}")

    # Visualize drift effect (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        # Original data (wrist-x axis)
        axes[0].plot(accel_data[:10000, 0], label='Original', alpha=0.7)
        axes[0].axvline(x=5000, color='r', linestyle='--', label='Drift @ 5000')
        axes[0].set_ylabel('Acceleration (g)')
        axes[0].legend()
        axes[0].set_title('Sudden Drift Injection')

        # Drifted data
        axes[1].plot(drifted_data[:10000, 0], label='Drifted', alpha=0.7)
        axes[1].axvline(x=5000, color='r', linestyle='--')
        axes[1].set_ylabel('Acceleration (g)')
        axes[1].legend()
        axes[1].set_title('After Sudden Drift')

        # Multi-drift scenario
        axes[2].plot(multi_drifted[:30000, 0], label='Multi-Drifted', alpha=0.7)
        axes[2].axvline(x=5000, color='r', linestyle='--', label='Sudden')
        axes[2].axvline(x=15000, color='orange', linestyle='--', label='Gradual')
        axes[2].axvline(x=25000, color='green', linestyle='--', label='Recurring')
        axes[2].set_ylabel('Acceleration (g)')
        axes[2].set_xlabel('Sample Index')
        axes[2].legend()
        axes[2].set_title('Standard Multi-Drift Scenario')

        plt.tight_layout()
        plt.savefig('../../results/drift_injection_test.png')
        print("\nPlot saved to results/drift_injection_test.png")

    except ImportError:
        print("\nMatplotlib not available - skipping visualization")
