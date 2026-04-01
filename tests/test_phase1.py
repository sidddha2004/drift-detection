"""
Test Suite for Phase 1: Data Streaming Pipeline

Tests:
- Preprocessing (loading, filtering, interpolation, normalization)
- Windowing (sliding windows, feature extraction)
- Drift Injection (all 4 drift types)

Usage:
    python tests/test_preprocessing.py
    python tests/test_windowing.py
    python tests/test_drift_injection.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data_pipeline import (
    PAMAP2Preprocessor,
    PreprocessingConfig,
    FeatureExtractor,
    SlidingWindowGenerator,
    create_data_pipeline,
    DriftInjector,
    DriftConfig,
    DriftType,
    MultiDriftInjector,
    create_standard_drift_scenario
)


def test_preprocessing():
    """Test Phase 1 preprocessing module."""
    print("=" * 60)
    print("TEST: Preprocessing Module")
    print("=" * 60)

    # Find a data file
    base_dirs = [
        "PAMAP2_Dataset/Protocol",
        "../PAMAP2_Dataset/Protocol",
        "../../PAMAP2_Dataset/Protocol"
    ]

    test_file = None
    for base in base_dirs:
        candidate = os.path.join(base, "subject101.dat")
        if os.path.exists(candidate):
            test_file = candidate
            break

    if test_file is None:
        print("SKIP: PAMAP2 dataset not found. Download from:")
        print("  https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring")
        return False

    print(f"Using data file: {test_file}")

    # Initialize preprocessor
    preprocessor = PAMAP2Preprocessor()

    # Test loading
    print("\n1. Loading data...")
    df = preprocessor.load_file(test_file)
    print(f"   Raw data shape: {df.shape}")
    print(f"   Expected: (n_samples, 54)")
    assert df.shape[1] == 54, "Expected 54 columns"
    print("   PASS")

    # Test filtering
    print("\n2. Filtering transient activities...")
    df_filtered = preprocessor.filter_transient_activities(df)
    n_transient = len(df) - len(df_filtered)
    print(f"   Removed {n_transient} transient samples")
    assert 0 not in df_filtered[1].values, "Activity ID 0 should be filtered"
    print("   PASS")

    # Test interpolation
    print("\n3. Interpolating missing values...")
    df_interp = preprocessor.interpolate_missing_values(df_filtered)
    nan_count_before = df_filtered.isna().sum().sum()
    nan_count_after = df_interp.isna().sum().sum()
    print(f"   NaN before: {nan_count_before}, after: {nan_count_after}")
    assert nan_count_after == 0, "All NaN should be interpolated"
    print("   PASS")

    # Test full pipeline
    print("\n4. Running full preprocessing pipeline...")
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)
    print(f"   Accelerometer shape: {accel_data.shape}")
    print(f"   Activity labels shape: {activity_labels.shape}")
    print(f"   Unique activities: {np.unique(activity_labels)}")
    print("   PASS")

    # Test normalization
    print("\n5. Verifying normalization...")
    mean = accel_data.mean()
    std = accel_data.std()
    print(f"   Mean: {mean:.6f} (expected ~0)")
    print(f"   Std: {std:.6f} (expected ~1)")
    assert abs(mean) < 0.1, "Mean should be close to 0"
    assert 0.5 < std < 2.0, "Std should be close to 1"
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL PREPROCESSING TESTS PASSED")
    print("=" * 60)
    return True


def test_windowing():
    """Test Phase 1 windowing module."""
    print("\n" + "=" * 60)
    print("TEST: Windowing Module")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    n_samples = 1000
    n_axes = 9
    np.random.seed(42)
    accel_data = np.random.randn(n_samples, n_axes)
    activity_labels = np.random.randint(1, 18, n_samples)
    print(f"   Generated {n_samples} samples, {n_axes} axes")
    print("   PASS")

    # Test window generator
    print("\n2. Testing sliding window generation...")
    window_gen = SlidingWindowGenerator()
    windows, window_labels = window_gen.generate_windows(accel_data, activity_labels)

    expected_windows = (n_samples - window_gen.config.window_size) // window_gen.config.stride + 1
    print(f"   Windows shape: {windows.shape}")
    print(f"   Expected windows: {expected_windows}")
    assert windows.shape[0] == expected_windows, "Window count mismatch"
    assert windows.shape[1] == 256, "Window size should be 256"
    assert windows.shape[2] == 9, "Should have 9 axes"
    print("   PASS")

    # Test feature extractor
    print("\n3. Testing feature extraction...")
    extractor = FeatureExtractor()
    features = extractor.extract_features_batch(windows)
    print(f"   Features shape: {features.shape}")
    print(f"   Expected: ({expected_windows}, 45)")
    assert features.shape == (expected_windows, 45), "Feature dimension mismatch"
    print("   PASS")

    # Test full pipeline
    print("\n4. Testing full data pipeline...")
    pipeline = create_data_pipeline(window_size=256, stride=128)
    features, labels = pipeline.process(accel_data, activity_labels)
    print(f"   Pipeline output - Features: {features.shape}, Labels: {labels.shape}")
    assert features.shape[0] == labels.shape[0], "Feature/label count mismatch"
    print("   PASS")

    # Test class distribution
    print("\n5. Checking class distribution...")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   Classes: {unique}")
    print(f"   Counts: {counts}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL WINDOWING TESTS PASSED")
    print("=" * 60)
    return True


def test_drift_injection():
    """Test Phase 1 drift injection module."""
    print("\n" + "=" * 60)
    print("TEST: Drift Injection Module")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    n_samples = 10000
    np.random.seed(42)
    accel_data = np.random.randn(n_samples, 9)
    print(f"   Generated {n_samples} samples")
    print("   PASS")

    # Test Sudden drift
    print("\n2. Testing Sudden drift injection...")
    config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        magnitude=1.0,
        injection_point=5000
    )
    injector = DriftInjector(config)
    drifted = injector.inject_drift(accel_data.copy())

    mean_before = accel_data[:5000].mean()
    mean_after = drifted[:5000].mean()
    mean_after_drift = drifted[5000:].mean()

    print(f"   Mean before injection: {mean_before:.3f}")
    print(f"   Mean after injection (pre-drift): {mean_after_drift:.3f}")
    print(f"   Drift offset: {mean_after_drift - mean_before:.3f} (expected ~1.0)")
    assert abs((mean_after_drift - mean_before) - 1.0) < 0.1, "Sudden drift magnitude incorrect"
    print("   PASS")

    # Test Gradual drift
    print("\n3. Testing Gradual drift injection...")
    config = DriftConfig(
        drift_type=DriftType.GRADUAL,
        magnitude=1.0,
        injection_point=5000,
        ramp_samples=5000
    )
    injector = DriftInjector(config)
    drifted = injector.inject_drift(accel_data.copy())

    # Check ramp: should be partial at midpoint
    mean_ramp_mid = drifted[7500:8000].mean()
    mean_ramp_end = drifted[9500:10000].mean()
    print(f"   Mean at ramp midpoint (7500): {mean_ramp_mid:.3f}")
    print(f"   Mean at ramp end (9500+): {mean_ramp_end:.3f}")
    assert mean_ramp_end > mean_ramp_mid, "Gradual drift should increase over time"
    print("   PASS")

    # Test Recurring drift
    print("\n4. Testing Recurring drift injection...")
    config = DriftConfig(
        drift_type=DriftType.RECURRING,
        magnitude=1.0,
        injection_point=5000,
        period=2000,
        duration=1000
    )
    injector = DriftInjector(config)
    drifted = injector.inject_drift(accel_data.copy())
    mask = injector.get_drift_mask(n_samples)

    mean_in_drift = drifted[mask].mean()
    mean_out_drift = drifted[~mask].mean()
    print(f"   Mean during drift phase: {mean_in_drift:.3f}")
    print(f"   Mean outside drift phase: {mean_out_drift:.3f}")
    print(f"   Difference: {mean_in_drift - mean_out_drift:.3f} (expected ~1.0)")
    assert abs((mean_in_drift - mean_out_drift) - 1.0) < 0.2, "Recurring drift magnitude incorrect"
    print("   PASS")

    # Test Feature-Level drift
    print("\n5. Testing Feature-Level drift injection...")
    config = DriftConfig(
        drift_type=DriftType.FEATURE_LEVEL,
        magnitude=1.0,
        injection_point=5000,
        target_axis=0  # wrist-x
    )
    injector = DriftInjector(config)
    drifted = injector.inject_drift(accel_data.copy())

    mean_target_before = accel_data[5000:, 0].mean()
    mean_target_after = drifted[5000:, 0].mean()
    mean_other_after = drifted[5000:, 1].mean()

    print(f"   Target axis (0) mean shift: {mean_target_after - mean_target_before:.3f}")
    print(f"   Other axis (1) mean shift: {mean_other_after - accel_data[5000:, 1].mean():.3f}")
    assert abs((mean_target_after - mean_target_before) - 1.0) < 0.1, "Feature-level drift magnitude incorrect"
    assert abs(mean_other_after - accel_data[5000:, 1].mean()) < 0.01, "Other axes should not be affected"
    print("   PASS")

    # Test Multi-Drift scenario
    print("\n6. Testing Multi-Drift scenario...")
    multi_injector = create_standard_drift_scenario()
    multi_drifted, drift_labels = multi_injector.inject_all(accel_data.copy())

    unique_drifts, counts = np.unique(drift_labels, return_counts=True)
    print(f"   Drift labels: {dict(zip(unique_drifts, counts))}")
    assert len(multi_injector.drift_events) == 3, "Should have 3 drift events"
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL DRIFT INJECTION TESTS PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all Phase 1 tests."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA STREAMING PIPELINE - TEST SUITE")
    print("=" * 60)

    results = {
        'preprocessing': test_preprocessing(),
        'windowing': test_windowing(),
        'drift_injection': test_drift_injection()
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED OR SKIPPED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
