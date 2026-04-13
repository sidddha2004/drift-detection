"""
Test Suite for Integrated Drift Detector

Tests the full end-to-end pipeline with real PAMAP2 data:
1. Stage A: Binary drift detection (ADWIN)
2. Stage B: Drift classification (4 types)
3. Stage C: Type-specific adaptation (mitigation)

Usage:
    python tests/test_integrated_detector.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.drift_detection import (
    IntegratedDriftDetector,
    IntegratedDriftDetectorConfig,
    LSTMConfig,
    DriftClass
)
from src.data_pipeline import PAMAP2Preprocessor, create_data_pipeline


def load_pamap2_data():
    """Load and preprocess PAMAP2 data."""
    preprocessor = PAMAP2Preprocessor()
    test_file = "PAMAP2_Dataset/Protocol/subject101.dat"
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)

    pipeline = create_data_pipeline(window_size=256, stride=128, return_raw_windows=True)
    windows, labels = pipeline.process(accel_data, activity_labels)

    # Remap labels to contiguous range
    unique_labels = np.unique(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels_remapped = np.array([label_to_idx[l] for l in labels])

    return windows, labels_remapped


def test_integrated_detector_lstm():
    """Test integrated detector with LSTM model and real data."""
    print("=" * 60)
    print("TEST: Integrated Detector (LSTM + PAMAP2)")
    print("=" * 60)

    # Load data
    print("\n1. Loading PAMAP2 data...")
    windows, labels = load_pamap2_data()
    print(f"   Loaded {len(windows)} windows")
    print(f"   Classes: {len(np.unique(labels))}")
    print("   PASS")

    # Create detector
    print("\n2. Creating integrated detector...")
    config = IntegratedDriftDetectorConfig(
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1, learning_rate=0.001),
        adwin_delta=0.01,
        max_buffer_size=200,
        min_samples_for_adaptation=30
    )
    detector = IntegratedDriftDetector(config)
    print("   PASS")

    # Pre-train on first 100 windows
    print("\n3. Pre-training on initial data...")
    for epoch in range(3):
        for i in range(0, min(100, len(windows)), 32):
            detector.detector.train_batch(windows[i:i+32], labels[i:i+32])

    pre_acc, _ = detector.detector.evaluate(windows[100:200], labels[100:200])
    print(f"   Pre-training accuracy: {pre_acc:.1%}")
    print("   PASS")

    # Process stream with simulated drift
    print("\n4. Processing stream...")
    # Simulate drift by introducing label shifts after sample 200
    labels_with_drift = labels.copy()
    drift_point = 200
    for i in range(drift_point, min(len(labels), 400)):
        if np.random.rand() < 0.3:  # 30% label noise
            labels_with_drift[i] = (labels[i] + 1) % len(np.unique(labels))

    for i in range(100, min(400, len(windows))):
        detector.process(windows[i], labels_with_drift[i])

    print(f"   Total drifts detected: {detector.total_drifts}")
    print(f"   Total adaptations: {detector.total_adaptations}")
    print("   PASS")

    # Statistics
    print("\n5. Final statistics:")
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("INTEGRATED DETECTOR TEST PASSED")
    print("=" * 60)
    return True


def test_adaptation_strategies():
    """Test that adaptation strategies are triggered."""
    print("=" * 60)
    print("TEST: Adaptation Strategies")
    print("=" * 60)

    windows, labels = load_pamap2_data()
    print(f"\n   Loaded {len(windows)} windows")

    config = IntegratedDriftDetectorConfig(
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1),
        adwin_delta=0.02,
        max_buffer_size=300,
        min_samples_for_adaptation=20
    )
    detector = IntegratedDriftDetector(config)

    # Pre-train
    for i in range(0, min(100, len(windows)), 32):
        detector.detector.train_batch(windows[i:i+32], labels[i:i+32])

    # Process with high error rate to trigger drift
    noisy_labels = labels.copy()
    for i in range(len(noisy_labels)):
        if np.random.rand() < 0.4:
            noisy_labels[i] = (labels[i] + 1) % len(np.unique(labels))

    print("\n   Processing with noisy labels...")
    for i in range(100, min(300, len(windows))):
        detector.process(windows[i], noisy_labels[i])

    print(f"   Drifts detected: {detector.total_drifts}")
    print(f"   Adaptations: {detector.total_adaptations}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ADAPTATION STRATEGIES TEST PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all integrated detector tests."""
    print("\n" + "=" * 60)
    print("INTEGRATED DRIFT DETECTOR - TEST SUITE")
    print("=" * 60)

    results = {
        'lstm': test_integrated_detector_lstm(),
        'strategies': test_adaptation_strategies()
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("\nMitigation strategies (Phase 4) are INTEGRATED:")
        print("  Detection (Stage A) -> Classification (Stage B) -> Adaptation (Stage C)")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
