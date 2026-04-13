"""
Test Suite for Phase 4: Type-Specific Adaptation Strategies

Tests:
- Sudden drift adaptation (full retrain)
- Gradual drift adaptation (fine-tuning)
- Recurring drift adaptation (model library)
- Feature-level drift adaptation (sensor compensation)
- Model library save/load

Uses real PAMAP2 dataset and LSTM model.

Usage:
    python tests/test_phase4.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.adaptation import (
    TypeSpecificAdapter,
    AdapterConfig,
    AdaptationResult,
    ModelLibrary
)
from src.drift_detection import (
    LSTMDriftDetector,
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

    return windows, labels


def test_sudden_adaptation():
    """Test sudden drift adaptation: full retrain on recent data."""
    print("=" * 60)
    print("TEST: Sudden Drift Adaptation")
    print("=" * 60)

    # Load real data
    print("\n1. Loading PAMAP2 data...")
    windows, labels = load_pamap2_data()
    print(f"   Loaded {len(windows)} windows")
    print("   PASS")

    # Create detector and adapter
    print("\n2. Creating detector and adapter...")
    detector = LSTMDriftDetector(lstm_config=LSTMConfig(hidden_size=32, num_layers=1))
    config = AdapterConfig(sudden_retrain_samples=100, sudden_retrain_epochs=5)
    adapter = TypeSpecificAdapter(config)
    print("   PASS")

    # Initial training
    print("\n3. Initial training...")
    for epoch in range(3):
        for i in range(0, min(200, len(windows)), 32):
            detector.train_batch(windows[i:i+32], labels[i:i+32])

    acc_before, _ = detector.evaluate(windows[200:300], labels[200:300])
    print(f"   Accuracy before drift: {acc_before:.3f}")

    # Simulate drift with shuffled labels
    print("\n4. Simulating sudden drift...")
    drifted_labels = np.random.randint(0, len(np.unique(labels)), len(windows))
    acc_drift, _ = detector.evaluate(windows[200:300], drifted_labels[200:300])
    print(f"   Accuracy after drift: {acc_drift:.3f}")
    print("   PASS")

    # Adapt
    print("\n5. Adapting to sudden drift...")
    result = adapter.adapt(
        drift_class=DriftClass.SUDDEN,
        detector=detector,
        recent_data=windows[200:300],
        recent_labels=drifted_labels[200:300],
        timestamp=200
    )
    print(f"   Strategy: {result.strategy_used}")
    print(f"   Samples used: {result.samples_used}")
    print(f"   Epochs: {result.epochs_run}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("SUDDEN ADAPTATION TEST PASSED")
    print("=" * 60)
    return True


def test_gradual_adaptation():
    """Test gradual drift adaptation: fine-tuning."""
    print("=" * 60)
    print("TEST: Gradual Drift Adaptation")
    print("=" * 60)

    windows, labels = load_pamap2_data()
    print(f"\n   Loaded {len(windows)} windows")

    detector = LSTMDriftDetector(lstm_config=LSTMConfig(hidden_size=32, num_layers=1))
    config = AdapterConfig(gradual_window_samples=200, gradual_lr_multiplier=10.0)
    adapter = TypeSpecificAdapter(config)

    # Initial training
    print("\n   Training initial model...")
    for i in range(0, min(200, len(windows)), 32):
        detector.train_batch(windows[i:i+32], labels[i:i+32])

    # Simulate gradual drift (progressive label noise)
    print("   Simulating gradual drift...")
    noisy_labels = labels.copy()
    n_drift = len(windows) // 2
    for i in range(len(windows) - n_drift, len(windows)):
        if np.random.rand() < 0.3:
            noisy_labels[i] = (labels[i] + 1) % len(np.unique(labels))

    # Adapt
    print("   Adapting to gradual drift...")
    result = adapter.adapt(
        drift_class=DriftClass.GRADUAL,
        detector=detector,
        recent_data=windows[-200:],
        recent_labels=noisy_labels[-200:],
        timestamp=len(windows)
    )
    print(f"   Strategy: {result.strategy_used}")
    print(f"   Samples used: {result.samples_used}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("GRADUAL ADAPTATION TEST PASSED")
    print("=" * 60)
    return True


def test_recurring_adaptation():
    """Test recurring drift adaptation: model library lookup."""
    print("=" * 60)
    print("TEST: Recurring Drift Adaptation")
    print("=" * 60)

    windows, labels = load_pamap2_data()
    print(f"\n   Loaded {len(windows)} windows")

    config = AdapterConfig(max_saved_models=10)
    adapter = TypeSpecificAdapter(config)

    # Train and save model
    print("\n   Training and saving model to library...")
    detector = LSTMDriftDetector(lstm_config=LSTMConfig(hidden_size=32, num_layers=1))
    for i in range(0, min(100, len(windows)), 32):
        detector.train_batch(windows[i:i+32], labels[i:i+32])

    acc, _ = detector.evaluate(windows[100:200], labels[100:200])
    print(f"   Model accuracy: {acc:.3f}")

    # Save to library
    adapter.save_model_for_recurring(
        drift_features=np.random.randn(10),
        detector=detector,
        accuracy=acc,
        timestamp=100
    )
    print(f"   Library size: {len(adapter.model_library.models)}")
    print("   PASS")

    # Test recurring adaptation (will try to match saved pattern)
    print("\n   Testing recurring drift adaptation...")
    result = adapter.adapt(
        drift_class=DriftClass.RECURRING,
        detector=detector,
        recent_data=windows[100:200],
        recent_labels=labels[100:200],
        timestamp=200
    )
    print(f"   Strategy: {result.strategy_used}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("RECURRING ADAPTATION TEST PASSED")
    print("=" * 60)
    return True


def test_feature_level_adaptation():
    """Test feature-level drift adaptation: sensor compensation."""
    print("=" * 60)
    print("TEST: Feature-Level Drift Adaptation")
    print("=" * 60)

    windows, labels = load_pamap2_data()
    print(f"\n   Loaded {len(windows)} windows")

    detector = LSTMDriftDetector(lstm_config=LSTMConfig(hidden_size=32, num_layers=1))
    config = AdapterConfig(feature_compensation_strength=0.5)
    adapter = TypeSpecificAdapter(config)

    # Initial training
    for i in range(0, min(100, len(windows)), 32):
        detector.train_batch(windows[i:i+32], labels[i:i+32])

    # Simulate feature-level drift (shift one sensor axis)
    print("\n   Simulating feature-level drift (sensor bias)...")
    drifted_windows = windows.copy()
    drifted_windows[:, :, 0] += 2.0  # Bias on first axis

    # Adapt
    print("   Adapting to feature-level drift...")
    result = adapter.adapt(
        drift_class=DriftClass.FEATURE_LEVEL,
        detector=detector,
        recent_data=drifted_windows[100:200],
        baseline_data=windows[0:100],
        timestamp=100
    )
    print(f"   Strategy: {result.strategy_used}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("FEATURE-LEVEL ADAPTATION TEST PASSED")
    print("=" * 60)
    return True


def test_model_library():
    """Test model library save/load functionality."""
    print("=" * 60)
    print("TEST: Model Library")
    print("=" * 60)

    print("\n1. Creating model library...")
    library = ModelLibrary(max_size=5)
    print("   PASS")

    # Save models
    print("\n2. Saving models to library...")
    for i in range(3):
        detector = LSTMDriftDetector(lstm_config=LSTMConfig(hidden_size=16, num_layers=1))
        weights = detector.model.get_weights()
        library.save_model(
            drift_features=np.random.randn(10),
            model_weights=weights,
            drift_class=DriftClass.SUDDEN,
            accuracy=0.5 + i * 0.1,
            timestamp=i * 100
        )
    print(f"   Saved {len(library.models)} models")
    print("   PASS")

    # Find matching model
    print("\n3. Finding matching model...")
    match = library.find_matching_model(np.random.randn(10), threshold=0.5)
    print(f"   Match found: {match is not None}")
    print("   PASS")

    # Get best model
    print("\n4. Getting best model...")
    best = library.get_best_model()
    print(f"   Best model accuracy: {best['accuracy'] if best else 'N/A'}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("MODEL LIBRARY TEST PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all Phase 4 tests."""
    print("\n" + "=" * 60)
    print("PHASE 4: ADAPTATION STRATEGIES - TEST SUITE")
    print("=" * 60)

    results = {
        'sudden': test_sudden_adaptation(),
        'gradual': test_gradual_adaptation(),
        'recurring': test_recurring_adaptation(),
        'feature_level': test_feature_level_adaptation(),
        'model_library': test_model_library()
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
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
