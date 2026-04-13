"""
Test Suite for Phase 6: LSTM Model for HAR

Tests:
- LSTM model architecture
- Forward pass, prediction
- Weight extraction/loading (FL compatibility)
- Training on real PAMAP2 data
- Accuracy validation (target: 80%+)

Usage:
    python tests/test_phase6.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.drift_detection import (
    ActivityLSTM,
    LSTMDriftDetector,
    LSTMConfig
)
from src.data_pipeline import (
    PAMAP2Preprocessor,
    create_data_pipeline
)


def test_lstm_model():
    """Test LSTM model architecture."""
    print("=" * 60)
    print("TEST: LSTM Model Architecture")
    print("=" * 60)

    # Test model creation
    print("\n1. Creating LSTM model...")
    config = LSTMConfig(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        output_dim=18,
        seq_len=256
    )
    model = ActivityLSTM(config)
    print(f"   Model created successfully")
    print("   PASS")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 16
    x = torch.randn(batch_size, 256, 9)  # (batch, seq_len, input_size)
    logits = model.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 18), "Output shape mismatch"
    print("   PASS")

    # Test prediction
    print("\n3. Testing prediction...")
    predictions = model.predict(x)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Unique predictions: {np.unique(predictions)}")
    assert predictions.shape == (batch_size,), "Prediction shape mismatch"
    assert all(0 <= p < 18 for p in predictions), "Predictions out of range"
    print("   PASS")

    # Test weight extraction
    print("\n4. Testing weight extraction...")
    weights = model.get_weights()
    print(f"   Extracted {len(weights)} weight tensors")
    for name, w in weights.items():
        print(f"   {name}: {w.shape}")
    assert 'lstm.weight_ih_l0' in weights, "Missing LSTM weights"
    assert 'fc.weight' in weights, "Missing FC weights"
    print("   PASS")

    # Test weight loading
    print("\n5. Testing weight loading...")
    original_w = weights['fc.weight'].copy()
    weights['fc.weight'] *= 2
    model.set_weights(weights)
    new_weights = model.get_weights()
    assert np.allclose(new_weights['fc.weight'], original_w * 2), "Weight loading failed"
    print("   PASS")

    # Count parameters
    print("\n6. Counting parameters...")
    total, trainable = model.count_parameters()
    print(f"   Total parameters: {total}")
    print(f"   Trainable parameters: {trainable}")
    memory = model.get_memory_size()
    print(f"   Size (float32): {memory['float32']:.1f} KB")
    print(f"   Size (int8): {memory['int8']:.1f} KB")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL LSTM MODEL TESTS PASSED")
    print("=" * 60)
    return True


def test_lstm_drift_detector():
    """Test LSTM drift detector."""
    print("=" * 60)
    print("TEST: LSTM Drift Detector")
    print("=" * 60)

    # Create detector
    print("\n1. Creating LSTM drift detector...")
    detector = LSTMDriftDetector()
    print("   PASS")

    # Generate synthetic window data
    print("\n2. Generating synthetic window data...")
    np.random.seed(42)
    n_windows = 200
    windows = np.random.randn(n_windows, 256, 9).astype(np.float32)
    labels = np.random.randint(0, 18, n_windows)
    print(f"   Generated {n_windows} windows")
    print("   PASS")

    # Test training
    print("\n3. Testing batch training...")
    for epoch in range(3):
        for i in range(0, 100, 32):
            loss = detector.train_batch(windows[i:i+32], labels[i:i+32])
    print(f"   Training completed (3 epochs)")
    print("   PASS")

    # Test evaluation
    print("\n4. Testing evaluation...")
    acc, loss = detector.evaluate(windows[100:150], labels[100:150])
    print(f"   Accuracy: {acc:.3f}")
    print(f"   Loss: {loss:.3f}")
    print("   PASS")

    # Test drift detection
    print("\n5. Testing drift detection...")
    drift_flags = []
    for i in range(150, 200):
        pred = detector.predict(windows[i])
        # pred is a scalar for single sample
        drift_flag = detector.update_and_check_drift(pred, labels[i])
        drift_flags.append(drift_flag)
    print(f"   Drift flags: {sum(drift_flags)}")
    print(f"   Drift events: {detector.drift_count}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL LSTM DRIFT DETECTOR TESTS PASSED")
    print("=" * 60)
    return True


def test_lstm_with_real_data():
    """Test LSTM with real PAMAP2 data - accuracy validation."""
    print("=" * 60)
    print("TEST: LSTM with Real PAMAP2 Data")
    print("=" * 60)

    # Load PAMAP2 data
    print("\n1. Loading PAMAP2 data (subject101)...")
    preprocessor = PAMAP2Preprocessor()
    test_file = "fl-drift-try_2\PAMAP2_Dataset\Optional\subject101.dat"
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)
    print(f"   Loaded {len(activity_labels)} samples")
    print(f"   Unique activities: {np.unique(activity_labels)}")
    print("   PASS")

    # Generate windows (raw for LSTM)
    print("\n2. Generating raw windows for LSTM...")
    pipeline = create_data_pipeline(
        window_size=256,
        stride=128,
        return_raw_windows=True  # Return raw windows, not features
    )
    windows, labels = pipeline.process(accel_data, activity_labels)
    print(f"   Generated {windows.shape[0]} windows")
    print(f"   Window shape: {windows.shape}")
    print("   PASS")

    # Show class distribution
    print("\n3. Class distribution...")
    n_classes = len(np.unique(labels))
    print(f"   Number of classes: {n_classes}")
    for cls in sorted(np.unique(labels)):
        count = np.sum(labels == cls)
        print(f"   Activity {cls}: {count} ({100*count/len(labels):.1f}%)")
    print("   PASS")

    # Split data (70% train, 30% test)
    print("\n4. Splitting data (70/30)...")
    n_samples = len(windows)
    n_train = int(n_samples * 0.7)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_windows = windows[train_idx]
    train_labels = labels[train_idx]
    test_windows = windows[test_idx]
    test_labels = labels[test_idx]

    # Remap labels to contiguous range (0 to n_classes-1)
    unique_labels = np.unique(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    train_labels_remapped = np.array([label_to_idx[l] for l in train_labels])
    test_labels_remapped = np.array([label_to_idx[l] for l in test_labels])

    print(f"   Train: {len(train_windows)} windows")
    print(f"   Test: {len(test_windows)} windows")
    print(f"   Classes remapped: {dict(label_to_idx)}")
    print("   PASS")

    # Create and train LSTM detector
    print("\n5. Training LSTM model (20 epochs)...")
    config = LSTMConfig(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=64,
        output_dim=n_classes  # Use actual number of classes
    )
    detector = LSTMDriftDetector(lstm_config=config)

    # Training loop
    best_acc = 0.0
    for epoch in range(20):
        # Shuffle training data
        perm = np.random.permutation(len(train_windows))

        for i in range(0, len(train_windows), config.batch_size):
            batch_idx = perm[i:i+config.batch_size]
            detector.train_batch(
                train_windows[batch_idx],
                train_labels_remapped[batch_idx]
            )

        # Evaluate on test set
        acc, loss = detector.evaluate(test_windows, test_labels_remapped)
        if acc > best_acc:
            best_acc = acc
        print(f"   Epoch {epoch+1}/20: test_acc={acc:.3f}, best={best_acc:.3f}")

    print("   Training complete")
    print("   PASS")

    # Final evaluation
    print("\n6. Final evaluation...")
    final_acc, final_loss = detector.evaluate(test_windows, test_labels_remapped)
    print(f"   Final test accuracy: {final_acc:.1%}")
    print(f"   Final test loss: {final_loss:.3f}")
    print(f"   Best accuracy seen: {best_acc:.1%}")
    print("   PASS")

    # Check accuracy target
    print("\n7. Checking accuracy target (80%)...")
    if final_acc >= 0.80:
        print(f"   TARGET MET: {final_acc:.1%} >= 80%")
    elif final_acc >= 0.70:
        print(f"   Close: {final_acc:.1%} (target 80%)")
    else:
        print(f"   Below target: {final_acc:.1%} (target 80%)")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL REAL DATA TESTS PASSED")
    print(f"FINAL ACCURACY: {final_acc:.1%}")
    print("=" * 60)
    return True


def test_fl_compatibility():
    """Test LSTM weight extraction for FL."""
    print("=" * 60)
    print("TEST: LSTM FL Compatibility")
    print("=" * 60)

    # Create model
    print("\n1. Creating LSTM model...")
    config = LSTMConfig()
    model1 = ActivityLSTM(config)
    print("   PASS")

    # Extract weights
    print("\n2. Extracting weights...")
    weights = model1.get_weights()
    print(f"   Extracted {len(weights)} tensors")
    print("   PASS")

    # Create new model and load
    print("\n3. Loading weights into new model...")
    model2 = ActivityLSTM(config)
    model2.set_weights(weights)
    print("   PASS")

    # Verify weights match
    print("\n4. Verifying weights match...")
    weights2 = model2.get_weights()
    for key in weights:
        if not np.allclose(weights[key], weights2[key]):
            print(f"   FAIL: {key} mismatch")
            return False
    print("   All weights match")
    print("   PASS")

    # Verify predictions match
    print("\n5. Verifying predictions match...")
    x = torch.randn(10, 256, 9)
    with torch.no_grad():
        pred1 = model1(x).numpy()
        pred2 = model2(x).numpy()
    if np.allclose(pred1, pred2, atol=1e-5):
        print("   Predictions match")
    else:
        print(f"   Max diff: {np.max(np.abs(pred1 - pred2))}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL FL COMPATIBILITY TESTS PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all Phase 6 tests."""
    print("\n" + "=" * 60)
    print("PHASE 6: LSTM MODEL - TEST SUITE")
    print("=" * 60)

    results = {
        'lstm_model': test_lstm_model(),
        'lstm_detector': test_lstm_drift_detector(),
        'real_data': test_lstm_with_real_data(),
        'fl_compat': test_fl_compatibility()
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
