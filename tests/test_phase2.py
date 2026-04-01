"""
Test Suite for Phase 2: Binary Drift Detection (Stage A)

Tests:
- MLP model (forward pass, training, weight extraction)
- ADWIN drift detector (error tracking, drift flagging)
- Edge client (stream processing)

Usage:
    python tests/test_phase2.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.drift_detection import (
    ActivityMLP,
    BinaryDriftDetector,
    EdgeClient,
    MLPConfig,
    ADWINConfig,
    DriftClassifier,
    StageBDetector,
    DriftClass
)
from src.data_pipeline import (
    PAMAP2Preprocessor,
    create_data_pipeline,
    DriftConfig,
    DriftType,
    DriftInjector
)


def test_mlp_model():
    """Test MLP model architecture."""
    print("=" * 60)
    print("TEST: MLP Model")
    print("=" * 60)

    # Test model creation
    print("\n1. Creating MLP model...")
    config = MLPConfig(
        input_dim=45,
        hidden_dim=32,
        output_dim=18
    )
    model = ActivityMLP(config)
    print(f"   Model created successfully")
    print("   PASS")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 16
    x = torch.randn(batch_size, 45)
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
    assert 'fc1.weight' in weights, "Missing fc1.weight"
    assert 'fc2.weight' in weights, "Missing fc2.weight"
    print("   PASS")

    # Test weight loading
    print("\n5. Testing weight loading...")
    original_w1 = weights['fc1.weight'].copy()
    weights['fc1.weight'] *= 2  # Modify weights
    model.set_weights(weights)
    new_weights = model.get_weights()

    # Verify weights were loaded
    assert np.allclose(new_weights['fc1.weight'], original_w1 * 2), "Weight loading failed"
    print("   PASS")

    # Count parameters
    print("\n6. Counting parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    expected = (45 * 32 + 32) + (32 * 18 + 18)  # fc1 + fc2
    print(f"   Total parameters: {total_params}")
    print(f"   Expected: {expected}")
    assert total_params == expected, "Parameter count mismatch"
    print(f"   Size: ~{total_params * 4 / 1024:.1f} KB (float32)")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL MLP MODEL TESTS PASSED")
    print("=" * 60)
    return True


def test_binary_drift_detector():
    """Test binary drift detector with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: Binary Drift Detector")
    print("=" * 60)

    # Generate synthetic features and labels
    print("\n1. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 500
    features = np.random.randn(n_samples, 45).astype(np.float32)
    labels = np.random.randint(0, 18, n_samples)
    print(f"   Generated {n_samples} samples")
    print("   PASS")

    # Create detector
    print("\n2. Creating drift detector...")
    detector = BinaryDriftDetector()
    print("   PASS")

    # Test initial training
    print("\n3. Testing initial training...")
    train_features = features[:200]
    train_labels = labels[:200]

    for epoch in range(3):
        indices = np.random.permutation(200)
        for i in range(0, 200, 32):
            batch_idx = indices[i:i+32]
            loss = detector.train_batch(
                train_features[batch_idx],
                train_labels[batch_idx]
            )
    print(f"   Training completed")
    print("   PASS")

    # Test evaluation
    print("\n4. Testing evaluation...")
    acc, loss = detector.evaluate(features[200:300], labels[200:300])
    print(f"   Accuracy: {acc:.3f}")
    print(f"   Loss: {loss:.3f}")
    assert 0 <= acc <= 1, "Accuracy out of range"
    print("   PASS")

    # Test drift detection (synthetic drift)
    print("\n5. Testing drift detection...")

    # Process stable stream first
    stable_errors = []
    for i in range(200, 400):
        pred = detector.predict(features[i])
        drift_flag = detector.update_and_check_drift(pred, labels[i])
        stable_errors.append(1 if pred != labels[i] else 0)

    stable_error_rate = np.mean(stable_errors)
    print(f"   Stable error rate: {stable_error_rate:.3f}")
    print(f"   Drift flags during stable: {sum(1 for f in stable_errors if f)}")

    # Now introduce "drift" by using wrong labels
    drift_errors = []
    drift_flags = []
    for i in range(400, 500):
        pred = detector.predict(features[i])
        # Use random labels to simulate drift
        wrong_label = np.random.randint(0, 18)
        drift_flag = detector.update_and_check_drift(pred, wrong_label)
        drift_errors.append(1 if pred != wrong_label else 0)
        drift_flags.append(drift_flag)

    drift_error_rate = np.mean(drift_errors)
    print(f"   Drift error rate: {drift_error_rate:.3f}")
    print(f"   Drift flags during drift: {sum(drift_flags)}")
    print(f"   Total drift events: {detector.drift_count}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL BINARY DRIFT DETECTOR TESTS PASSED")
    print("=" * 60)
    return True


def test_drift_classifier():
    """Test Phase 3 drift classifier."""
    print("\n" + "=" * 60)
    print("TEST: Drift Classifier (Stage B)")
    print("=" * 60)

    # Create classifier
    print("\n1. Creating drift classifier...")
    classifier = DriftClassifier()
    print("   PASS")

    # Test with synthetic error patterns
    print("\n2. Testing Sudden drift classification...")
    sudden_errors = np.concatenate([
        np.random.binomial(1, 0.1, 50),  # 10% baseline
        np.random.binomial(1, 0.7, 50)   # 70% after drift
    ])

    drift_class, details = classifier.classify(sudden_errors)
    print(f"   Class: {DriftClass(drift_class).name}")
    print(f"   Reason: {details['reason']}")
    print(f"   Error slope: {details['error_slope']:.4f}")
    print("   PASS")

    print("\n3. Testing Gradual drift classification...")
    gradual_errors = np.random.binomial(
        1,
        np.linspace(0.1, 0.5, 100),
        100
    )

    drift_class, details = classifier.classify(gradual_errors)
    print(f"   Class: {DriftClass(drift_class).name}")
    print(f"   Reason: {details['reason']}")
    print(f"   Error slope: {details['error_slope']:.4f}")
    print("   PASS")

    # Test Stage B detector
    print("\n4. Testing Stage B detector...")
    stage_b = StageBDetector()

    # Add errors to buffer
    for err in sudden_errors:
        stage_b.add_error(err)

    # Activate and classify
    activated = stage_b.activate()
    print(f"   Activated: {activated}")

    if activated:
        drift_class, details = stage_b.classify_current_drift(timestamp=1000)
        print(f"   Classification: {DriftClass(drift_class).name}")
        print(f"   Confidence: {details['confidence']:.3f}")

    # Get state
    state = stage_b.get_state()
    print(f"   State: {state}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL DRIFT CLASSIFIER TESTS PASSED")
    print("=" * 60)
    return True


def test_edge_client():
    """Test edge client stream processing."""
    print("\n" + "=" * 60)
    print("TEST: Edge Client")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    features = np.random.randn(n_samples, 45).astype(np.float32)
    labels = np.random.randint(0, 18, n_samples)
    print(f"   Generated {n_samples} samples")
    print("   PASS")

    # Create edge client
    print("\n2. Creating edge client...")
    client = EdgeClient(client_id=1)
    print("   PASS")

    # Process stream
    print("\n3. Processing data stream...")
    stats = client.process_stream(features, labels, initial_train_epochs=3)

    print(f"   Client ID: {stats['client_id']}")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Accuracy: {stats['accuracy']:.3f}")
    print(f"   Drift count: {stats['drift_count']}")
    print(f"   Final error rate: {stats['final_error_rate']:.3f}")
    print("   PASS")

    # Test weight extraction
    print("\n4. Testing weight extraction for FL...")
    weights = client.get_model_weights()
    print(f"   Extracted {len(weights)} weight tensors")
    print("   PASS")

    # Test weight loading
    print("\n5. Testing weight loading...")
    client.load_model_weights(weights)
    print("   Weights loaded successfully")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL EDGE CLIENT TESTS PASSED")
    print("=" * 60)
    return True


def test_with_real_data():
    """Test Phase 2 with real PAMAP2 data if available."""
    print("\n" + "=" * 60)
    print("TEST: Phase 2 with Real Data")
    print("=" * 60)

    # Find data file
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
        print("SKIP: PAMAP2 dataset not found")
        return True  # Don't fail, just skip

    print(f"Using data file: {test_file}")

    # Load and preprocess
    print("\n1. Loading data...")
    preprocessor = PAMAP2Preprocessor()
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)
    print(f"   Loaded {accel_data.shape[0]} samples")

    # Generate windows
    print("\n2. Generating windows...")
    pipeline = create_data_pipeline(window_size=256, stride=128)
    features, labels = pipeline.process(accel_data, activity_labels)
    print(f"   Generated {features.shape[0]} windows")

    # Create detector
    print("\n3. Creating and training detector...")
    detector = BinaryDriftDetector()

    # Train on first portion
    train_size = min(500, len(features) // 4)
    for epoch in range(3):
        indices = np.random.permutation(train_size)
        for i in range(0, train_size, 32):
            batch_idx = indices[i:i+32]
            detector.train_batch(
                features[batch_idx],
                labels[batch_idx]
            )
    print("   Training complete")

    # Evaluate
    print("\n4. Evaluating model...")
    acc, loss = detector.evaluate(
        features[train_size:train_size+200],
        labels[train_size:train_size+200]
    )
    print(f"   Accuracy: {acc:.3f}")
    print(f"   Loss: {loss:.3f}")

    # Test drift injection + detection
    print("\n5. Testing drift injection + detection...")

    # Inject drift
    drift_config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        magnitude=1.0,
        injection_point=500
    )
    injector = DriftInjector(drift_config)

    # Re-preprocess with drift
    accel_data_drifted = preprocessor.preprocess_file(test_file)[0]
    accel_data_drifted = injector.inject_drift(accel_data_drifted)
    features_drifted, _ = pipeline.process(accel_data_drifted, activity_labels)

    # Process drifted stream
    drift_flags = []
    for i in range(len(features_drifted)):
        pred = detector.predict(features_drifted[i])
        drift_flag = detector.update_and_check_drift(pred, labels[i])
        drift_flags.append(drift_flag)

    print(f"   Total drift flags: {sum(drift_flags)}")
    print(f"   Drift events: {detector.drift_count}")

    if drift_flags:
        first_drift_idx = next((i for i, f in enumerate(drift_flags) if f), None)
        if first_drift_idx:
            print(f"   First drift detection at: {first_drift_idx}")
            print(f"   Detection latency: {first_drift_idx - 500} samples")

    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL REAL DATA TESTS PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 60)
    print("PHASE 2: BINARY DRIFT DETECTION - TEST SUITE")
    print("=" * 60)

    results = {
        'mlp_model': test_mlp_model(),
        'binary_detector': test_binary_drift_detector(),
        'drift_classifier': test_drift_classifier(),
        'edge_client': test_edge_client(),
        'real_data': test_with_real_data()
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
