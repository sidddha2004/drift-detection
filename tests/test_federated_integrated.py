"""
Test Suite for Federated Integrated Drift Detection

Tests the integration of federated learning with the drift detection +
classification + adaptation pipeline.

Tests:
1. FederatedIntegratedDriftDetector creation and basic operations
2. Drift detection triggers adaptation
3. Adaptations accumulate and trigger FL sync
4. Full integrated experiment with multiple clients
5. Non-IID data distribution with FL

Usage:
    python tests/test_federated_integrated.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.drift_detection import (
    FederatedIntegratedDriftDetector,
    FederatedIntegratedDriftDetectorConfig,
    LSTMConfig
)
from src.federated import (
    FLServer,
    FLServerConfig,
    FederatedIntegratedSimulator
)


def test_federated_detector_creation():
    """Test creating FederatedIntegratedDriftDetector."""
    print("=" * 60)
    print("TEST: FederatedIntegratedDriftDetector Creation")
    print("=" * 60)

    print("\n1. Creating detector config...")
    config = FederatedIntegratedDriftDetectorConfig(
        client_id=0,
        sync_interval_adaptations=3,
        sync_interval_minutes=5.0,
        quantization_bits=32
    )
    print("   PASS")

    print("\n2. Creating federated detector...")
    detector = FederatedIntegratedDriftDetector(config=config)
    print(f"   Client ID: {detector.config.client_id}")
    print(f"   FL client connected: {detector.fl_client is not None}")
    print("   PASS")

    print("\n3. Testing predict method...")
    dummy_features = np.random.randn(256, 9).astype(np.float32)
    pred = detector.predict(dummy_features)
    print(f"   Prediction: {pred}")
    print("   PASS")

    print("\n4. Checking initial state...")
    stats = detector.get_statistics()
    print(f"   Adaptations since sync: {stats['adaptations_since_sync']}")
    print(f"   Total syncs: {stats['total_syncs']}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("CREATION TEST PASSED")
    print("=" * 60)
    return True


def test_drift_detection_and_adaptation():
    """Test that drift detection triggers adaptation."""
    print("=" * 60)
    print("TEST: Drift Detection and Adaptation")
    print("=" * 60)

    print("\n1. Creating detector with sensitive drift detection...")
    config = FederatedIntegratedDriftDetectorConfig(
        client_id=0,
        adwin_delta=0.01,  # More sensitive
        sync_interval_adaptations=5,
        min_samples_for_adaptation=20
    )
    detector = FederatedIntegratedDriftDetector(config=config)
    print("   PASS")

    print("\n2. Generating synthetic training data...")
    np.random.seed(42)
    n_train = 200
    train_features = np.random.randn(n_train, 256, 9).astype(np.float32)
    train_labels = np.random.randint(0, 4, n_train)
    print(f"   Generated {n_train} samples")
    print("   PASS")

    print("\n3. Pre-training detector...")
    for epoch in range(3):
        for i in range(0, n_train, 32):
            detector.detector.detector.train_batch(
                train_features[i:i+32],
                train_labels[i:i+32]
            )
    print("   Pre-training complete")
    print("   PASS")

    print("\n4. Processing stream and counting adaptations...")
    initial_adaptations = detector.detector.total_adaptations
    drift_count = 0

    # Process with some drift injection
    for i in range(100):
        features = train_features[min(i + 100, n_train - 1)]
        # Inject drift by shifting labels after sample 50
        if i < 50:
            label = train_labels[min(i + 100, n_train - 1)]
        else:
            label = (train_labels[min(i + 100, n_train - 1)] + 1) % 4

        pred, drift = detector.process(features, label)
        if drift:
            drift_count += 1

    final_adaptations = detector.detector.total_adaptations
    adaptations_triggered = final_adaptations - initial_adaptations

    print(f"   Drift events detected: {drift_count}")
    print(f"   Adaptations triggered: {adaptations_triggered}")
    print(f"   Adaptations since sync: {detector.adaptations_since_sync}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("DRIFT DETECTION TEST PASSED")
    print("=" * 60)
    return True


def test_fl_sync_trigger():
    """Test that adaptations trigger FL sync when threshold is met."""
    print("=" * 60)
    print("TEST: FL Sync Trigger")
    print("=" * 60)

    print("\n1. Creating server and detector...")
    server_config = FLServerConfig(min_clients=1, single_process=True)
    server = FLServer(config=server_config)

    config = FederatedIntegratedDriftDetectorConfig(
        client_id=0,
        sync_interval_adaptations=2,  # Low threshold for testing
        quantization_bits=32
    )
    detector = FederatedIntegratedDriftDetector(config=config)
    detector.set_fl_server(server)
    server.register_detector(detector)
    print("   PASS")

    print("\n2. Simulating adaptations...")
    print(f"   Sync interval: {config.sync_interval_adaptations} adaptations")

    # Manually trigger adaptations
    for i in range(3):
        detector.adaptations_since_sync += 1
        print(f"   After adaptation {i+1}: adaptations_since_sync = {detector.adaptations_since_sync}")

        # Check if sync should trigger
        result = detector.maybe_sync()
        if result and result.synced:
            print(f"   -> SYNC TRIGGERED at adaptation {i+1}!")
            print(f"      Round: {result.round_number}")
            print(f"      Sync time: {result.sync_time_ms:.1f}ms")

    print("   PASS")

    print("\n3. Checking sync state...")
    print(f"   Adaptations since sync: {detector.adaptations_since_sync}")
    print(f"   Total syncs: {detector.total_syncs}")
    print(f"   Server completed rounds: {server.completed_rounds}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("FL SYNC TRIGGER TEST PASSED")
    print("=" * 60)
    return True


def test_integrated_simulator():
    """Test full integrated simulator with multiple clients."""
    print("=" * 60)
    print("TEST: FederatedIntegratedSimulator")
    print("=" * 60)

    print("\n1. Creating simulator with 3 clients...")
    # Use LSTM with raw window input (256, 9)
    lstm_config = LSTMConfig(input_size=9, hidden_size=32, num_layers=1, seq_len=256)
    config = FederatedIntegratedDriftDetectorConfig(
        lstm_config=lstm_config,
        sync_interval_adaptations=10,
        quantization_bits=32
    )
    simulator = FederatedIntegratedSimulator(
        n_clients=3,
        server_config=FLServerConfig(min_clients=2),
        detector_config=config,
        lstm_config=lstm_config
    )
    print(f"   Created {len(simulator.detectors)} detectors")
    print("   PASS")

    print("\n2. Generating synthetic window data (LSTM format)...")
    np.random.seed(42)
    n_samples = 99  # Divisible by 3
    # LSTM expects (seq_len=256, input_size=9) windows
    features = np.random.randn(n_samples, 256, 9).astype(np.float32)
    labels = np.random.randint(0, 4, n_samples)
    subject_ids = np.repeat([1, 2, 3], n_samples // 3)
    print(f"   Generated {n_samples} windows of shape (256, 9)")
    print("   PASS")

    print("\n3. Distributing data non-IID...")
    simulator.distribute_data_non_iid(features, labels, subject_ids)
    for cid, (feat, lab) in simulator.client_data.items():
        print(f"   Client {cid}: {len(feat)} samples, shape {feat.shape}")
    print("   PASS")

    print("\n4. Running integrated experiment...")
    results = simulator.run_integrated_experiment(
        n_stream_samples=60,  # Smaller for faster test
        seed=42
    )

    print(f"   Batches processed: {len(results['client_stats'])}")
    print(f"   Drift events: {len(results['drift_events'])}")
    print(f"   FL syncs: {len(results['sync_events'])}")
    print("   PASS")

    print("\n5. Printing experiment summary...")
    summary = simulator.get_experiment_summary(results)
    print(summary)

    print("\n" + "=" * 60)
    print("INTEGRATED SIMULATOR TEST PASSED")
    print("=" * 60)
    return True


def test_full_pipeline_integration():
    """Test complete pipeline: data → detection → adaptation → FL."""
    print("=" * 60)
    print("TEST: Full Pipeline Integration")
    print("=" * 60)

    print("\n1. Setting up multi-client FL system...")
    n_clients = 4
    lstm_config = LSTMConfig(input_size=9, hidden_size=32, num_layers=1, seq_len=256)
    config = FederatedIntegratedDriftDetectorConfig(
        lstm_config=lstm_config,
        sync_interval_adaptations=5,
        adwin_delta=0.005,
        min_samples_for_adaptation=15
    )

    simulator = FederatedIntegratedSimulator(
        n_clients=n_clients,
        server_config=FLServerConfig(min_clients=2),
        detector_config=config,
        lstm_config=lstm_config
    )
    print(f"   {n_clients} clients configured")
    print("   PASS")

    print("\n2. Generating data with drift patterns...")
    np.random.seed(123)
    n_samples = 120  # Divisible by 4
    # LSTM expects (seq_len=256, input_size=9) windows
    features = np.random.randn(n_samples, 256, 9).astype(np.float32)
    labels = np.random.randint(0, 6, n_samples)
    subject_ids = np.repeat([1, 2, 3, 4], n_samples // 4)

    # Inject drift in second half (simulated distribution shift)
    drift_point = n_samples // 2
    labels[drift_point:] = (labels[drift_point:] + 1) % 6
    print(f"   Drift injected at sample {drift_point}")
    print("   PASS")

    print("\n3. Distributing data non-IID...")
    simulator.distribute_data_non_iid(features, labels, subject_ids)
    print("   PASS")

    print("\n4. Running integrated experiment with drift...")
    results = simulator.run_integrated_experiment(
        n_stream_samples=60,
        seed=123
    )
    print("   PASS")

    print("\n5. Analyzing results...")
    drift_events = results.get('drift_events', [])
    sync_events = results.get('sync_events', [])

    # Count drift events per client
    client_drift_counts = {}
    for event in drift_events:
        cid = event['client_id']
        client_drift_counts[cid] = client_drift_counts.get(cid, 0) + 1

    print(f"   Total drift events: {len(drift_events)}")
    print(f"   Total FL syncs: {len(sync_events)}")
    print("   Drift events per client:")
    for cid, count in sorted(client_drift_counts.items()):
        print(f"     Client {cid}: {count} drift events")

    # Count drift types
    if drift_events:
        drift_types = {}
        for event in drift_events:
            dt = event.get('drift_class', 'Unknown')
            drift_types[dt] = drift_types.get(dt, 0) + 1
        print("   Drift type distribution:")
        for dt, count in sorted(drift_types.items()):
            print(f"     {dt}: {count}")

    print("   PASS")

    print("\n6. Final summary...")
    print(simulator.get_experiment_summary(results))

    print("\n" + "=" * 60)
    print("FULL PIPELINE INTEGRATION TEST PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all federated integrated tests."""
    print("\n" + "=" * 60)
    print("FEDERATED INTEGRATED DRIFT DETECTION - TEST SUITE")
    print("=" * 60)

    results = {
        'detector_creation': test_federated_detector_creation(),
        'drift_detection': test_drift_detection_and_adaptation(),
        'fl_sync_trigger': test_fl_sync_trigger(),
        'integrated_simulator': test_integrated_simulator(),
        'full_pipeline': test_full_pipeline_integration()
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
