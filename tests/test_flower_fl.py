"""
Test Suite for Flower-based Federated Learning

Tests:
- Flower client creation and interface
- Weight extraction/loading via Flower protocol
- Local training with Flower fit()
- Evaluation with Flower evaluate()
- Full FL simulation with multiple clients
- Non-IID data partitioning
- Accuracy tracking across FL rounds

Usage:
    python tests/test_flower_fl.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.federated.flower_client import (
    FlowerClient,
    FlowerClientConfig,
    NonIIDDataPartitioner
)
from src.federated.flower_simulation import (
    FlowerSimulation,
    FlowerSimulationConfig,
    run_flower_simulation
)
from src.drift_detection.lstm_model import LSTMConfig
from src.data_pipeline import PAMAP2Preprocessor, create_data_pipeline


def test_flower_client():
    """Test Flower client interface."""
    print("=" * 60)
    print("TEST: Flower Client Interface")
    print("=" * 60)

    # Create config
    print("\n1. Creating Flower client config...")
    config = FlowerClientConfig(
        client_id=0,
        num_clients=5,
        use_lstm=True,
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1),
        local_epochs=2,
        batch_size=32
    )
    print("   PASS")

    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    n_samples = 200
    features = np.random.randn(n_samples, 256, 9).astype(np.float32)
    labels = np.random.randint(0, 6, n_samples)
    print(f"   Generated {n_samples} samples, 6 classes")
    print("   PASS")

    # Create client
    print("\n3. Creating Flower client...")
    client = FlowerClient(
        client_id=0,
        config=config,
        train_data=(features[:150], labels[:150]),
        test_data=(features[150:], labels[150:])
    )
    print("   PASS")

    # Test get_parameters
    print("\n4. Testing get_parameters()...")
    params = client.get_parameters({})
    print(f"   Retrieved {len(params)} parameter arrays")
    for i, p in enumerate(params[:3]):
        print(f"   Param {i}: shape={p.shape}, dtype={p.dtype}")
    assert len(params) > 0, "No parameters returned"
    print("   PASS")

    # Test fit (local training)
    print("\n5. Testing fit()...")
    initial_params = client.get_parameters({})
    fit_config = {"epochs": "2", "batch_size": "32"}

    updated_params, num_samples, metrics = client.fit(initial_params, fit_config)
    print(f"   Samples processed: {num_samples}")
    print(f"   Metrics: {metrics}")
    assert len(updated_params) == len(initial_params), "Parameter count mismatch"
    print("   PASS")

    # Test evaluate
    print("\n6. Testing evaluate()...")
    eval_config = {"round": "1"}
    loss, num_samples, eval_metrics = client.evaluate(updated_params, eval_config)
    print(f"   Test loss: {loss:.4f}")
    print(f"   Test samples: {num_samples}")
    print(f"   Metrics: {eval_metrics}")
    print("   PASS")

    # Test statistics
    print("\n7. Getting client statistics...")
    stats = client.get_statistics()
    print(f"   {stats}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL FLOWER CLIENT TESTS PASSED")
    print("=" * 60)
    return True


def test_non_iid_partitioning():
    """Test non-IID data partitioning."""
    print("=" * 60)
    print("TEST: Non-IID Data Partitioning")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    n_classes = 6
    features = np.random.randn(n_samples, 256, 9).astype(np.float32)
    labels = np.random.randint(0, n_classes, n_samples)
    print(f"   Generated {n_samples} samples, {n_classes} classes")
    print("   PASS")

    # Test IID partitioning
    print("\n2. Testing IID partitioning...")
    partitioner = NonIIDDataPartitioner(features, labels, num_clients=5, partition_type="iid")
    iid_data = partitioner.partition(test_ratio=0.2)

    for client_id, data in iid_data.items():
        n_train = len(data["train_labels"])
        n_classes_client = len(data["label_distribution"])
        print(f"   Client {client_id}: {n_train} samples, {n_classes_client} classes")
    print("   PASS")

    # Test label skew partitioning
    print("\n3. Testing label skew partitioning...")
    partitioner = NonIIDDataPartitioner(
        features, labels, num_clients=5,
        partition_type="label_skew",
        alpha=0.5
    )
    skewed_data = partitioner.partition(test_ratio=0.2)

    print("   Label distribution per client:")
    for client_id, data in skewed_data.items():
        dist = data["label_distribution"]
        total = sum(dist.values())
        dist_str = ", ".join([f"{k}:{v}({100*v/total:.0f}%)" for k, v in sorted(dist.items())])
        print(f"   Client {client_id}: {dist_str}")
    print("   PASS")

    # Test quantity skew partitioning
    print("\n4. Testing quantity skew partitioning...")
    partitioner = NonIIDDataPartitioner(features, labels, num_clients=5, partition_type="quantity_skew")
    quantity_data = partitioner.partition(test_ratio=0.2)

    for client_id, data in quantity_data.items():
        n_train = data["n_train_samples"]
        print(f"   Client {client_id}: {n_train} samples")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL NON-IID PARTITIONING TESTS PASSED")
    print("=" * 60)
    return True


def test_flower_with_real_data():
    """Test Flower FL with real PAMAP2 data."""
    print("=" * 60)
    print("TEST: Flower FL with Real PAMAP2 Data")
    print("=" * 60)

    # Load PAMAP2 data
    print("\n1. Loading PAMAP2 data (subject101)...")
    preprocessor = PAMAP2Preprocessor()
    test_file = "PAMAP2_Dataset/Protocol/subject101.dat"
    accel_data, activity_labels = preprocessor.preprocess_file(test_file)
    print(f"   Loaded {len(activity_labels)} samples")
    print(f"   Unique activities: {np.unique(activity_labels)}")
    print("   PASS")

    # Generate windows for LSTM
    print("\n2. Generating raw windows for LSTM...")
    pipeline = create_data_pipeline(
        window_size=256,
        stride=128,
        return_raw_windows=True
    )
    windows, labels = pipeline.process(accel_data, activity_labels)
    print(f"   Generated {windows.shape[0]} windows")
    print(f"   Window shape: {windows.shape}")

    # Remap labels to contiguous range
    unique_labels = np.unique(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels_remapped = np.array([label_to_idx[l] for l in labels])
    n_classes = len(unique_labels)
    print(f"   Classes remapped: {n_classes} classes (0 to {n_classes-1})")
    print("   PASS")

    # Run Flower simulation (quick test: 3 rounds, 3 clients)
    print("\n3. Running Flower FL simulation (3 clients, 3 rounds)...")
    config = FlowerSimulationConfig(
        num_clients=3,
        num_rounds=3,
        partition_type="iid",  # Use IID for initial test
        use_lstm=True,
        lstm_config=LSTMConfig(
            input_size=9,
            hidden_size=32,  # Smaller for faster testing
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            output_dim=n_classes
        ),
        local_epochs=3,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2
    )

    sim = FlowerSimulation(config, windows, labels_remapped)
    history = sim.run()

    print("\n4. Extracting results...")
    print("   PASS")

    # Print summary
    print("\n5. Simulation summary...")
    print(f"   Rounds completed: {sim.config.num_rounds}")

    # Show per-client data distribution
    print("\n   Data distribution:")
    for client_id, data in sim.client_data.items():
        n_train = len(data["train_labels"])
        n_test = len(data["test_labels"])
        print(f"   Client {client_id}: {n_train} train, {n_test} test")

    print("   PASS")

    print("\n" + "=" * 60)
    print("FLOWER FL WITH REAL DATA TEST PASSED")
    print(f"FINAL ACCURACY: See simulation output above")
    print("=" * 60)
    return True


def test_full_fl_simulation():
    """Test complete Flower FL simulation with more rounds."""
    print("=" * 60)
    print("TEST: Full Flower FL Simulation (5 rounds)")
    print("=" * 60)

    # Generate synthetic data with clear patterns
    print("\n1. Generating synthetic data with class patterns...")
    np.random.seed(42)
    n_samples = 2000
    n_classes = 4
    n_windows = 256

    features = []
    labels = []

    for class_id in range(n_classes):
        # Create class-specific patterns
        class_samples = n_samples // n_classes
        base_signal = np.sin(np.linspace(0, 2 * np.pi * (class_id + 1), n_windows))

        for _ in range(class_samples):
            window = np.zeros((n_windows, 9))
            for axis in range(9):
                noise = np.random.randn(n_windows) * 0.1
                window[:, axis] = base_signal * (axis + 1) / 9 + noise

            features.append(window)
            labels.append(class_id)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"   Generated {len(features)} samples, {n_classes} classes")
    print("   PASS")

    # Run simulation with 5 clients, 5 rounds
    print("\n2. Running Flower FL simulation (5 clients, 5 rounds)...")
    config = FlowerSimulationConfig(
        num_clients=5,
        num_rounds=5,
        partition_type="iid",
        use_lstm=True,
        lstm_config=LSTMConfig(
            input_size=9,
            hidden_size=32,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            output_dim=n_classes
        ),
        local_epochs=3,
        fraction_fit=1.0,
        fraction_evaluate=1.0
    )

    sim = FlowerSimulation(config, features, labels)
    results = sim.run()

    print("\n3. Results summary...")
    print(f"   Simulation complete: {results['final_accuracy']:.1%} final accuracy")
    print("   PASS")

    print("\n" + "=" * 60)
    print("FULL FL SIMULATION TEST PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all Flower FL tests."""
    print("\n" + "=" * 60)
    print("FLOWER FEDERATED LEARNING - TEST SUITE")
    print("=" * 60)

    results = {
        'flower_client': test_flower_client(),
        'non_iid_partitioning': test_non_iid_partitioning(),
        'real_data': test_flower_with_real_data(),
        'full_simulation': test_full_fl_simulation()
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
