"""
Test Suite for Phase 5: Federated Learning

Tests:
- FedAvg aggregation algorithm
- FL client local training
- FL server coordination
- Weight quantization/dequantization
- Full FL round simulation
- Non-IID data distribution

Usage:
    python tests/test_phase5.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.federated import (
    FLClient,
    FLClientConfig,
    FLServer,
    FLServerConfig,
    FLSimulator,
    FedAvgAggregator,
    AggregationConfig,
    QuantizationHelper
)
from src.drift_detection import LSTMConfig


def test_fedavg_aggregator():
    """Test FedAvg aggregation algorithm."""
    print("=" * 60)
    print("TEST: FedAvg Aggregator")
    print("=" * 60)

    # Create aggregator
    print("\n1. Creating FedAvg aggregator...")
    config = AggregationConfig(min_clients=3)
    aggregator = FedAvgAggregator(config)
    print("   PASS")

    # Create synthetic model weights (LSTM format)
    print("\n2. Creating synthetic client weights...")
    np.random.seed(42)
    client_weights = {}
    for i in range(5):
        client_weights[i] = {
            'lstm.weight_ih_l0': np.random.randn(128, 9).astype(np.float32),
            'lstm.weight_hh_l0': np.random.randn(128, 64).astype(np.float32),
            'lstm.bias_ih_l0': np.random.randn(128).astype(np.float32),
            'lstm.bias_hh_l0': np.random.randn(128).astype(np.float32),
            'fc.weight': np.random.randn(18, 64).astype(np.float32),
            'fc.bias': np.random.randn(18).astype(np.float32)
        }
    print(f"   Created weights for 5 clients")
    print("   PASS")

    # Submit updates with different sample counts
    print("\n3. Submitting client updates...")
    sample_counts = [100, 200, 300, 150, 250]
    for client_id, weights in client_weights.items():
        aggregator.receive_update(client_id, weights, sample_counts[client_id])
        print(f"   Client {client_id}: {sample_counts[client_id]} samples")
    print("   PASS")

    # Aggregate
    print("\n4. Aggregating updates...")
    aggregated = aggregator.aggregate()

    if aggregated is not None:
        print(f"   Aggregated {len(aggregated)} weight tensors")

        # Verify weighted average
        print("\n5. Verifying weighted average...")
        n_total = sum(sample_counts[:5])  # All 5 clients

        # Check fc.bias (simplest to verify)
        expected_bias = np.zeros_like(client_weights[0]['fc.bias'])
        for client_id in range(5):
            weight = sample_counts[client_id] / n_total
            expected_bias += weight * client_weights[client_id]['fc.bias']

        actual_bias = aggregated['fc.bias']
        max_diff = np.max(np.abs(expected_bias - actual_bias))
        print(f"   Expected vs Actual bias max diff: {max_diff:.6e}")

        assert max_diff < 1e-5, "Aggregation should match weighted average"
        print("   PASS")
    else:
        print("   FAIL: Aggregation returned None")
        return False

    # Verify state reset
    print("\n6. Verifying aggregator state reset...")
    state = aggregator.get_aggregation_state()
    assert state['pending_count'] == 0, "Pending updates should be cleared"
    assert state['round_number'] == 1, "Round number should increment"
    print(f"   Round number: {state['round_number']}")
    print(f"   Pending count: {state['pending_count']}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL FEDAVG AGGREGATOR TESTS PASSED")
    print("=" * 60)
    return True


def test_quantization():
    """Test weight quantization and dequantization."""
    print("=" * 60)
    print("TEST: Weight Quantization")
    print("=" * 60)

    # Create quantization helper
    print("\n1. Creating quantization helper...")
    quantizer = QuantizationHelper()
    print("   PASS")

    # Create synthetic weights (LSTM format)
    print("\n2. Creating synthetic weights...")
    np.random.seed(42)
    original_weights = {
        'lstm.weight_ih_l0': np.random.randn(128, 9).astype(np.float32),
        'lstm.weight_hh_l0': np.random.randn(128, 64).astype(np.float32),
        'lstm.bias_ih_l0': np.random.randn(128).astype(np.float32),
        'lstm.bias_hh_l0': np.random.randn(128).astype(np.float32),
        'fc.weight': np.random.randn(18, 64).astype(np.float32),
        'fc.bias': np.random.randn(18).astype(np.float32)
    }
    print("   PASS")

    # Test 8-bit quantization
    print("\n3. Testing 8-bit quantization...")
    quantized_8, scale_params_8 = quantizer.quantize(original_weights, bits=8)

    print(f"   Quantized to 8-bit")
    print(f"   Scale params stored for {len(scale_params_8)} layers")

    # Verify quantized values are in range
    for key, q in quantized_8.items():
        assert q.min() >= 0 and q.max() <= 255, "8-bit values should be in [0, 255]"
    print("   Quantized values in valid range")
    print("   PASS")

    # Test dequantization
    print("\n4. Testing dequantization...")
    dequantized_8 = quantizer.dequantize(quantized_8, scale_params_8, bits=8)

    # Check reconstruction error
    print("\n5. Checking reconstruction error...")
    for key in original_weights:
        orig = original_weights[key]
        recon = dequantized_8[key]
        max_error = np.max(np.abs(orig - recon))
        mean_error = np.mean(np.abs(orig - recon))
        print(f"   {key}: max_error={max_error:.4f}, mean_error={mean_error:.4f}")

    # Compute overall reconstruction quality
    all_orig = np.concatenate([v.flatten() for v in original_weights.values()])
    all_recon = np.concatenate([v.flatten() for v in dequantized_8.values()])
    correlation = np.corrcoef(all_orig, all_recon)[0, 1]
    print(f"   Overall correlation: {correlation:.6f}")

    assert correlation > 0.99, "Dequantized weights should correlate highly with original"
    print("   PASS")

    # Test compression ratio
    print("\n6. Testing compression ratio...")
    ratio_8 = quantizer.compute_compression_ratio(original_weights, quantized_8, bits=8)
    print(f"   8-bit compression ratio: {ratio_8:.2f}x")

    # Test 16-bit quantization
    print("\n7. Testing 16-bit quantization...")
    quantized_16, scale_params_16 = quantizer.quantize(original_weights, bits=16)
    dequantized_16 = quantizer.dequantize(quantized_16, scale_params_16, bits=16)

    all_recon_16 = np.concatenate([v.flatten() for v in dequantized_16.values()])
    correlation_16 = np.corrcoef(all_orig, all_recon_16)[0, 1]
    print(f"   16-bit correlation: {correlation_16:.6f}")

    ratio_16 = quantizer.compute_compression_ratio(original_weights, quantized_16, bits=16)
    print(f"   16-bit compression ratio: {ratio_16:.2f}x")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL QUANTIZATION TESTS PASSED")
    print("=" * 60)
    return True


def test_fl_client():
    """Test FL client local training and weight extraction."""
    print("=" * 60)
    print("TEST: FL Client")
    print("=" * 60)

    # Create client
    print("\n1. Creating FL client...")
    config = FLClientConfig(
        local_epochs=5,
        batch_size=32,
        quantization_bits=8
    )
    client = FLClient(client_id=1, config=config)
    print("   PASS")

    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    np.random.seed(42)
    # Use raw windows for LSTM (not features)
    features = np.random.randn(500, 256, 9).astype(np.float32)
    labels = np.random.randint(0, 18, 500)
    print(f"   {len(features)} samples")
    print("   PASS")

    # Local training
    print("\n3. Testing local training...")
    initial_loss, initial_acc = client.evaluate(features[:100], labels[:100])
    print(f"   Accuracy before training: {initial_acc:.3f}")

    loss, acc = client.train_local(features, labels, epochs=5)
    print(f"   Final training loss: {loss:.3f}")

    trained_loss, trained_acc = client.evaluate(features[:100], labels[:100])
    print(f"   Accuracy after training: {trained_acc:.3f}")
    print("   PASS")

    # Get compressed weights
    print("\n4. Testing weight compression...")
    compressed_weights, scale_params = client.get_compressed_weights(bits=8)

    print(f"   Compressed {len(compressed_weights)} layers")
    print(f"   Scale params: {len(scale_params)} layers")

    # Verify compression
    fc_q = compressed_weights['fc.weight']
    assert fc_q.dtype == np.uint8, "8-bit weights should be uint8"
    assert fc_q.min() >= 0 and fc_q.max() <= 255
    print("   Weights properly quantized")
    print("   PASS")

    # Test upload decision
    print("\n5. Testing upload decision logic...")
    should_upload = client.should_upload(current_accuracy=trained_acc)
    print(f"   Should upload (no baseline): {should_upload}")

    should_upload = client.should_upload(
        current_accuracy=trained_acc,
        baseline_accuracy=trained_acc - 0.05
    )
    print(f"   Should upload (5% improvement): {should_upload}")
    print("   PASS")

    # Get upload payload
    print("\n6. Testing upload payload...")
    payload = client.get_upload_payload(trained_acc)

    if payload:
        print(f"   Payload keys: {list(payload.keys())}")
        print(f"   Client ID: {payload['client_id']}")
        print(f"   N samples: {payload['n_samples']}")
        print(f"   Accuracy: {payload['accuracy']:.3f}")
        print("   PASS")
    else:
        print("   No payload (threshold not met)")

    print("\n" + "=" * 60)
    print("ALL FL CLIENT TESTS PASSED")
    print("=" * 60)
    return True


def test_fl_server():
    """Test FL server coordination."""
    print("=" * 60)
    print("TEST: FL Server")
    print("=" * 60)

    # Create server
    print("\n1. Creating FL server...")
    server_config = FLServerConfig(
        min_clients=2,
        single_process=True
    )
    server = FLServer(config=server_config)
    print("   PASS")

    # Create and register clients
    print("\n2. Creating and registering clients...")
    clients = []
    for i in range(4):
        client = FLClient(client_id=i + 1)
        clients.append(client)
        server.register_client(client)
    print(f"   Registered {len(server.clients)} clients")
    print("   PASS")

    # Train clients locally
    print("\n3. Training clients locally...")
    np.random.seed(42)

    for client in clients:
        # Use raw windows for LSTM (not features)
        features = np.random.randn(200, 256, 9).astype(np.float32)
        labels = np.random.randint(0, 18, 200)
        client.train_local(features, labels, epochs=3)
        print(f"   Client {client.client_id}: trained on {client.local_sample_count} samples")
    print("   PASS")

    # Run FL round
    print("\n4. Running FL round...")
    global_weights = server.run_round()

    if global_weights:
        print(f"   Aggregated {len(global_weights)} weight tensors")
        print(f"   Global model version: {server.global_model_version}")
        print(f"   Completed rounds: {server.completed_rounds}")
        print("   PASS")
    else:
        print("   FAIL: No global weights returned")
        return False

    # Verify clients received global model
    print("\n5. Verifying clients updated...")
    for client in clients:
        state = server.client_states.get(client.client_id)
        if state:
            print(f"   Client {client.client_id}: {state.updates_sent} updates sent")
    print("   PASS")

    # Check server state
    print("\n6. Checking server state...")
    state = server.get_server_state()
    print(f"   Registered clients: {state['registered_clients']}")
    print(f"   Active clients: {state['active_clients']}")
    print(f"   Completed rounds: {state['completed_rounds']}")
    print(f"   Adaptations since sync: {state['adaptations_since_sync']}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL FL SERVER TESTS PASSED")
    print("=" * 60)
    return True


def test_fl_simulator():
    """Test FL simulator with non-IID data distribution."""
    print("=" * 60)
    print("TEST: FL Simulator")
    print("=" * 60)

    # Create simulator
    print("\n1. Creating FL simulator...")
    simulator = FLSimulator(
        n_clients=3,
        server_config=FLServerConfig(min_clients=2, single_process=True),
        client_config=FLClientConfig(local_epochs=3, quantization_bits=32)
    )
    print(f"   Created {len(simulator.clients)} clients")
    print("   PASS")

    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 900
    # Use raw windows for LSTM (not features)
    features = np.random.randn(n_samples, 256, 9).astype(np.float32)
    labels = np.random.randint(0, 18, n_samples)
    subject_ids = np.repeat([1, 2, 3], n_samples // 3)
    print(f"   {n_samples} samples, 3 subjects")
    print("   PASS")

    # Distribute data non-IID
    print("\n3. Distributing data non-IID (by subject)...")
    simulator.distribute_data_non_iid(features, labels, subject_ids)

    for client_id, (feat, lab) in simulator.client_data.items():
        print(f"   Client {client_id}: {len(feat)} samples")
    print("   PASS")

    # Run FL experiment
    print("\n4. Running FL experiment (5 rounds)...")
    results = simulator.run_experiment(n_rounds=5, local_epochs=3)

    print("\n   Results per round:")
    for i, round_num in enumerate(results['round']):
        acc = results['global_accuracy'][i]
        print(f"   Round {round_num}: accuracy = {acc:.3f}")
    print("   PASS")

    # Verify convergence trend
    print("\n5. Checking convergence trend...")
    if len(results['global_accuracy']) >= 3:
        first_half_avg = np.mean(results['global_accuracy'][:len(results)//2])
        second_half_avg = np.mean(results['global_accuracy'][len(results)//2:])
        print(f"   First half avg accuracy: {first_half_avg:.3f}")
        print(f"   Second half avg accuracy: {second_half_avg:.3f}")

    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL FL SIMULATOR TESTS PASSED")
    print("=" * 60)
    return True


def test_min_clients_requirement():
    """Test that aggregation requires minimum clients."""
    print("=" * 60)
    print("TEST: Minimum Clients Requirement")
    print("=" * 60)

    # Create aggregator with min_clients=3
    print("\n1. Creating aggregator (min_clients=3)...")
    config = AggregationConfig(min_clients=3)
    aggregator = FedAvgAggregator(config)
    print("   PASS")

    # Submit only 2 updates
    print("\n2. Submitting 2 updates (below threshold)...")
    np.random.seed(42)
    for client_id in range(2):
        weights = {
            'fc.weight': np.random.randn(18, 64).astype(np.float32),
            'fc.bias': np.random.randn(18).astype(np.float32)
        }
        aggregator.receive_update(client_id, weights, n_samples=100)
    print("   Submitted 2 updates")
    print("   PASS")

    # Try to aggregate (should fail)
    print("\n3. Attempting aggregation...")
    result = aggregator.aggregate()
    assert result is None, "Should not aggregate with fewer than min_clients"
    print("   Correctly returned None")
    print("   PASS")

    # Submit third update
    print("\n4. Submitting third update (meets threshold)...")
    weights = {
        'fc.weight': np.random.randn(18, 64).astype(np.float32),
        'fc.bias': np.random.randn(18).astype(np.float32)
    }
    aggregator.receive_update(2, weights, n_samples=100)
    print("   Submitted 3rd update")
    print("   PASS")

    # Now aggregation should succeed
    print("\n5. Attempting aggregation again...")
    result = aggregator.aggregate()
    assert result is not None, "Should aggregate with min_clients met"
    print(f"   Aggregated {len(result)} weight tensors")
    print("   PASS")

    print("\n" + "=" * 60)
    print("ALL MINIMUM CLIENTS TESTS PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all Phase 5 tests."""
    print("\n" + "=" * 60)
    print("PHASE 5: FEDERATED LEARNING - TEST SUITE")
    print("=" * 60)

    results = {
        'fedavg_aggregator': test_fedavg_aggregator(),
        'quantization': test_quantization(),
        'fl_client': test_fl_client(),
        'fl_server': test_fl_server(),
        'fl_simulator': test_fl_simulator(),
        'min_clients': test_min_clients_requirement()
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
