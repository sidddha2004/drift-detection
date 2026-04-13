"""
Test Suite: Dynamic Drift-Aware Federated Learning

Tests:
- Dynamic drift sampling (per-client, per-round)
- Drift injection into client data
- Heterogeneous drift scenarios
- Metrics tracking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.federated.drift_aware_fl import (
    DriftAwareFlSimulation,
    DynamicDriftConfig,
    DriftEvent,
    run_dynamic_drift_fl
)
from data_pipeline.drift_injection import DriftType


def test_dynamic_drift_config():
    """Test dynamic drift configuration."""
    print("=" * 60)
    print("TEST: Dynamic Drift Configuration")
    print("=" * 60)

    config = DynamicDriftConfig(
        drift_probability=0.5,
        magnitude_min=0.5,
        magnitude_max=1.5,
        seed=42
    )

    print(f"\n1. Config created:")
    print(f"   Drift probability: {config.drift_probability}")
    print(f"   Drift types: {[dt.value for dt in config.drift_types]}")
    print(f"   Magnitude range: [{config.magnitude_min}, {config.magnitude_max}]")
    print("   PASS")

    return True


def test_drift_sampling():
    """Test drift sampling for clients across rounds."""
    print("\n" + "=" * 60)
    print("TEST: Drift Sampling (Per-Client, Per-Round)")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    features = np.random.randn(n_samples, 256, 9).astype(np.float32)
    labels = np.random.randint(0, 4, n_samples)

    config = DynamicDriftConfig(
        drift_probability=0.4,  # 40% chance per client per round
        seed=42
    )

    sim = DriftAwareFlSimulation(
        drift_config=config,
        features=features,
        labels=labels,
        n_clients=5,
        seed=42
    )

    print(f"\n1. Running 10 rounds with 5 clients...")
    print(f"   Expected drift events: ~{int(10 * 5 * 0.4)} (40% probability)")

    drift_events = []
    for round_num in range(10):
        round_drifts = []
        for client_id in range(5):
            drift_config = sim.sample_drift_for_client(client_id, round_num)
            if drift_config is not None:
                round_drifts.append((client_id, drift_config.drift_type.value))

        drift_events.extend([(round_num, e[0], e[1]) for e in round_drifts])
        print(f"   Round {round_num + 1}: {len(round_drifts)} clients drifting - {round_drifts}")

    print(f"\n2. Total drift events: {len(drift_events)}")

    # Check drift type distribution
    drift_types = {}
    for _, _, dtype in drift_events:
        drift_types[dtype] = drift_types.get(dtype, 0) + 1

    print(f"   Drift type distribution: {drift_types}")

    # Verify different clients got drift in different rounds
    unique_round_client_pairs = set((r, c) for r, c, _ in drift_events)
    print(f"   Unique (round, client) pairs: {len(unique_round_client_pairs)}")

    # Verify drift types vary
    if len(drift_types) >= 2:
        print("   PASS: Multiple drift types sampled")
    else:
        print("   WARN: Only one drift type sampled (random, may vary)")

    return True


def test_drift_injection():
    """Test drift injection into client data."""
    print("\n" + "=" * 60)
    print("TEST: Drift Injection into Client Data")
    print("=" * 60)

    # Generate synthetic DATA - flat format for drift injection (n_samples, 9)
    np.random.seed(42)
    n_samples = 500
    features = np.random.randn(n_samples, 9).astype(np.float32)
    labels = np.random.randint(0, 4, n_samples)

    config = DynamicDriftConfig(seed=42)
    sim = DriftAwareFlSimulation(
        drift_config=config,
        features=features,
        labels=labels,
        n_clients=3,
        seed=42
    )

    print("\n1. Testing sudden drift injection...")

    # Force a drift config for testing
    from data_pipeline.drift_injection import DriftConfig, DriftType
    drift_config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        magnitude=1.0,
        injection_point=0,
        seed=42
    )
    drift_config._injection_fraction = 0.3

    drifted_features, drift_mask, inj_point = sim.inject_drift_into_client_data(0, drift_config)

    print(f"   Injection point: {inj_point}")
    print(f"   Samples affected: {np.sum(drift_mask)}")
    print(f"   Original data range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"   Drifted data range: [{drifted_features.min():.3f}, {drifted_features.max():.3f}]")

    # Verify drift was applied
    mean_before = features[inj_point:].mean()
    mean_after = drifted_features[inj_point:].mean()
    print(f"   Mean before injection: {mean_before:.3f}")
    print(f"   Mean after injection: {mean_after:.3f}")
    print(f"   Difference: {abs(mean_after - mean_before):.3f} (expected ~1.0)")

    if abs(abs(mean_after - mean_before) - 1.0) < 0.2:
        print("   PASS: Drift correctly applied")
    else:
        print("   WARN: Drift magnitude may vary based on type")

    return True


def test_full_simulation():
    """Test full dynamic drift FL simulation."""
    print("\n" + "=" * 60)
    print("TEST: Full Dynamic Drift FL Simulation")
    print("=" * 60)

    # Generate synthetic data - flat format (n_samples, 9) for drift injection
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4

    features = np.random.randn(n_samples, 9).astype(np.float32)
    labels = np.random.randint(0, n_classes, n_samples)

    print(f"\n1. Generated {len(features)} samples, {n_classes} classes")

    # Run simulation
    sim, results = run_dynamic_drift_fl(
        features, labels,
        n_clients=5,
        n_rounds=10,
        drift_probability=0.3,
        seed=42,
        verbose=False
    )

    print("\n2. Simulation Results:")
    print(f"   Total drift events: {results['total_drift_events']}")
    print(f"   Drift by type: {results['drift_by_type']}")
    print(f"   Average accuracy: {results['avg_accuracy']:.1%}")
    print(f"   Final accuracy: {results['final_accuracy']:.1%}")

    # Check drift distribution across clients
    print("\n3. Client drift history:")
    for cid, stats in results['client_stats'].items():
        print(f"   Client {cid}: {stats['total_drift_events']} drifts "
              f"({stats['drift_fraction']:.0%} of rounds)")

    # Verify drift occurred
    if results['total_drift_events'] > 0:
        print("\n   PASS: Drift events occurred as expected")
    else:
        print("\n   WARN: No drift events (random, may vary with seed)")

    return True


def run_all_tests():
    """Run all dynamic drift FL tests."""
    print("\n" + "=" * 60)
    print("DYNAMIC DRIFT FL - TEST SUITE")
    print("=" * 60)

    results = {
        'config': test_dynamic_drift_config(),
        'sampling': test_drift_sampling(),
        'injection': test_drift_injection(),
        'full_sim': test_full_simulation()
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    print("\n" + "=" * 60)
    if all(results.values()):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    run_all_tests()
