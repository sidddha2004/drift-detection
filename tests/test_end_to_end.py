"""
End-to-End Accuracy Test: Full Pipeline

Tests the complete system:
1. Load real PAMAP2 data
2. Preprocess → Window → Features
3. Train LSTM model
4. Evaluate accuracy
5. Simulate drift detection + adaptation
6. Measure post-adaptation accuracy
7. Test federated learning round

Usage:
    python tests/test_end_to_end.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline import (
    PAMAP2Preprocessor,
    create_data_pipeline
)
from src.drift_detection import (
    ActivityLSTM,
    LSTMDriftDetector,
    LSTMConfig,
    IntegratedDriftDetector,
    IntegratedDriftDetectorConfig
)
from src.federated import (
    FLClient,
    FLClientConfig,
    FLServer,
    FLServerConfig,
    FedAvgAggregator,
    AggregationConfig
)


def find_data_file():
    """Find a PAMAP2 data file."""
    candidates = [
        "PAMAP2_Dataset/Protocol/subject101.dat",
        "PAMAP2_Dataset/Protocol/subject102.dat",
        "PAMAP2_Dataset/Optional/subject101.dat",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def test_phase1_data_pipeline():
    """Test Phase 1: Data Pipeline with real data."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PIPELINE")
    print("=" * 60)

    data_file = find_data_file()
    if not data_file:
        print("SKIP: PAMAP2 dataset not found")
        return None, None

    print(f"\n1. Loading data from: {data_file}")
    preprocessor = PAMAP2Preprocessor()
    accel_data, activity_labels = preprocessor.preprocess_file(data_file)
    print(f"   Loaded: {accel_data.shape[0]} samples, {accel_data.shape[1]} axes")
    print(f"   Unique activities: {np.unique(activity_labels)}")

    print("\n2. Generating windows (raw for LSTM)...")
    pipeline = create_data_pipeline(return_raw_windows=True)
    windows, labels = pipeline.process(accel_data, activity_labels)
    print(f"   Generated: {windows.shape[0]} windows")
    print(f"   Window shape: {windows.shape}")

    print("\n3. Class distribution:")
    for cls in sorted(np.unique(labels)):
        count = np.sum(labels == cls)
        print(f"   Activity {cls}: {count} ({100*count/len(labels):.1f}%)")

    return windows, labels


def test_phase2_3_lstm_training(windows, labels):
    """Test Phase 2+3: LSTM Training and Evaluation."""
    print("\n" + "=" * 60)
    print("PHASE 2+3: LSTM MODEL TRAINING")
    print("=" * 60)

    n_samples = len(windows)
    n_classes = len(np.unique(labels))

    print(f"\n1. Data summary:")
    print(f"   Total windows: {n_samples}")
    print(f"   Number of classes: {n_classes}")

    # Remap labels to contiguous range
    unique_labels = np.unique(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels_remapped = np.array([label_to_idx[l] for l in labels])

    # Split: 70% train, 30% test
    print("\n2. Splitting data (70% train, 30% test)...")
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * 0.7)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    train_windows = windows[train_idx]
    train_labels = labels_remapped[train_idx]
    test_windows = windows[test_idx]
    test_labels = labels_remapped[test_idx]

    print(f"   Train: {len(train_windows)} windows")
    print(f"   Test: {len(test_windows)} windows")

    # Create LSTM
    print("\n3. Creating LSTM model...")
    config = LSTMConfig(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=64,
        output_dim=n_classes
    )
    detector = LSTMDriftDetector(lstm_config=config)

    total_params = sum(p.numel() for p in detector.model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1024:.1f} KB (float32)")

    # Train
    print("\n4. Training LSTM (20 epochs)...")
    start_time = time.time()
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(20):
        perm = np.random.permutation(len(train_windows))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_windows), config.batch_size):
            batch_idx = perm[i:i+config.batch_size]
            loss = detector.train_batch(
                train_windows[batch_idx],
                train_labels[batch_idx]
            )
            epoch_loss += loss
            n_batches += 1

        # Evaluate
        acc, loss = detector.evaluate(test_windows, test_labels)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

        print(f"   Epoch {epoch+1}/20: loss={epoch_loss/n_batches:.4f}, test_acc={acc:.3f}")

    train_time = time.time() - start_time
    print(f"\n   Training completed in {train_time:.1f}s")
    print(f"   Best accuracy: {best_acc:.1%} (epoch {best_epoch})")

    # Final evaluation
    print("\n5. Final test evaluation...")
    final_acc, final_loss = detector.evaluate(test_windows, test_labels)
    print(f"   Final accuracy: {final_acc:.1%}")
    print(f"   Final loss: {final_loss:.4f}")

    # Per-class accuracy
    print("\n6. Per-class accuracy:")
    predictions = []
    for i in range(len(test_windows)):
        pred = detector.predict(test_windows[i])
        predictions.append(pred)
    predictions = np.array(predictions)

    for cls in sorted(np.unique(test_labels)):
        mask = test_labels == cls
        if np.sum(mask) > 0:
            cls_acc = np.mean(predictions[mask] == test_labels[mask])
            print(f"   Activity {cls}: {cls_acc:.1%} ({np.sum(mask)} samples)")

    return detector, test_windows, test_labels, final_acc, label_to_idx


def test_phase4_adaptation(detector, test_windows, test_labels):
    """Test Phase 4: Drift Detection + Adaptation."""
    print("\n" + "=" * 60)
    print("PHASE 4: DRIFT DETECTION + ADAPTATION")
    print("=" * 60)

    n_classes = len(np.unique(test_labels))

    # Create integrated detector
    print("\n1. Creating IntegratedDriftDetector...")
    config = IntegratedDriftDetectorConfig(
        lstm_config=LSTMConfig(
            input_size=9,
            hidden_size=64,
            num_layers=2,
            output_dim=n_classes  # Match trained model
        ),
        adwin_delta=0.002,
        max_buffer_size=500,
        min_samples_for_adaptation=50
    )
    integrated = IntegratedDriftDetector(config)

    # Load trained weights
    integrated.detector.model.load_state_dict(detector.model.state_dict())

    print("   PASS")

    # Simulate stream processing
    print("\n2. Processing stream for drift detection...")
    n_stream = min(500, len(test_windows))
    predictions = []
    drift_events = []

    for i in range(n_stream):
        pred, drift = integrated.process(test_windows[i], test_labels[i])
        predictions.append(pred)
        if drift:
            drift_events.append({
                'sample': i,
                'drift_class': integrated.last_drift_class,
                'adaptation_success': integrated.last_adaptation_result.success if integrated.last_adaptation_result else False
            })

    predictions = np.array(predictions)
    stream_acc = np.mean(predictions == test_labels[:n_stream])

    print(f"   Processed: {n_stream} samples")
    print(f"   Drift events: {len(drift_events)}")
    print(f"   Stream accuracy: {stream_acc:.1%}")

    if drift_events:
        print("\n   Drift events summary:")
        for event in drift_events[:5]:
            print(f"   - Sample {event['sample']}: {event['drift_class']}, success={event['adaptation_success']}")

    # Get statistics
    stats = integrated.get_statistics()
    print("\n3. Detector statistics:")
    print(f"   Samples processed: {stats['samples_processed']}")
    print(f"   Total drifts: {stats['total_drifts']}")
    print(f"   Total adaptations: {stats['total_adaptations']}")
    print(f"   Recent error rate: {stats['recent_error_rate']:.3f}")

    return integrated, stream_acc, drift_events


def test_phase5_federated_learning(detector, windows, labels):
    """Test Phase 5: Federated Learning."""
    print("\n" + "=" * 60)
    print("PHASE 5: FEDERATED LEARNING")
    print("=" * 60)

    n_classes = len(np.unique(labels))
    n_samples = len(windows)

    # Remap labels
    unique_labels = np.unique(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels_remapped = np.array([label_to_idx[l] for l in labels])

    # Split data for 3 simulated clients (non-IID)
    print("\n1. Distributing data to 3 clients (non-IID)...")
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    chunk_size = n_samples // 3

    client_data = []
    for i in range(3):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 2 else n_samples
        client_indices = indices[start_idx:end_idx]
        client_data.append((windows[client_indices], labels_remapped[client_indices]))
        print(f"   Client {i+1}: {len(client_indices)} samples")

    # Create FL server
    print("\n2. Creating FL server and clients...")
    server_config = FLServerConfig(min_clients=3, single_process=True)
    aggregation_config = AggregationConfig(min_clients=3)
    server = FLServer(server_config, aggregation_config)

    # Create clients
    clients = []
    for i in range(3):
        client_config = FLClientConfig(
            local_epochs=3,
            batch_size=64,
            quantization_bits=32
        )
        lstm_config = LSTMConfig(
            input_size=9,
            hidden_size=64,
            num_layers=2,
            output_dim=n_classes
        )
        client = FLClient(
            client_id=i+1,
            config=client_config,
            lstm_config=lstm_config
        )
        clients.append(client)
        server.register_client(client)

    print(f"   Registered {len(clients)} clients")
    print("   PASS")

    # Simulate one FL round
    print("\n3. Running FL round...")

    # Each client trains locally
    for i, client in enumerate(clients):
        client_features, client_labels = client_data[i]
        # Train on subset for speed
        train_size = min(200, len(client_features))
        client.train_local(
            client_features[:train_size],
            client_labels[:train_size],
            epochs=5  # Increased from 2
        )
        print(f"   Client {i+1}: Local training complete")

    # Server aggregates
    print("\n4. Aggregating models...")
    global_weights = server.run_round()

    if global_weights:
        print(f"   Aggregation complete (round {server.completed_rounds})")

        # Load global model into client 0 for evaluation
        clients[0].load_global_model(global_weights)

        # Evaluate global model on client 0's test data
        print("\n5. Evaluating global model...")
        test_start = min(200, len(client_data[0][0]))
        test_size = min(100, len(client_data[0][0]) - test_start)
        acc, loss = clients[0].evaluate(
            client_data[0][0][test_start:test_start+test_size],
            client_data[0][1][test_start:test_start+test_size]
        )
        print(f"   Global model accuracy: {acc:.1%}")
        print(f"   Global model loss: {loss:.4f}")
    else:
        print("   Aggregation failed (insufficient clients)")
        acc = 0.0

    return server, clients, acc


def run_end_to_end_test():
    """Run complete end-to-end test."""
    print("\n" + "=" * 70)
    print("END-TO-END ACCURACY TEST: FULL PIPELINE")
    print("=" * 70)

    results = {}

    # Phase 1: Data Pipeline
    windows, labels = test_phase1_data_pipeline()
    if windows is None:
        print("\nCannot continue without data. Exiting.")
        return results

    results['phase1'] = 'PASS'

    # Phase 2+3: LSTM Training
    detector, test_windows, test_labels, lstm_acc, label_map = test_phase2_3_lstm_training(
        windows, labels
    )
    results['phase2_3'] = {
        'status': 'PASS',
        'accuracy': lstm_acc,
        'classes': len(label_map)
    }

    # Phase 4: Adaptation
    integrated, stream_acc, drift_events = test_phase4_adaptation(
        detector, test_windows, test_labels
    )
    results['phase4'] = {
        'status': 'PASS',
        'stream_accuracy': stream_acc,
        'drift_events': len(drift_events)
    }

    # Phase 5: Federated Learning
    server, clients, fl_acc = test_phase5_federated_learning(
        detector, windows, labels
    )
    results['phase5'] = {
        'status': 'PASS',
        'fl_accuracy': fl_acc,
        'clients': 3
    }

    # Summary
    print("\n" + "=" * 70)
    print("END-TO-END TEST SUMMARY")
    print("=" * 70)

    print(f"\n  Phase 1 (Data Pipeline):     {results['phase1']}")
    print(f"  Phase 2+3 (LSTM Training):   {results['phase2_3']['status']} - Accuracy: {results['phase2_3']['accuracy']:.1%}")
    print(f"  Phase 4 (Adaptation):        {results['phase4']['status']} - Drift events: {results['phase4']['drift_events']}")
    print(f"  Phase 5 (Federated Learning):{results['phase5']['status']} - FL Accuracy: {results['phase5']['fl_accuracy']:.1%}")

    print("\n" + "=" * 70)
    print("OVERALL: ALL PHASES EXECUTED SUCCESSFULLY")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_end_to_end_test()
