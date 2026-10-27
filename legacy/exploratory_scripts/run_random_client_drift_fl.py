"""
Random Client Drift Federated Learning Simulation.

This experiment uses the existing PAMAP2 preprocessing, LSTM FL client/server,
and drift injector to simulate federated rounds where a random subset of
clients experiences drift in a given round.

The goal is to measure how much global accuracy can be maintained when only
some clients drift intermittently across rounds.

Usage:
    python experiments/run_random_client_drift_fl.py

Examples:
    python experiments/run_random_client_drift_fl.py --n-rounds 8 --drift-probability 0.4
    python experiments/run_random_client_drift_fl.py --n-clients 5 --output results/drift_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_pipeline import (  # noqa: E402
    PAMAP2Preprocessor,
    DriftConfig,
    DriftInjector,
    DriftType,
    create_data_pipeline,
)
from src.drift_detection import LSTMConfig  # noqa: E402
from src.federated import FLClient, FLClientConfig, FLServer, FLServerConfig  # noqa: E402


DEFAULT_PROTOCOL_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]


@dataclass
class ClientDataset:
    client_id: int
    subject_id: int
    train_raw: np.ndarray
    train_labels: np.ndarray
    test_raw: np.ndarray
    test_labels: np.ndarray
    test_windows: np.ndarray
    test_window_labels: np.ndarray


@dataclass
class DriftEventSummary:
    round_num: int
    client_id: int
    subject_id: int
    drift_type: str
    magnitude: float
    injection_point: int
    n_samples_affected: int


@dataclass
class RoundSummary:
    round_num: int
    drifted_clients: List[int] = field(default_factory=list)
    drift_events: List[DriftEventSummary] = field(default_factory=list)
    pre_sync_client_accuracies: Dict[int, float] = field(default_factory=dict)
    post_sync_client_accuracies: Dict[int, float] = field(default_factory=dict)
    drifted_client_accuracy: float = 0.0
    non_drifted_client_accuracy: float = 0.0
    global_accuracy: float = 0.0
    mean_pre_sync_accuracy: float = 0.0
    mean_post_sync_accuracy: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "round_num": self.round_num,
            "drifted_clients": self.drifted_clients,
            "drift_events": [asdict(event) for event in self.drift_events],
            "pre_sync_client_accuracies": self.pre_sync_client_accuracies,
            "post_sync_client_accuracies": self.post_sync_client_accuracies,
            "drifted_client_accuracy": self.drifted_client_accuracy,
            "non_drifted_client_accuracy": self.non_drifted_client_accuracy,
            "global_accuracy": self.global_accuracy,
            "mean_pre_sync_accuracy": self.mean_pre_sync_accuracy,
            "mean_post_sync_accuracy": self.mean_post_sync_accuracy,
        }


def parse_subject_ids(raw_value: Optional[str]) -> List[int]:
    if not raw_value:
        return DEFAULT_PROTOCOL_SUBJECTS.copy()

    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def find_subject_files(data_root: Path, protocol: bool, subject_ids: Sequence[int]) -> Dict[int, Path]:
    dataset_subdir = "Protocol" if protocol else "Optional"
    base_dir = data_root / "PAMAP2_Dataset" / dataset_subdir

    subject_files: Dict[int, Path] = {}
    for subject_id in subject_ids:
        candidate = base_dir / f"subject{subject_id}.dat"
        if candidate.exists():
            subject_files[subject_id] = candidate

    return subject_files


def load_subject_raw_data(subject_files: Dict[int, Path]) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, int]]:
    """Load raw subject streams and build a global label remapping."""
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    label_values: List[int] = []

    for subject_id, subject_path in subject_files.items():
        preprocessor = PAMAP2Preprocessor()
        accel_data, labels = preprocessor.preprocess_file(str(subject_path))
        subject_data[subject_id] = (accel_data.astype(np.float32), labels.astype(np.int64))
        label_values.extend(int(label) for label in np.unique(labels))

    unique_labels = sorted(set(label_values))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    return subject_data, label_map


def split_and_window_subject(
    accel_data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    train_ratio: float,
) -> ClientDataset:
    """Split a subject stream into train/test segments and window both."""
    pipeline = create_data_pipeline(window_size=window_size, stride=stride, return_raw_windows=True)

    split_point = int(len(accel_data) * train_ratio)
    min_segment = window_size + stride
    split_point = max(min_segment, split_point)
    split_point = min(split_point, len(accel_data) - min_segment)

    train_raw = accel_data[:split_point]
    train_labels = labels[:split_point]
    test_raw = accel_data[split_point:]
    test_labels = labels[split_point:]

    _, _ = pipeline.process(train_raw, train_labels)
    test_windows, test_window_labels = pipeline.process(test_raw, test_labels)

    return ClientDataset(
        client_id=-1,
        subject_id=-1,
        train_raw=train_raw,
        train_labels=train_labels,
        test_raw=test_raw,
        test_labels=test_labels,
        test_windows=test_windows,
        test_window_labels=test_window_labels,
    )


def build_client_datasets(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    label_map: Dict[int, int],
    subject_ids: Sequence[int],
    train_ratio: float,
    window_size: int,
    stride: int,
) -> List[ClientDataset]:
    datasets: List[ClientDataset] = []

    for subject_id in subject_ids:
        if subject_id not in subject_data:
            continue

        accel_data, labels = subject_data[subject_id]
        remapped_labels = np.array([label_map[int(label)] for label in labels], dtype=np.int64)

        dataset = split_and_window_subject(
            accel_data=accel_data,
            labels=remapped_labels,
            window_size=window_size,
            stride=stride,
            train_ratio=train_ratio,
        )
        dataset.subject_id = subject_id
        datasets.append(dataset)

    return datasets


def sample_drift_config(
    rng: np.random.Generator,
    train_length: int,
    drift_types: Sequence[DriftType],
    magnitude_range: Tuple[float, float],
    injection_fraction_range: Tuple[float, float],
    target_axes: Sequence[int],
) -> DriftConfig:
    drift_type = rng.choice(list(drift_types))
    magnitude = float(rng.uniform(*magnitude_range))
    injection_fraction = float(rng.uniform(*injection_fraction_range))
    injection_point = int(train_length * injection_fraction)
    injection_point = max(0, min(train_length - 1, injection_point))

    config = DriftConfig(
        drift_type=drift_type,
        magnitude=magnitude,
        injection_point=injection_point,
        seed=int(rng.integers(0, 1_000_000)),
    )

    if drift_type == DriftType.GRADUAL:
        config.ramp_samples = int(rng.integers(500, 5000))
    elif drift_type == DriftType.RECURRING:
        config.period = int(rng.integers(5000, 15000))
        config.duration = int(rng.integers(500, 3000))
    elif drift_type == DriftType.FEATURE_LEVEL:
        config.target_axis = int(rng.choice(list(target_axes)))

    return config


def inject_client_drift(
    client_dataset: ClientDataset,
    drift_config: DriftConfig,
) -> Tuple[np.ndarray, int]:
    injector = DriftInjector(drift_config)
    drifted_train_raw = injector.inject_drift(client_dataset.train_raw)
    drift_mask = injector.get_drift_mask(len(client_dataset.train_raw))
    n_samples_affected = int(np.sum(drift_mask))
    return drifted_train_raw, n_samples_affected


def compute_weighted_accuracy(client_accuracies: Dict[int, float], client_datasets: Sequence[ClientDataset]) -> float:
    total_samples = 0
    weighted_sum = 0.0

    dataset_by_client = {dataset.client_id: dataset for dataset in client_datasets}
    for client_id, accuracy in client_accuracies.items():
        n_samples = len(dataset_by_client[client_id].test_window_labels)
        total_samples += n_samples
        weighted_sum += accuracy * n_samples

    return weighted_sum / total_samples if total_samples else 0.0


def compute_group_accuracy(client_accuracies: Dict[int, float], client_ids: Sequence[int]) -> float:
    group_values = [client_accuracies[client_id] for client_id in client_ids if client_id in client_accuracies]
    return float(np.mean(group_values)) if group_values else 0.0


def run_simulation(
    data_root: Path,
    subject_ids: Sequence[int],
    n_clients: int,
    n_rounds: int,
    local_epochs: int,
    drift_probability: float,
    train_ratio: float,
    window_size: int,
    stride: int,
    seed: int,
    protocol: bool,
    output_path: Optional[Path] = None,
) -> Dict:
    rng = np.random.default_rng(seed)

    subject_files = find_subject_files(data_root, protocol, subject_ids)
    if not subject_files:
        raise FileNotFoundError("No PAMAP2 subject files were found for the requested configuration.")

    subject_data, label_map = load_subject_raw_data(subject_files)
    selected_subject_ids = list(subject_files.keys())[:n_clients]
    client_datasets = build_client_datasets(
        subject_data=subject_data,
        label_map=label_map,
        subject_ids=selected_subject_ids,
        train_ratio=train_ratio,
        window_size=window_size,
        stride=stride,
    )

    if len(client_datasets) < 2:
        raise RuntimeError("Need at least two clients with usable train/test splits for the simulation.")

    n_classes = len(label_map)
    lstm_config = LSTMConfig(
        input_size=9,
        seq_len=window_size,
        output_dim=n_classes,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=64,
    )

    client_config = FLClientConfig(
        local_epochs=local_epochs,
        batch_size=64,
        quantization_bits=32,
        min_samples_for_upload=10,
    )

    server = FLServer(
        config=FLServerConfig(
            min_clients=max(2, min(3, len(client_datasets))),
            single_process=True,
        )
    )

    clients: List[FLClient] = []
    for dataset in client_datasets:
        client = FLClient(
            client_id=dataset.subject_id,
            config=client_config,
            lstm_config=lstm_config,
        )
        dataset.client_id = client.client_id
        clients.append(client)
        server.register_client(client)

    # Start all clients from the same initial weights.
    initial_weights = clients[0].detector.model.get_weights()
    server.global_weights = {name: value.copy() for name, value in initial_weights.items()}
    server.broadcast()

    round_history: List[Dict] = []

    pipeline = create_data_pipeline(window_size=window_size, stride=stride, return_raw_windows=True)
    drift_types = [DriftType.SUDDEN, DriftType.GRADUAL, DriftType.RECURRING, DriftType.FEATURE_LEVEL]

    print("=" * 80)
    print("RANDOM CLIENT DRIFT FL SIMULATION")
    print("=" * 80)
    print(f"Clients: {len(client_datasets)} | Rounds: {n_rounds} | Drift probability: {drift_probability:.2f}")
    print(f"Classes: {n_classes} | Window size: {window_size} | Stride: {stride}")
    print()

    for round_num in range(1, n_rounds + 1):
        drifted_clients: List[int] = []
        drift_events: List[DriftEventSummary] = []
        pre_sync_accuracies: Dict[int, float] = {}

        for client, dataset in zip(clients, client_datasets):
            should_drift = rng.random() < drift_probability
            train_raw = dataset.train_raw

            if should_drift:
                drift_config = sample_drift_config(
                    rng=rng,
                    train_length=len(dataset.train_raw),
                    drift_types=drift_types,
                    magnitude_range=(0.5, 1.5),
                    injection_fraction_range=(0.15, 0.65),
                    target_axes=[0, 3, 6],
                )
                train_raw, n_samples_affected = inject_client_drift(dataset, drift_config)
                drifted_clients.append(client.client_id)
                drift_events.append(
                    DriftEventSummary(
                        round_num=round_num,
                        client_id=client.client_id,
                        subject_id=dataset.subject_id,
                        drift_type=drift_config.drift_type.value,
                        magnitude=drift_config.magnitude,
                        injection_point=drift_config.injection_point,
                        n_samples_affected=n_samples_affected,
                    )
                )

            train_windows, train_window_labels = pipeline.process(train_raw, dataset.train_labels)
            if len(train_windows) == 0 or len(dataset.test_windows) == 0:
                continue

            client.train_local(train_windows, train_window_labels, epochs=local_epochs)
            pre_sync_acc, _ = client.evaluate(dataset.test_windows, dataset.test_window_labels)
            pre_sync_accuracies[client.client_id] = float(pre_sync_acc)

        server.run_round()

        post_sync_accuracies: Dict[int, float] = {}
        for client, dataset in zip(clients, client_datasets):
            post_sync_acc, _ = client.evaluate(dataset.test_windows, dataset.test_window_labels)
            post_sync_accuracies[client.client_id] = float(post_sync_acc)

        global_accuracy = compute_weighted_accuracy(post_sync_accuracies, client_datasets)
        drifted_client_accuracy = compute_group_accuracy(post_sync_accuracies, drifted_clients)
        non_drifted_clients = [client.client_id for client in clients if client.client_id not in drifted_clients]
        non_drifted_client_accuracy = compute_group_accuracy(post_sync_accuracies, non_drifted_clients)
        mean_pre_sync_accuracy = float(np.mean(list(pre_sync_accuracies.values()))) if pre_sync_accuracies else 0.0
        mean_post_sync_accuracy = float(np.mean(list(post_sync_accuracies.values()))) if post_sync_accuracies else 0.0

        round_summary = RoundSummary(
            round_num=round_num,
            drifted_clients=drifted_clients,
            drift_events=drift_events,
            pre_sync_client_accuracies=pre_sync_accuracies,
            post_sync_client_accuracies=post_sync_accuracies,
            drifted_client_accuracy=drifted_client_accuracy,
            non_drifted_client_accuracy=non_drifted_client_accuracy,
            global_accuracy=global_accuracy,
            mean_pre_sync_accuracy=mean_pre_sync_accuracy,
            mean_post_sync_accuracy=mean_post_sync_accuracy,
        )
        round_history.append(round_summary.to_dict())

        drift_info = ", ".join(str(client_id) for client_id in drifted_clients) if drifted_clients else "none"
        print(
            f"Round {round_num:02d} | drifted clients: {drift_info:<20} | "
            f"drifted acc: {drifted_client_accuracy:.3f} | "
            f"non-drifted acc: {non_drifted_client_accuracy:.3f} | "
            f"global acc: {global_accuracy:.3f}"
        )

    final_global_accuracy = round_history[-1]["global_accuracy"] if round_history else 0.0
    average_global_accuracy = float(np.mean([entry["global_accuracy"] for entry in round_history])) if round_history else 0.0
    drift_rounds = sum(1 for entry in round_history if entry["drifted_clients"])

    summary = {
        "seed": seed,
        "n_clients": len(client_datasets),
        "n_rounds": n_rounds,
        "n_classes": n_classes,
        "subject_ids": selected_subject_ids,
        "drift_probability": drift_probability,
        "train_ratio": train_ratio,
        "window_size": window_size,
        "stride": stride,
        "final_global_accuracy": final_global_accuracy,
        "average_global_accuracy": average_global_accuracy,
        "drift_rounds": drift_rounds,
        "rounds": round_history,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Final global accuracy:   {final_global_accuracy:.3f}")
    print(f"Average global accuracy: {average_global_accuracy:.3f}")
    print(f"Rounds with drift:       {drift_rounds}/{n_rounds}")
    if output_path is not None:
        print(f"Saved summary to:        {output_path}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate FL rounds with random client drift.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT, help="Project root containing PAMAP2_Dataset")
    parser.add_argument("--subjects", type=str, default="", help="Comma-separated subject IDs to use. Defaults to all protocol subjects.")
    parser.add_argument("--n-clients", type=int, default=9, help="Number of clients to simulate.")
    parser.add_argument("--n-rounds", type=int, default=8, help="Number of FL rounds.")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local epochs per round.")
    parser.add_argument("--drift-probability", type=float, default=0.35, help="Probability a client drifts in a round.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of each subject stream used for training.")
    parser.add_argument("--window-size", type=int, default=256, help="Window size in samples.")
    parser.add_argument("--stride", type=int, default=128, help="Window stride in samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--protocol", action="store_true", default=True, help="Use PAMAP2 Protocol dataset.")
    parser.add_argument("--optional", action="store_true", help="Use PAMAP2 Optional dataset instead of Protocol.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to store the round-by-round summary.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    protocol = not args.optional
    subject_ids = parse_subject_ids(args.subjects) if args.subjects else DEFAULT_PROTOCOL_SUBJECTS.copy()

    try:
        run_simulation(
            data_root=args.data_root,
            subject_ids=subject_ids,
            n_clients=args.n_clients,
            n_rounds=args.n_rounds,
            local_epochs=args.local_epochs,
            drift_probability=args.drift_probability,
            train_ratio=args.train_ratio,
            window_size=args.window_size,
            stride=args.stride,
            seed=args.seed,
            protocol=protocol,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"Simulation failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())