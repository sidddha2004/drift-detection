# Multiclass Drift Detection for Edge HAR

A two-stage drift detection system for resource-constrained edge devices running Human Activity Recognition.

## Overview

This system implements:
1. **Binary drift detection** (always-on, lightweight ADWIN-based)
2. **Multiclass drift classification** (on-demand, 4 drift types)
3. **Type-specific adaptation** (different strategies per drift type)
4. **Federated learning** (weight sharing without raw data)

## Target Hardware

- Raspberry Pi Zero 2 W (512MB RAM, 1GHz CPU)
- Simulated edge clients using PAMAP2 dataset subjects

## Project Structure

```
├── config/              # YAML configuration files
├── src/
│   ├── data_pipeline/   # Phase 1: Preprocessing, windowing, drift injection
│   ├── drift_detection/ # Phase 2-3: Binary detection + multiclass classification
│   ├── adaptation/      # Phase 4: Type-specific adaptation strategies
│   ├── federated/       # Phase 5: FL client + server
│   ├── models/          # Neural network architectures
│   └── evaluation/      # Phase 6: Metrics + baselines
├── tests/               # Unit and integration tests
├── experiments/         # Experiment outputs
└── results/             # CSVs and plots
```

## Quick Start

### Prerequisites

```bash
pip install torch numpy pandas pyyaml river psutil matplotlib seaborn
```

### Run Phase 1-2 Pipeline

```bash
# Test preprocessing
python tests/test_preprocessing.py

# Test drift injection
python tests/test_drift_injection.py

# Test binary drift detection
python tests/test_drift_detection.py
```

## Dataset

Uses PAMAP2 Physical Activity Monitoring Dataset:
- 9 subjects, 18 activities
- 3 IMU sensors (wrist, chest, ankle) @ 100Hz
- Download: [PAMAP2 Dataset](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)

## Configuration

Edit YAML files in `config/`:
- `drift_config.yaml` - Drift injection parameters
- `model_config.yaml` - Model architecture, ADWIN settings
- `fl_config.yaml` - Federated learning parameters

## License

For academic research use.

## Citation

If you use this code, please cite:
```
@inproceedings{drift-detection-edge-har,
  title={Multiclass Drift Detection for Low-End Edge Human Activity Recognition},
  author={Your Name},
  booktitle={IEEE PerCom / ISWC},
  year={2026}
}
```
