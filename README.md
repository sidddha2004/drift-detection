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

## Status

Research restructure in progress. See `docs/RESEARCH_PLAN.md` (plan) and `docs/decisions.md` (log).

## Project Structure

```
├── config/              # YAML configuration files
├── src/
│   ├── data/            # PAMAP2 bout loader, leak-free federation splits
│   ├── drift/           # drift schedules, real + virtual injectors, per-client streams
│   ├── drift_detection/ # LSTM model, ADWIN, drift-type classifier
│   ├── adaptation/      # type-specific adaptation
│   └── federated/       # FL client / server / aggregation
├── tests/               # pytest suite (python -m pytest)
├── experiments/         # runnable experiment entry points
├── results/             # outputs
├── docs/                # plan, decision log, specs
└── legacy/              # retired code kept for reference
```

## Quick Start

```bash
pip install -r requirements.txt   # see the CUDA note inside for the GPU build of torch
python -m pytest
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
