# Design Specification: Multiclass Drift Detection for Low-End Edge HAR

**Date:** 2026-03-30
**Status:** Approved for Implementation
**Target Venue:** IEEE PerCom / ISWC / IoT Journals

---

## 1. Problem Statement

Human Activity Recognition (HAR) systems deployed on edge devices face **sensor drift** - gradual or sudden changes in data distribution caused by hardware degradation, recalibration, behavioral changes, or environmental shifts. Existing solutions detect drift as a binary signal but lack:

1. **Multiclass classification** of drift trajectory type
2. **Type-specific adaptation** strategies
3. **Resource-aware design** for low-end edge devices (<512MB RAM)

This system addresses these gaps with a two-stage cascaded architecture: binary detection (always-on) → multiclass classification (on-demand) → type-specific adaptation.

---

## 2. System Overview

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CENTRAL SERVER                               │
│  - FedAvg aggregation: w_global = Σ(n_i · w_i) / Σ(n_i)              │
│  - Sync interval: Every K local updates or T minutes                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↑↓ encrypted weights
┌─────────────────────────────────────────────────────────────────────┐
│                    EDGE DEVICE (×9 clients)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 1: Data Streaming → Phase 2: Binary Detection (Stage A)      │
│           ↓ (DriftFlag=True)                                        │
│  Phase 3: Drift Classification (Stage B) → 4-class output           │
│           ↓                                                         │
│  Phase 4: Type-Specific Adaptation                                  │
│           ↓ (when stable)                                           │
│  Phase 5: Federated Sync                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Target Hardware

| Component | Specification |
|-----------|---------------|
| **Device** | Raspberry Pi Zero 2 W (or equivalent) |
| **RAM** | 512MB (limits model + buffer sizes) |
| **CPU** | Quad-core 1GHz ARM Cortex-A53 |
| **Network** | WiFi 802.11n |
| **Power** | 5V 500mA (~2.5W max) |

### 2.3 Dataset

| Property | Value |
|----------|-------|
| **Name** | PAMAP2 Physical Activity Monitoring |
| **Subjects** | 9 (1 female, 8 males, age 27.22 ± 3.31) |
| **Sensors** | 3 IMUs (wrist, chest, ankle) @ 100Hz |
| **Activities** | 18 labeled classes (ID 1-24, excluding 0) |
| **Features** | 3D accelerometer, gyroscope, magnetometer + heart rate |

---

## 3. Phase 1: Data Streaming Pipeline

### 3.1 Data Partitioning

- **Strategy:** 1 subject = 1 edge client (9 clients total)
- **Rationale:** Natural non-IID distribution (different body types, movement patterns)
- **Files:** `Protocol/subject101.dat` through `subject109.dat`

### 3.2 Preprocessing

| Step | Operation | Justification |
|------|-----------|---------------|
| **Activity Filtering** | Drop rows where `activityID = 0` | Transient activities between labeled segments |
| **NaN Handling** | Linear interpolation for missing IMU values | Wireless sensor dropout is rare (<0.5%) |
| **Sensor Selection** | Use ±16g accelerometer only (columns 2-4, 21-23, 38-40) | ±6g saturates during high-impact activities |

### 3.3 Normalization

**Running Z-score per axis per sensor:**

```
μ_running(t) = α · μ_running(t-1) + (1-α) · x(t)    # α = 0.95
σ²_running(t) = α · σ²_running(t-1) + (1-α) · (x(t) - μ(t))²
z(t) = (x(t) - μ_running(t)) / √(σ²_running(t) + ε)
```

- **Why running?** Handles non-stationarity, no look-ahead bias
- **Per-subject:** Each client maintains independent normalization

### 3.4 Windowing

| Parameter | Value |
|-----------|-------|
| **Window Size** | 256 samples (2.56 seconds @ 100Hz) |
| **Stride** | 128 samples (50% overlap) |
| **Features per Window** | Mean, Std, Min, Max, 25th/75th percentile |
| **Feature Dimension** | 3 axes × 3 sensors × 5 stats = 45 features |

### 3.5 Drift Injection

**Injection Points:** Samples T = 5000, 15000, 25000 (multiple events per stream)

| Drift Type | Equation | Parameters |
|------------|----------|------------|
| **Sudden** | `x'(t) = x(t) + δ` for t ≥ T | δ ∈ {0.5g, 1.0g, 2.0g} |
| **Gradual** | `x'(t) = x(t) + α(t)·δ` where `α(t) = min(1, (t-T)/N)` | N = 5000 samples |
| **Recurring** | `x'(t) = x(t) + δ` if `(t-T) mod P < D` | P = 10000, D = 2000 |
| **Feature-Level** | `x'_axis = x_axis + δ` for wrist-X only | δ ∈ {0.5g, 1.0g} |

---

## 4. Phase 2: Binary Drift Detection (Stage A)

### 4.1 Architecture

```
Window Features (45) → MLP (45→32→18) → Activity Prediction (18 classes)
                                                    ↓
                                    Binary Error: error_t = 1[y_pred ≠ y_true]
                                                    ↓
                                    ADWIN Monitor → DriftFlag (boolean)
```

### 4.2 Model Specification

| Property | Value |
|----------|-------|
| **Architecture** | MLP: 45 → 32 (ReLU) → 18 (Softmax) |
| **Parameters** | (45×32 + 32) + (32×18 + 18) = 2,066 weights |
| **Size** | ~8KB (float32) |
| **Optimizer** | SGD with momentum (μ=0.9) |
| **Learning Rate** | 0.01 (initial), 0.001 (fine-tuning) |
| **Loss** | Cross-entropy |

### 4.3 ADWIN Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| **Delta** | 0.002 | Controls false positive rate (river default) |
| **Max Buffer** | 1000 error samples | Memory bound |
| **Min Window** | 30 samples | Avoid premature detection |

**ADWIN Logic:**
- Maintains adaptive window of recent errors
- Detects distribution change when subwindow means differ significantly
- Shrinks window on detection (forgets old data)
- Outputs: `drift_detected: bool`

---

## 5. Phase 3: Drift Classification (Stage B)

### 5.1 Trigger Mechanism

- **Activation:** Only when `DriftFlag = True` from Stage A
- **Dormant State:** ~0% CPU, ~10KB reserved memory
- **Active State:** ~5ms inference, ~10KB active memory

### 5.2 Input Features (10 total)

| Feature | Computation | Discriminative Power |
|---------|-------------|---------------------|
| **Error Slope** | Linear regression on last 100 errors | Steep → Sudden, shallow → Gradual |
| **Error Variance** | Std of last 100 errors | High → Sudden, low → Gradual |
| **Raw Variance Shift** | Δstd(raw data) from baseline | Large → Feature-Level |
| **Memory Match Score** | Cosine similarity to saved error patterns | High → Recurring |
| **Time Since Last Drift** | Samples elapsed since previous DriftFlag | Periodic → Recurring |
| **Error Kurtosis** | Peakiness of error distribution | Impulsive → Sudden |
| **Error Autocorrelation** | Lag-1 correlation | Temporal structure |
| **Baseline Deviation** | |error_rate - baseline_rate| | Magnitude indicator |
| **Sensor Asymmetry** | Var(wrist) vs Var(chest) vs Var(ankle) | Feature-Level indicator |
| **Recovery Rate** | d(error)/dt after initial spike | Gradual vs Sudden |

### 5.3 Classifier

**Rule-Based (Primary):**

```python
def classify_drift(features):
    if features.error_slope > 0.05 and features.error_variance > 0.3:
        return DriftType.SUDDEN
    elif 0.01 < features.error_slope <= 0.05:
        return DriftType.GRADUAL
    elif features.memory_match_score > 0.85:
        return DriftType.RECURRING
    elif features.raw_variance_shift > 2.0 and features.error_slope < 0.01:
        return DriftType.FEATURE_LEVEL
    else:
        return DriftType.SUDDEN  # Conservative default
```

**Decision Tree (Alternative):**

- `max_depth = 3` (interpretability)
- Train on labeled synthetic drift data
- Export rules for inspection

### 5.4 Output Classes

| Class | Description | Example Cause |
|-------|-------------|---------------|
| **Sudden** | Abrupt error increase | Sensor replacement, injury onset |
| **Gradual** | Slow error ramp | Sensor degradation, fitness change |
| **Recurring** | Matches historical pattern | Intermittent behavior, seasonal |
| **Feature-Level** | Single-axis variance spike | Axis-specific calibration drift |

---

## 6. Phase 4: Type-Specific Adaptation

### 6.1 Adaptation Strategies

| Drift Type | Strategy | Parameters | Memory |
|------------|----------|------------|--------|
| **Sudden** | Mini-batch SGD on recent 100 windows | LR=0.1, epochs=10, batch=32 | ~40KB |
| **Gradual** | Weighted sliding window loss | α=0.95 decay | ~40KB |
| **Recurring** | Load saved weights from concept memory | Match pattern ID | ~40KB + history |
| **Feature-Level** | Recompute Z-score μ/σ on recent 50 windows | Warmstart=50 | ~10KB |

### 6.2 Concept Memory (Recurring Drift)

| Property | Value |
|----------|-------|
| **Storage** | `(pattern_id, model_weights, error_signature)` |
| **Max Capacity** | 10 patterns × 8KB = 80KB |
| **Eviction** | LRU when full |
| **Match Threshold** | Cosine similarity > 0.85 |

### 6.3 Stability Criterion

Adaptation complete when:
- `accuracy_recent_50 >= 0.95 × baseline_accuracy`
- OR max 20 adaptation iterations reached

---

## 7. Phase 5: Federated Communication

### 7.1 Sync Protocol

| Property | Value |
|----------|-------|
| **Trigger** | Every 5 adaptation cycles OR 10 minutes |
| **Payload** | Model weights (~8KB) |
| **Encryption** | TLS 1.3 (simulated: shared dict) |
| **Aggregation** | FedAvg: `w_global = Σ(n_i · w_i) / Σ(n_i)` |
| **Server** | Simulated coordinator (single process) |

### 7.2 Weight Serialization

```python
# Extract
state_dict = model.state_dict()
flat_weights = torch.cat([p.flatten() for p in state_dict.values()])

# Load
# Reverse operation during global model download
```

---

## 8. Phase 6: Evaluation Metrics

### 8.1 Primary Metrics

| Metric | Measurement | Target |
|--------|-------------|-------|
| **Activity F1-Score** | Macro F1 across 18 activities | >85% post-adaptation |
| **Detection Latency** | Samples: drift injection → DriftFlag | <500 samples |
| **Classification Accuracy** | 4-class drift type accuracy | >75% |
| **Recovery Time** | Samples: DriftFlag → 95% baseline accuracy | <1000 samples |

### 8.2 Resource Metrics

| Metric | Measurement | Target |
|--------|-------------|-------|
| **Peak RAM (Stage A)** | `psutil` RSS during inference | <50MB |
| **Peak RAM (Stage B)** | `psutil` RSS during classification | <100MB |
| **Inference Latency** | `time.perf_counter()` per window | <10ms |
| **CPU Utilization** | `psutil.cpu_percent()` | <30% average |
| **Energy Proxy** | CPU time × 2.5W | Report mAh/hour |

### 8.3 Baseline Comparisons

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **B1: Global Model** | Centralized training, no FL | Upper-bound accuracy |
| **B2: Local No-Adapt** | Local model, no drift handling | Quantify drift impact |
| **B3: FL No-Adapt** | FedAvg, no drift detection | FL benefit without adaptation |
| **B4: ADWIN + Generic** | Binary detection, same adapt for all | Value of multiclass classification |
| **B5: Periodic Retraining** | Retrain every N windows | Industry heuristic baseline |

---

## 9. Implementation Details

### 9.1 Technology Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.9+ |
| **ML Framework** | PyTorch 2.0 |
| **Stream Mining** | `river` library (ADWIN) |
| **Preprocessing** | NumPy, Pandas |
| **Profiling** | `psutil`, `time.perf_counter()` |
| **Config** | YAML (PyYAML) |

### 9.2 File Structure

```
fl-drift-try_2/
├── config/
│   ├── drift_config.yaml
│   ├── model_config.yaml
│   └── fl_config.yaml
├── src/
│   ├── data_pipeline/
│   │   ├── preprocessing.py
│   │   ├── windowing.py
│   │   └── drift_injection.py
│   ├── drift_detection/
│   │   ├── binary_detector.py
│   │   └── drift_classifier.py
│   ├── adaptation/
│   │   ├── sudden_adapter.py
│   │   ├── gradual_adapter.py
│   │   ├── recurring_adapter.py
│   │   └── feature_adapter.py
│   ├── federated/
│   │   ├── client.py
│   │   └── server.py
│   ├── models/
│   │   ├── activity_mlp.py
│   │   └── concept_memory.py
│   └── evaluation/
│       ├── metrics.py
│       └── baselines.py
├── experiments/
│   └── [experiment outputs]
├── results/
│   └── [csv, plots]
└── tests/
    ├── test_preprocessing.py
    ├── test_drift_detection.py
    └── test_adaptation.py
```

---

## 10. Assumptions & Limitations

### 10.1 Assumptions

1. PAMAP2 dataset is representative of edge HAR deployment scenarios
2. Simulated FL server approximates real networked behavior
3. RPi Zero 2 W is defensible as "low-end edge" for research purposes
4. Running Z-score handles inter-subject normalization adequately

### 10.2 Limitations

1. **9 subjects** - Small sample size, note as limitation
2. **Simulated FL** - No network latency/faults modeled
3. **Single dataset** - PAMAP2 only, no cross-dataset validation
4. **Energy proxy** - CPU-time, not direct power measurement

---

## 11. Success Criteria

System is considered successful if:

1. ✅ Multiclass classification accuracy >75% (4 drift types)
2. ✅ Detection latency <500 samples (5 seconds @ 100Hz)
3. ✅ Recovery time <1000 samples post-drift
4. ✅ Stage A memory footprint <50MB on RPi
5. ✅ Activity F1-score >85% after adaptation
6. ✅ Type-specific adaptation outperforms generic (p < 0.05)

---

## 12. Future Work

1. **Real FL Deployment** - Networked clients, actual communication costs
2. **Lower-End Hardware** - ESP32/Arduino (requires quantization)
3. **Cross-Dataset Validation** - OPPORTUNITY, UniMiB-SHAR datasets
4. **Online Hyperparameter Tuning** - Adaptive ADWIN delta, LR scheduling
5. **Concept Drift + Concept Evolution** - New activity classes over time

---

## Appendix A: Configuration Defaults

### drift_config.yaml
```yaml
injection:
  points: [5000, 15000, 25000]
  magnitudes: [0.5, 1.0, 2.0]
  types: [sudden, gradual, recurring, feature_level]

gradual:
  ramp_samples: 5000

recurring:
  period: 10000
  duration: 2000

feature_level:
  target_axis: wrist_x
```

### model_config.yaml
```yaml
mlp:
  input_dim: 45
  hidden_dim: 32
  output_dim: 18
  learning_rate: 0.01
  momentum: 0.9

adwin:
  delta: 0.002
  max_buffer: 1000
  min_window: 30
```

### fl_config.yaml
```yaml
federated:
  sync_interval_adaptations: 5
  sync_interval_minutes: 10
  encryption: simulated
  aggregation: fedavg
```
