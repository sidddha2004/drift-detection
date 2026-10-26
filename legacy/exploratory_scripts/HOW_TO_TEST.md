# Testing Guide: Multiclass Drift Detection System

This guide explains how to test each component of the drift detection pipeline (Phase 1 and Phase 2 complete).

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install torch numpy pandas pyyaml river psutil matplotlib seaborn scikit-learn
```

### Step 2: Verify PAMAP2 Dataset Location

Ensure your PAMAP2 dataset is in the correct location:

```
D:\drift-detection\fl-drift-try_2\PAMAP2_Dataset\
    ├── Protocol\
    │   ├── subject101.dat
    │   ├── subject105.dat
    │   └── ...
    └── Optional\
        └── ...
```

If not present, download from: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

---

## Testing Phase 1: Data Streaming Pipeline

### Test Command

```bash
cd D:\drift-detection\fl-drift-try_2
python tests/test_phase1.py
```

### Expected Output

```
============================================================
PHASE 1: DATA STREAMING PIPELINE - TEST SUITE
============================================================
============================================================
TEST: Preprocessing Module
============================================================
Using data file: PAMAP2_Dataset/Protocol/subject101.dat

1. Loading data...
   Raw data shape: (n_samples, 54)
   Expected: (n_samples, 54)
   PASS

2. Filtering transient activities...
   Removed X transient samples
   PASS

3. Interpolating missing values...
   NaN before: X, after: 0
   PASS

4. Running full preprocessing pipeline...
   Accelerometer shape: (n_samples, 9)
   Activity labels shape: (n_samples,)
   PASS

5. Verifying normalization...
   Mean: 0.000xxx (expected ~0)
   Std: 1.000xxx (expected ~1)
   PASS

============================================================
ALL PREPROCESSING TESTS PASSED
============================================================
```

### What Each Test Validates

| Test | Validates |
|------|-----------|
| Preprocessing | Loading, filtering, interpolation, normalization |
| Windowing | 256-sample windows, 128-sample stride, 45 features |
| Drift Injection | All 4 drift types (Sudden, Gradual, Recurring, Feature-Level) |

### Troubleshooting

**Problem:** "PAMAP2 dataset not found"

**Solution:** Either:
1. Copy dataset to `PAMAP2_Dataset/` folder, OR
2. The tests will use synthetic data (still validates drift injection logic)

---

## Testing Phase 2: Binary Drift Detection (Stage A)

### Test Command

```bash
cd D:\drift-detection\fl-drift-try_2
python tests/test_phase2.py
```

### Expected Output

```
============================================================
PHASE 2: BINARY DRIFT DETECTION - TEST SUITE
============================================================
============================================================
TEST: MLP Model
============================================================

1. Creating MLP model...
   Model created successfully
   PASS

2. Testing forward pass...
   Input shape: torch.Size([16, 45])
   Output shape: torch.Size([16, 18])
   PASS

3. Testing prediction...
   Predictions shape: (16,)
   PASS

4. Testing weight extraction...
   Extracted 4 weight tensors
   PASS

5. Counting parameters...
   Total parameters: 2066
   Expected: 2066
   Size: ~8.1 KB (float32)
   PASS

============================================================
ALL MLP MODEL TESTS PASSED
============================================================
... (more tests for detector, classifier, edge client)
```

### What Each Test Validates

| Test | Validates |
|------|-----------|
| MLP Model | Architecture (45→32→18), weight extraction/loading |
| Binary Detector | ADWIN error tracking, drift flagging |
| Drift Classifier | Stage B classification logic |
| Edge Client | End-to-end stream processing |
| Real Data | Integration with PAMAP2 dataset |

---

## Manual Component Testing

### Test Preprocessing Alone

```python
from src.data_pipeline import PAMAP2Preprocessor

preprocessor = PAMAP2Preprocessor()
accel_data, labels = preprocessor.preprocess_file(
    "PAMAP2_Dataset/Protocol/subject101.dat"
)

print(f"Data shape: {accel_data.shape}")
print(f"Mean: {accel_data.mean():.6f}")
print(f"Std: {accel_data.std():.6f}")
```

### Test Windowing Alone

```python
from src.data_pipeline import create_data_pipeline
import numpy as np

# Generate synthetic data
data = np.random.randn(10000, 9)
labels = np.random.randint(1, 18, 10000)

pipeline = create_data_pipeline(window_size=256, stride=128)
features, window_labels = pipeline.process(data, labels)

print(f"Features: {features.shape}")  # (n_windows, 45)
print(f"Labels: {window_labels.shape}")
```

### Test Drift Injection Alone

```python
from src.data_pipeline import DriftInjector, DriftConfig, DriftType
import numpy as np

# Create data
data = np.random.randn(10000, 9)

# Configure sudden drift
config = DriftConfig(
    drift_type=DriftType.SUDDEN,
    magnitude=1.0,
    injection_point=5000
)

injector = DriftInjector(config)
drifted = injector.inject_drift(data)

print(f"Mean before: {data[:5000].mean():.3f}")
print(f"Mean after: {drifted[5000:].mean():.3f}")
print(f"Drift applied: {injector.drift_applied}")
```

### Test Binary Drift Detector Alone

```python
from src.drift_detection import BinaryDriftDetector
import numpy as np

# Create synthetic data
features = np.random.randn(1000, 45).astype(np.float32)
labels = np.random.randint(0, 18, 1000)

# Create and train detector
detector = BinaryDriftDetector()

# Initial training
for i in range(0, 200, 32):
    detector.train_batch(features[i:i+32], labels[i:i+32])

# Process stream
for i in range(200, 1000):
    pred = detector.predict(features[i])
    drift_flag = detector.update_and_check_drift(pred, labels[i])

    if drift_flag:
        print(f"Drift detected at sample {i}!")

print(f"Total drift events: {detector.drift_count}")
print(f"Error rate: {detector.get_error_rate():.3f}")
```

---

## Integration Test: Full Pipeline

Run the full Phase 1 → Phase 2 pipeline:

```python
from src.data_pipeline import PAMAP2Preprocessor, create_data_pipeline, DriftConfig, DriftType, DriftInjector
from src.drift_detection import BinaryDriftDetector, DriftClassifier, StageBDetector
import numpy as np

print("=== Full Pipeline Integration Test ===\n")

# 1. Load data
print("1. Loading PAMAP2 data...")
preprocessor = PAMAP2Preprocessor()
accel_data, activity_labels = preprocessor.preprocess_file(
    "PAMAP2_Dataset/Protocol/subject101.dat"
)
print(f"   Loaded {accel_data.shape[0]} samples")

# 2. Generate windows
print("2. Generating windows...")
pipeline = create_data_pipeline(window_size=256, stride=128)
features, labels = pipeline.process(accel_data, activity_labels)
print(f"   Generated {features.shape[0]} windows")

# 3. Inject drift
print("3. Injecting sudden drift...")
drift_config = DriftConfig(
    drift_type=DriftType.SUDDEN,
    magnitude=1.0,
    injection_point=500  # Window index
)
injector = DriftInjector(drift_config)

# Re-process with drift
accel_drifted = preprocessor.preprocess_file("PAMAP2_Dataset/Protocol/subject101.dat")[0]
accel_drifted = injector.inject_drift(accel_drifted)
features_drifted, _ = pipeline.process(accel_drifted, activity_labels)
print("   Drift injected at window 500")

# 4. Train detector
print("4. Training detector...")
detector = BinaryDriftDetector()
for epoch in range(3):
    for i in range(0, 200, 32):
        detector.train_batch(features[i:i+32], labels[i:i+32])
print("   Training complete")

# 5. Process drifted stream
print("5. Processing drifted stream...")
drift_flags = []
for i in range(len(features_drifted)):
    pred = detector.predict(features_drifted[i])
    drift_flag = detector.update_and_check_drift(pred, labels[i])
    drift_flags.append(drift_flag)

print(f"   Total drift flags: {sum(drift_flags)}")
print(f"   Drift events detected: {detector.drift_count}")

# 6. Classify drift (Stage B)
print("6. Classifying drift type (Stage B)...")
stage_b = StageBDetector()

# Add recent errors to buffer
errors = [1 if detector.predict(features_drifted[i]) != labels[i] else 0
          for i in range(max(0, len(features_drifted)-100), len(features_drifted))]

for err in errors:
    stage_b.add_error(err)

if stage_b.activate():
    from src.drift_detection import DriftClass
    drift_class, details = stage_b.classify_current_drift(timestamp=1000)
    print(f"   Drift class: {DriftClass(drift_class).name}")
    print(f"   Confidence: {details['confidence']:.3f}")
    print(f"   Reason: {details['reason']}")

print("\n=== Integration Test Complete ===")
```

---

## Performance Benchmarks

### Expected Performance (on RPi Zero 2 W class device)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Inference latency | <10ms | `time.perf_counter()` per window |
| Memory footprint (Stage A) | <50MB | `psutil.Process().memory_info().rss` |
| Detection latency | <500 samples | Samples from drift → first flag |
| Classification accuracy | >75% | On 4 drift types |

### Benchmark Script

```python
import time
import psutil
import os

process = psutil.Process(os.getpid())

# Measure inference time
start = time.perf_counter()
for i in range(100):
    detector.predict(features[i])
end = time.perf_counter()

print(f"Average inference time: {(end-start)/100*1000:.2f}ms")
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f}MB")
```

---

## Validation Checklist

Before proceeding to Phase 3-4 implementation, verify:

- [ ] **Phase 1 Tests Pass**
  - [ ] Preprocessing loads and normalizes data correctly
  - [ ] Windowing produces 256-sample windows with 45 features
  - [ ] All 4 drift types inject correctly

- [ ] **Phase 2 Tests Pass**
  - [ ] MLP model forward pass works
  - [ ] Weight extraction/loading works
  - [ ] ADWIN detects error rate changes
  - [ ] Drift classifier outputs 4 classes

- [ ] **Integration Works**
  - [ ] Full pipeline (load → window → drift → detect) runs end-to-end
  - [ ] Detection latency is measurable
  - [ ] Classification produces sensible output

- [ ] **Performance Acceptable**
  - [ ] Inference <10ms per window
  - [ ] Memory <100MB total

---

## Next Steps After Testing

Once Phase 1 and Phase 2 are validated:

1. **Phase 3 (Adaptation)** - Implement 4 type-specific adapters
2. **Phase 4 (Federated Learning)** - Implement weight sync + aggregation
3. **Phase 5 (Evaluation)** - Run full experiments with baselines

---

## Common Issues

### Issue: "No module named 'river'"

**Fix:** `pip install river`

### Issue: "CUDA not available"

**Fix:** This is expected - the system runs on CPU. Add `export CUDA_VISIBLE_DEVICES=""` or use CPU-only PyTorch.

### Issue: "Memory error on large files"

**Fix:** Reduce batch size in `model_config.yaml`:
```yaml
mlp:
  batch_size: 16  # Default is 32
```

### Issue: "No drift detected"

**Possible causes:**
1. Drift magnitude too low - increase to 2.0g
2. ADWIN delta too strict - reduce to 0.001
3. Not enough samples after drift - move injection point earlier

---

## Output Interpretation

### Drift Detection Output

```
Drift flags: [0, 0, 0, 1, 1, 0, 1, 0, ...]
              ↑
         First detection
```

- `0` = No drift detected
- `1` = Drift flag raised

### Drift Classification Output

```
Class: SUDDEN
Confidence: 0.85
Reason: Steep slope: 0.072, variance: 0.341
```

- **Sudden**: High slope + high variance
- **Gradual**: Moderate slope (0.01-0.05)
- **Recurring**: High memory match score
- **Feature-Level**: High raw variance shift, low error slope
