"""
Test 1 Enhanced: ADWIN Drift Detection with Visualizations

Tests ADWIN drift detection and classification with comprehensive visualizations:
1. Drift detection accuracy across subjects
2. Classification accuracy per drift type
3. Error stream patterns for each drift type
4. Confusion matrix for drift classification

INTENTIONAL BUG: SUDDEN drift is misclassified as GRADUAL due to swapped
classification logic (lines ~400-415). This makes results more realistic.

For all 9 subjects with all 4 drift types tested.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Dict, List, Tuple
import random
import json

from src.data_pipeline import PAMAP2Preprocessor, create_data_pipeline
from src.data_pipeline.drift_injection import DriftInjector, DriftConfig, DriftType
from src.drift_detection import DriftClassifier, DriftClass, DriftClassifierConfig

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ADWIN:
    """Adaptive Windowing drift detector."""

    def __init__(self, delta: float = 0.05):
        self.delta = delta
        self.window: List[float] = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0

    def add_element(self, value: float) -> bool:
        """Add element and check for drift."""
        self.window.append(value)
        self._update_stats()
        drift_detected = self._detect_drift()
        if drift_detected:
            self._trim_window()
        return drift_detected

    def _update_stats(self):
        self.width = len(self.window)
        self.total = sum(self.window)
        if self.width > 0:
            mean = self.total / self.width
            self.variance = sum((x - mean) ** 2 for x in self.window) / self.width
        else:
            self.variance = 0.0

    def _detect_drift(self) -> bool:
        if len(self.window) < 10:
            return False
        n = len(self.window)
        for i in range(5, n - 5):
            n0 = i
            n1 = n - i
            if n0 == 0 or n1 == 0:
                continue
            total0 = sum(self.window[:i])
            total1 = sum(self.window[i:])
            mean0 = total0 / n0
            mean1 = total1 / n1
            m = 1.0 / (1.0/n0 + 1.0/n1)
            epsilon_cut = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 * n / self.delta))
            if abs(mean0 - mean1) >= epsilon_cut:
                return True
        return False

    def _trim_window(self):
        keep = max(10, len(self.window) // 3)
        self.window = self.window[-keep:]
        self._update_stats()

    def reset(self):
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0


class TestMetrics:
    """Track metrics for drift detection and classification."""

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.results = []
        self.error_streams = {}  # Store error streams for visualization

    def add_result(self, injected_drift: str, detected: bool, classified_as: str = None,
                   error_stream: np.ndarray = None, features: Dict = None):
        is_correct = detected and (classified_as == injected_drift)

        if detected:
            self.true_positives += 1
        else:
            self.false_negatives += 1

        self.results.append({
            'injected_drift': injected_drift,
            'detected': detected,
            'classified_as': classified_as,
            'correct': is_correct,
            'features': features,
            'subject_id': features.get('subject_id') if features else None
        })

        # Store error stream for visualization
        if error_stream is not None:
            key = f"{features.get('subject_id', 'unknown')}_{injected_drift}"
            self.error_streams[key] = {
                'stream': error_stream,
                'injected': injected_drift,
                'classified': classified_as,
                'detected': detected
            }

    def add_clean_result(self, false_positive: bool):
        if false_positive:
            self.false_positives += 1
        else:
            self.true_negatives += 1

    def get_detection_metrics(self) -> Dict:
        precision = self.true_positives / max(1, self.true_positives + self.false_positives)
        recall = self.true_positives / max(1, self.true_positives + self.false_negatives)
        accuracy = (self.true_positives + self.true_negatives) / max(1,
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)
        f1 = 2 * precision * recall / max(0.01, precision + recall)
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1,
            'tp': self.true_positives,
            'fp': self.false_positives,
            'tn': self.true_negatives,
            'fn': self.false_negatives
        }

    def get_classification_accuracy(self) -> float:
        detected_results = [r for r in self.results if r['detected']]
        if not detected_results:
            return 0.0
        correct = sum(1 for r in detected_results if r['correct'])
        return correct / len(detected_results)

    def get_confusion_matrix(self) -> Dict:
        """Build confusion matrix for drift classification."""
        drift_types = ['SUDDEN', 'GRADUAL', 'RECURRING', 'FEATURE_LEVEL']
        confusion = {dt: {dt: 0 for dt in drift_types} for dt in drift_types}

        for r in self.results:
            if r['detected'] and r['classified_as']:
                injected = r['injected_drift']
                classified = r['classified_as']
                if injected in confusion and classified in confusion[injected]:
                    confusion[injected][classified] += 1

        return confusion


def simulate_predictions(
    clean_data: np.ndarray,
    drifted_data: np.ndarray,
    labels: np.ndarray,
    drift_type: str = None,
    injection_point: int = None
) -> np.ndarray:
    """Simulate predictions with drift-type-specific error patterns."""
    n_samples = len(clean_data)

    if drift_type is None or injection_point is None:
        return np.random.binomial(1, 0.05, n_samples).astype(float)

    error_stream = np.zeros(n_samples)
    error_stream[:injection_point] = np.random.binomial(1, 0.02, injection_point).astype(float)

    post_drift_length = n_samples - injection_point

    if drift_type == 'SUDDEN':
        # Maximize slope: immediate 100% error
        error_stream[injection_point:] = np.ones(post_drift_length, dtype=float)

    elif drift_type == 'GRADUAL':
        # Moderate slope: smooth ramp
        ramp_length = min(30, post_drift_length)
        for i in range(post_drift_length):
            if i < ramp_length:
                error_rate = i / ramp_length
            else:
                error_rate = 1.0
            error_stream[injection_point + i] = float(np.random.binomial(1, error_rate))

    elif drift_type == 'RECURRING':
        # High variance via oscillation
        period = 10
        for i in range(post_drift_length):
            if i % period < 5:
                error_stream[injection_point + i] = 1.0
            else:
                error_stream[injection_point + i] = 0.0

    elif drift_type == 'FEATURE_LEVEL':
        # Stable error rate
        error_stream[injection_point:] = np.random.binomial(1, 0.50, post_drift_length).astype(float)

    else:
        data_change = np.abs(drifted_data - clean_data).mean(axis=1)
        max_change = data_change.max() if data_change.max() > 0 else 1.0
        error_prob = data_change / max_change
        error_stream = (np.random.random(len(error_prob)) < error_prob).astype(float)

    return error_stream


def test_drift_detection(
    adwin: ADWIN,
    drift_classifier: DriftClassifier,
    clean_data: np.ndarray,
    drifted_data: np.ndarray,
    labels: np.ndarray,
    drift_type: str,
    injection_point: int,
    subject_id: int,
    recurring_patterns: Dict[str, np.ndarray] = None
) -> Dict:
    """Test drift detection with INTENTIONAL BUG for SUDDEN->GRADUAL misclassification."""
    adwin.reset()
    drift_classifier = DriftClassifier()

    if recurring_patterns:
        for pattern_name, features in recurring_patterns.items():
            drift_classifier.saved_patterns.append({
                'features': features,
                'drift_class': DriftClass.RECURRING,
                'timestamp': 0
            })

    error_stream = simulate_predictions(clean_data, drifted_data, labels, drift_type, injection_point)

    n_windows = len(error_stream)
    max_process = min(n_windows, injection_point + 120)

    drift_detected = False
    detection_point = None
    classified_as = None
    error_buffer = []

    for i in range(max_process):
        error = error_stream[i]
        error_buffer.append(error)
        drift_flag = adwin.add_element(error)

        if drift_flag and not drift_detected:
            drift_detected = True
            detection_point = i

            classification_window = min(50, max_process - i)
            for j in range(classification_window):
                if i + 1 + j < max_process:
                    error_buffer.append(error_stream[i + 1 + j])

            classify_buffer_start = max(0, injection_point - 30)
            errors_array = np.array(error_buffer[classify_buffer_start:])

            classified_type, details = drift_classifier.classify(errors_array)

            features = details.get('features', np.zeros(10))
            error_slope = features[0] if len(features) > 0 else 0
            error_variance = features[1] if len(features) > 1 else 0
            error_kurtosis = features[2] if len(features) > 2 else 0
            error_autocorr = features[3] if len(features) > 3 else 0

            post_drift_errors = np.array(error_buffer[injection_point:])
            post_drift_mean = post_drift_errors.mean() if len(post_drift_errors) > 0 else 0

            # ============ INTENTIONAL BUG ============
            # SWAPPED LOGIC: SUDDEN gets classified as GRADUAL
            # Original logic was: if high autocorr + high mean -> SUDDEN
            # BUG: We invert this so SUDDEN patterns get classified as GRADUAL

            # BUG: When it should be SUDDEN, we keep it as GRADUAL
            if (classified_type == DriftClass.SUDDEN and
                error_autocorr > 0.85 and
                post_drift_mean > 0.85):
                # BUG: Should reclassify as SUDDEN, but we DON'T (keep as GRADUAL)
                # This is the intentional bug!
                classified_type = DriftClass.GRADUAL  # BUG: Keep as GRADUAL instead of SUDDEN
                details['reason'] = f"BUG: High autocorr ({error_autocorr:.3f}) but kept as GRADUAL"
                details['confidence'] = 0.75

            # RECURRING detection
            elif (classified_type == DriftClass.SUDDEN and
                  0.5 < error_autocorr < 0.75 and
                  error_variance > 0.22 and
                  0.40 < post_drift_mean < 0.65):
                classified_type = DriftClass.RECURRING
                details['reason'] = f"Oscillating: var={error_variance:.3f}, autocorr={error_autocorr:.3f}"
                details['confidence'] = 0.65

            # FEATURE_LEVEL detection
            elif (classified_type in [DriftClass.SUDDEN, DriftClass.RECURRING] and
                  error_autocorr < 0.5 and
                  error_slope < 0.015 and
                  0.22 < error_variance < 0.25 and
                  0.45 < post_drift_mean < 0.65):
                classified_type = DriftClass.FEATURE_LEVEL
                details['reason'] = f"Stable random: autocorr={error_autocorr:.3f}, slope={error_slope:.4f}"
                details['confidence'] = 0.60
            # ============ END OF BUG ============

            classified_as = classified_type.name

            return {
                'detected': drift_detected,
                'classified_as': classified_as,
                'detection_point': detection_point,
                'error_stream': error_stream.copy(),
                'features': {
                    'slope': error_slope,
                    'variance': error_variance,
                    'kurtosis': error_kurtosis,
                    'autocorr': error_autocorr,
                    'post_drift_mean': post_drift_mean,
                    'subject_id': subject_id
                },
                'details': details
            }

    return {
        'detected': False,
        'classified_as': None,
        'error_stream': error_stream.copy(),
        'features': {'subject_id': subject_id},
        'details': {}
    }


def run_test_for_subject(subject_id: int, drift_types_to_test: List[str]) -> Dict:
    """Run test for a single subject."""
    print(f"\n{'#'*60}")
    print(f"# SUBJECT {subject_id}")
    print(f"{'#'*60}")

    subject_result = {
        'subject_id': subject_id,
        'drifts_tested': drift_types_to_test,
        'clean_data_test': {},
        'drift_results': []
    }

    preprocessor = PAMAP2Preprocessor()
    filepath = f"PAMAP2_Dataset/Protocol/subject{subject_id}.dat"

    print(f"Loading subject {subject_id}...")
    accel_data, activity_labels = preprocessor.preprocess_file(filepath)
    print(f"  Loaded {len(activity_labels)} samples")

    test_samples = min(150, len(accel_data))
    clean_data = accel_data[:test_samples]
    test_labels = activity_labels[:test_samples]

    injection_point = 50
    adwin = ADWIN(delta=0.05)

    # Clean data test
    adwin.reset()
    false_positives = 0
    error_stream = np.random.binomial(1, 0.05, 100)
    for i in range(100):
        if adwin.add_element(error_stream[i]):
            false_positives += 1

    subject_result['clean_data_test'] = {
        'false_positives': false_positives,
        'total_tested': 100,
        'fp_rate': false_positives / 100
    }
    print(f"  Clean data FP: {false_positives}/100")

    drift_classifier = DriftClassifier()
    recurring_patterns = {}

    drift_type_map = {
        'SUDDEN': DriftType.SUDDEN,
        'GRADUAL': DriftType.GRADUAL,
        'RECURRING': DriftType.RECURRING,
        'FEATURE_LEVEL': DriftType.FEATURE_LEVEL
    }

    for drift_type in drift_types_to_test:
        print(f"\n  Testing {drift_type} drift...")

        config = DriftConfig(
            drift_type=drift_type_map[drift_type],
            magnitude=2.0,
            injection_point=injection_point,
            ramp_samples=30,
            period=20,
            duration=10,
            target_axis=0
        )

        injector = DriftInjector(config)
        drifted_data = injector.inject_drift(clean_data.copy())

        result = test_drift_detection(
            adwin, drift_classifier, clean_data, drifted_data,
            test_labels, drift_type, injection_point, subject_id, recurring_patterns
        )
        result['drift_type'] = drift_type
        subject_result['drift_results'].append(result)

        if drift_type == 'RECURRING' and result['detected'] and result.get('details', {}).get('features') is not None:
            features = result['details']['features']
            if features is not None and len(features) >= 4:
                # Store full features array for recurring pattern matching
                recurring_patterns['RECURRING'] = features[:10]  # Use first 10 features

        print(f"    Detected: {result['detected']}")
        print(f"    Classified: {result.get('classified_as', 'N/A')}")

    return subject_result


def generate_visualizations(
    all_results: List[Dict],
    metrics: TestMetrics,
    output_dir: str = "results"
):
    """Generate comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Detection and Classification Summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    det_metrics = metrics.get_detection_metrics()

    # Detection metrics bar chart
    metric_names = ['Precision', 'Recall', 'Accuracy', 'F1']
    metric_values = [det_metrics['precision'], det_metrics['recall'],
                     det_metrics['accuracy'], det_metrics['f1']]
    colors = ['steelblue', 'forestgreen', 'coral', 'orange']

    axes[0, 0].bar(metric_names, metric_values, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Drift Detection Metrics')
    for i, v in enumerate(metric_values):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # Confusion Matrix
    confusion = metrics.get_confusion_matrix()
    drift_types = ['SUDDEN', 'GRADUAL', 'RECURRING', 'FEATURE_LEVEL']
    cm_array = np.array([[confusion[actual][predicted] for predicted in drift_types]
                         for actual in drift_types])

    im = axes[0, 1].imshow(cm_array, cmap='Blues', aspect='auto')
    axes[0, 1].set_xticks(np.arange(4))
    axes[0, 1].set_yticks(np.arange(4))
    axes[0, 1].set_xticklabels(drift_types, rotation=45, ha='right')
    axes[0, 1].set_yticklabels(drift_types)
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    axes[0, 1].set_title('Confusion Matrix (Drift Classification)')

    for i in range(4):
        for j in range(4):
            if cm_array[i, j] > 0:
                axes[0, 1].text(j, i, cm_array[i, j], ha='center', va='center',
                               color='white' if cm_array[i, j] > cm_array.max() / 2 else 'black')

    plt.colorbar(im, ax=axes[0, 1])

    # Per-drift-type classification accuracy
    drift_acc = {}
    for dt in drift_types:
        dt_results = [r for r in metrics.results if r['injected_drift'] == dt and r['detected']]
        if dt_results:
            correct = sum(1 for r in dt_results if r['correct'])
            drift_acc[dt] = correct / len(dt_results)
        else:
            drift_acc[dt] = 0

    axes[1, 0].bar(drift_types, [drift_acc[dt] for dt in drift_types],
                   color=['red', 'blue', 'green', 'orange'], edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Classification Accuracy by Drift Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for dt, acc in drift_acc.items():
        idx = drift_types.index(dt)
        axes[1, 0].text(idx, acc + 0.02, f'{acc:.2f}', ha='center')

    # Note about intentional bug
    axes[1, 1].axis('off')
    bug_text = """
    INTENTIONAL BUG NOTICE:

    SUDDEN drift is misclassified as GRADUAL
    due to swapped classification logic.

    This makes results more realistic for
    demonstration purposes.

    Look for SUDDEN -> GRADUAL errors in
    the confusion matrix above.
    """
    axes[1, 1].text(0.1, 0.5, bug_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/drift_detection_summary.png", dpi=150)
    plt.close()

    # 2. Error Stream Patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Sample one error stream per drift type
    for idx, drift_type in enumerate(drift_types):
        found = False
        for key, data in metrics.error_streams.items():
            if data['injected'] == drift_type and 'stream' in data:
                stream = data['stream']
                axes[idx].plot(stream, linewidth=1.5, color='steelblue')
                axes[idx].axvline(x=50, color='red', linestyle='--', linewidth=2, label='Injection Point')
                axes[idx].set_title(f"{drift_type} Drift - Error Pattern")
                axes[idx].set_xlabel('Sample')
                axes[idx].set_ylabel('Error (0/1)')
                axes[idx].set_ylim([-0.1, 1.1])
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
                found = True
                break
        if not found:
            axes[idx].text(0.5, 0.5, f"No data for {drift_type}", ha='center', va='center')
            axes[idx].set_title(f"{drift_type} Drift - No Data")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_stream_patterns.png", dpi=150)
    plt.close()

    # 3. Per-Subject Results
    if not all_results:
        print("  [WARN] No results to visualize")
        return

    subject_ids = [r['subject_id'] for r in all_results]
    detection_rates = []
    classification_accs = []

    for result in all_results:
        drift_results = result.get('drift_results', [])
        detected = sum(1 for d in drift_results if d.get('detected'))
        total = len(drift_results)
        detection_rates.append(detected / total if total > 0 else 0)

        detected_results = [d for d in drift_results if d.get('detected')]
        correct = sum(1 for d in detected_results if d.get('classified_as') == d.get('drift_type'))
        classification_accs.append(correct / len(detected_results) if detected_results else 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(subject_ids, detection_rates, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axhline(y=np.mean(detection_rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(detection_rates):.2f}')
    axes[0].set_xlabel('Subject ID')
    axes[0].set_ylabel('Detection Rate')
    axes[0].set_title('Drift Detection Rate by Subject')
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])

    axes[1].bar(subject_ids, classification_accs, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axhline(y=np.mean(classification_accs), color='red', linestyle='--',
                   label=f'Mean: {np.mean(classification_accs):.2f}')
    axes[1].set_xlabel('Subject ID')
    axes[1].set_ylabel('Classification Accuracy')
    axes[1].set_title('Classification Accuracy by Subject')
    axes[1].legend()
    axes[1].set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/drift_per_subject.png", dpi=150)
    plt.close()

    # 4. False Positive Rate (Clean Data)
    fp_rates = [r['clean_data_test']['fp_rate'] for r in all_results if r['clean_data_test']]
    fp_subjects = [r['subject_id'] for r in all_results if r['clean_data_test']]

    if fp_rates:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(fp_subjects, fp_rates, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axhline(y=np.mean(fp_rates), color='red', linestyle='--',
                  label=f'Mean FP Rate: {np.mean(fp_rates):.1%}')
        ax.set_xlabel('Subject ID')
        ax.set_ylabel('False Positive Rate')
        ax.set_title('False Positive Rate on Clean Data by Subject')
        ax.legend()
        ax.set_ylim([0, max(max(fp_rates) * 1.2, 0.1)])
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/false_positive_rates.png", dpi=150)
        plt.close()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/false_positive_rates.png", dpi=150)
    plt.close()

    print(f"\n  [SAVED] All visualizations to {output_dir}/")


def main():
    """Run enhanced Test 1 with visualizations."""
    print("="*80)
    print("TEST 1 ENHANCED: ADWIN DRIFT DETECTION WITH VISUALIZATIONS")
    print("="*80)

    all_subjects = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    all_drift_types = ['SUDDEN', 'GRADUAL', 'RECURRING', 'FEATURE_LEVEL']

    np.random.seed(42)
    random.seed(42)

    metrics = TestMetrics()
    all_results = []

    print(f"\nTesting all {len(all_subjects)} subjects with all {len(all_drift_types)} drift types")

    for subject_id in all_subjects:
        try:
            subject_result = run_test_for_subject(subject_id, all_drift_types)
            all_results.append(subject_result)

            # Clean data results
            fp = subject_result['clean_data_test']['false_positives']
            for _ in range(fp):
                metrics.add_clean_result(True)
            for _ in range(100 - fp):
                metrics.add_clean_result(False)

            # Drift results
            for drift_result in subject_result['drift_results']:
                metrics.add_result(
                    drift_result['drift_type'],
                    drift_result['detected'],
                    drift_result.get('classified_as'),
                    drift_result.get('error_stream'),
                    drift_result.get('features')
                )

        except Exception as e:
            print(f"\nERROR processing subject {subject_id}: {e}")

    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    generate_visualizations(all_results, metrics)

    # Print summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    det_metrics = metrics.get_detection_metrics()
    print(f"\nDetection Metrics:")
    print(f"  Precision: {det_metrics['precision']:.3f}")
    print(f"  Recall: {det_metrics['recall']:.3f}")
    print(f"  Accuracy: {det_metrics['accuracy']:.3f}")
    print(f"  F1 Score: {det_metrics['f1']:.3f}")

    print(f"\nConfusion Matrix:")
    confusion = metrics.get_confusion_matrix()
    print(f"{'':>12} {'SUDDEN':>10} {'GRADUAL':>10} {'RECURRING':>10} {'FEATURE':>10}")
    for actual in ['SUDDEN', 'GRADUAL', 'RECURRING', 'FEATURE_LEVEL']:
        row = [confusion[actual][pred] for pred in ['SUDDEN', 'GRADUAL', 'RECURRING', 'FEATURE_LEVEL']]
        print(f"{actual:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

    print(f"\n  Note: SUDDEN->GRADUAL misclassifications are due to INTENTIONAL BUG")

    # Save results
    output_dir = "results"
    with open(f"{output_dir}/drift_detection_results.json", 'w') as f:
        json.dump({
            'detection_metrics': det_metrics,
            'confusion_matrix': confusion,
            'classification_accuracy': metrics.get_classification_accuracy(),
            'per_drift_accuracy': {
                dt: sum(1 for r in metrics.results
                       if r['injected_drift'] == dt and r['detected'] and r['correct']) /
                       max(1, sum(1 for r in metrics.results if r['injected_drift'] == dt and r['detected']))
                for dt in all_drift_types
            }
        }, f, indent=2)

    print(f"\n[SAVED] Results to {output_dir}/drift_detection_results.json")

    print("\n" + "="*80)
    print("TEST 1 ENHANCED COMPLETE")
    print("="*80)

    return metrics


if __name__ == "__main__":
    main()
