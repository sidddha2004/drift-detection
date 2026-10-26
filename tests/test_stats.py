import numpy as np

from src.eval.stats import bootstrap_ci, holm, paired_wilcoxon


def test_bootstrap_ci_contains_mean_and_is_ordered():
    v = np.random.default_rng(0).normal(0.8, 0.02, 20)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi


def test_holm_is_monotone_and_not_smaller_than_raw():
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == 0.03 and adj["b"] == 0.04 and adj["c"] == 0.5
    assert all(adj[k] >= p for k, p in {"a": 0.01, "b": 0.02, "c": 0.5}.items())


def test_wilcoxon_identical_samples_not_significant():
    assert paired_wilcoxon([1, 2, 3], [1, 2, 3]) == 1.0


def test_wilcoxon_detects_consistent_shift():
    a = np.arange(10) + 5.0
    assert paired_wilcoxon(a, a - 1 + np.linspace(0, 0.1, 10)) < 0.01
