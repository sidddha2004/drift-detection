"""Closed-form input recalibration of a drifted sensor stream (virtual drift), used as the input adapter's fit.

Virtual drift (offset, gain, rotation of a sensor) is a transform of the INPUTS that does not depend on the activity. It can
therefore be estimated from the few labelled windows of the activities that happen to be in the recent buffer: for every
class k seen in the buffer, the mean raw sensor vector of the drifted windows should match the mean of the clean windows of
class k (client's own train pool). Per IMU we fit  out = m' A + t  (A 3x3, t 3) by ridge-regularised least squares towards
the identity, then express it in normalised coordinates as  x_corr = x W + b  (the AdaptedLSTM input layer).
Gradient training of the input layer was tried first (Phase 4 notes) and fails: the model is already right on the seen
classes, so the cross-entropy carries almost no signal about the transform.
"""
from typing import Dict, Tuple

import numpy as np


def class_means(x_raw: np.ndarray, y: np.ndarray, imu: int, min_n: int) -> Dict[int, Tuple[np.ndarray, int]]:
    cols = slice(3 * imu, 3 * imu + 3)
    out = {}
    for k in np.unique(y):
        sel = y == k
        if sel.sum() >= min_n:
            out[int(k)] = (x_raw[sel][..., cols].reshape(-1, 3).mean(0), int(sel.sum()))
    return out


def reference_means(x_train: np.ndarray, y_train: np.ndarray, mean: np.ndarray, std: np.ndarray, n_imu: int = 3):
    """Per IMU, per class mean raw vector of the clean train pool (computed once per client)."""
    raw = x_train.astype(np.float64) * std + mean
    return [class_means(raw, y_train, m, 1) for m in range(n_imu)]


def fit_input_alignment(x_buf: np.ndarray, y_buf: np.ndarray, ref: list, mean: np.ndarray, std: np.ndarray,
                        ridge: float = 1.0, min_n: int = 3, n_imu: int = 3, max_weight: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """x_buf (n, W, 9) NORMALISED drifted windows with labels y_buf. Returns (W_in (9, 9), b_in (9,)) such that
    x_buf @ W_in + b_in re-aligns the buffer with the clean class means. Identity when there is too little evidence."""
    C = n_imu * 3
    W, b = np.eye(C), np.zeros(C)
    raw = x_buf.astype(np.float64) * std + mean
    for m in range(n_imu):
        drift = class_means(raw, y_buf, m, min_n)
        ks = [k for k in drift if k in ref[m]]
        if len(ks) < 1:
            continue
        w = np.array([min(drift[k][1], max_weight) for k in ks], dtype=np.float64)
        Xd = np.stack([np.append(drift[k][0], 1.0) for k in ks])            # (K, 4) = [m', 1]
        M = np.stack([ref[m][k][0] for k in ks])                            # (K, 3) clean class means
        Wm = np.diag(w)
        G = Xd.T @ Wm @ Xd
        lam = ridge * np.trace(G) / 4.0 + 1e-9
        theta0 = np.vstack([np.eye(3), np.zeros((1, 3))])
        reg = lam * np.diag([1.0, 1.0, 1.0, 1e-3])                          # pull A to the identity, leave the bias free
        theta = np.linalg.solve(G + reg, Xd.T @ Wm @ M + reg @ theta0)
        A_T, t = theta[:3], theta[3]
        mu, sd = mean[3 * m:3 * m + 3], std[3 * m:3 * m + 3]
        W[3 * m:3 * m + 3, 3 * m:3 * m + 3] = (sd[:, None] * A_T) / sd[None, :]
        b[3 * m:3 * m + 3] = (mu @ A_T + t - mu) / sd
    return W, b
