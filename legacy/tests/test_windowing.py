import numpy as np
import pytest

from src.data_pipeline import create_data_pipeline


def test_window_shapes_and_count(stream):
    x, y = stream
    w, wl = create_data_pipeline(256, 128, return_raw_windows=True).process(x, y)
    assert w.shape == ((6000 - 256) // 128 + 1, 256, 9)
    assert wl.shape == (w.shape[0],)


def test_windows_are_contiguous_slices(stream):
    x, y = stream
    w, _ = create_data_pipeline(256, 128, return_raw_windows=True).process(x, y)
    np.testing.assert_array_equal(w[3], x[3 * 128: 3 * 128 + 256])


def test_pure_windows_get_their_label(stream):
    x, y = stream
    _, wl = create_data_pipeline(256, 128, return_raw_windows=True).process(x, y)
    assert wl[0] == 1 and wl[-1] == 3


@pytest.mark.xfail(strict=True, reason="Audit A7: boundary windows keep a majority label "
                                       "instead of being dropped by a purity filter (Phase 1)")
def test_no_window_straddles_a_label_boundary(stream):
    x, y = stream
    w, wl = create_data_pipeline(256, 128, return_raw_windows=True).process(x, y)
    # windows starting at 1792 (=14*128) cover samples 1792..2047: still pure.
    # window 15 covers 1920..2175 and straddles the 2000 boundary.
    n_boundary = sum(1 for i in range(len(wl)) if len(set(y[i * 128: i * 128 + 256])) > 1)
    assert n_boundary == 0  # currently 2 such windows are emitted
