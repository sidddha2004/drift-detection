import numpy as np

from src.data_pipeline import PAMAP2Preprocessor
from tests.conftest import PAMAP2_SUBJECT_101, requires_pamap2


@requires_pamap2
def test_subject101_loads_9_channels_without_nans_or_transient_label():
    x, y = PAMAP2Preprocessor().preprocess_file(str(PAMAP2_SUBJECT_101))
    assert x.shape[1] == 9 and len(x) == len(y)
    assert not np.isnan(x).any()
    assert 0 not in set(np.unique(y))
