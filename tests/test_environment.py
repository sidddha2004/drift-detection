import pytest
import torch

from src.drift_detection import ActivityLSTM, LSTMConfig


def test_lstm_parameter_count_is_53650():
    """Reported model size must match the code (the research-plan docx said ~77K)."""
    total, trainable = ActivityLSTM(LSTMConfig()).count_parameters()
    assert total == trainable == 53_650


def test_lstm_fp32_size_kb():
    total, _ = ActivityLSTM(LSTMConfig()).count_parameters()
    assert total * 4 / 1024 == pytest.approx(209.57, abs=0.1)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_lstm_forward_backward():
    m = torch.nn.LSTM(9, 64, 2, batch_first=True).cuda()
    out, _ = m(torch.randn(8, 256, 9, device="cuda"))
    out.sum().backward()
    assert out.shape == (8, 256, 64)
