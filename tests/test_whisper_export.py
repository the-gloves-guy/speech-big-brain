import torch

from lid.whisper_export_backend import WhisperTinyLIDClassifier, WhisperTinyLIDEncoder
from lid.whisper_model import pool_real_tokens, whisper_head


def test_exported_classifier_preserves_calibrated_log_probabilities():
    torch.manual_seed(42)
    original = whisper_head().eval()
    temperature = 1.902119886678446
    state = {key: value.clone() for key, value in original.state_dict().items()}
    state["3.weight"] /= temperature
    state["3.bias"] /= temperature
    exported = WhisperTinyLIDClassifier().eval()
    exported.load_state_dict(state)
    inputs = torch.randn(3, 768)
    expected = (original(inputs) / temperature).log_softmax(dim=-1)
    actual = exported(inputs.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones(3))


def test_export_frontend_crops_and_respects_relative_lengths(monkeypatch):
    import lid.whisper_export_backend as module

    seen = []

    def mel(signal, n_mels):
        seen.append(signal.clone())
        return torch.zeros(80, 3000)

    monkeypatch.setattr(module, "log_mel_spectrogram", mel)
    features = (
        torch.arange(1500 * 4, dtype=torch.float32).reshape(1, 1500, 4).repeat(2, 1, 1)
    )

    class FakeEncoder(torch.nn.Module):
        def forward(self, mels):
            assert mels.shape == (2, 80, 3000)
            return features

    encoder = WhisperTinyLIDEncoder(n_state=4, n_head=1, n_layer=1)
    encoder.encoder = FakeEncoder()
    waveforms = torch.linspace(-0.8, 0.8, 180000).repeat(2, 1)
    result = encoder(waveforms, torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(
        seen[0][:160000], waveforms[0, 10000:170000], atol=0, rtol=0
    )
    torch.testing.assert_close(seen[1][:90000], waveforms[1, :90000], atol=0, rtol=0)
    assert seen[0].shape == (480000,) and torch.count_nonzero(seen[0][160000:]) == 0
    assert torch.count_nonzero(seen[1][90000:]) == 0
    expected = pool_real_tokens(features, torch.tensor([160000, 90000])).unsqueeze(1)
    torch.testing.assert_close(result, expected)
