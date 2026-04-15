import torch

from bsp_surgical.models.viz_decoder import FeatureDecoder


def test_decoder_produces_128x128_rgb_in_01():
    decoder = FeatureDecoder(feature_dim=768).eval()
    z = torch.randn(4, 768)

    img = decoder(z)

    assert img.shape == (4, 3, 128, 128)
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_decoder_is_small_enough():
    """~5M params budget — purely for viz, should not dominate."""
    decoder = FeatureDecoder(feature_dim=768)
    params = sum(p.numel() for p in decoder.parameters())
    assert 2_000_000 < params < 10_000_000, f"got {params}"
