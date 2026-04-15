import torch

from bsp_surgical.models.perception import preprocess_rgb_batch


def test_preprocess_resizes_to_target_and_normalizes():
    # A pure 1.0 white image post-normalization should equal (1 - 0.485) / 0.229 for R
    x = torch.ones(2, 3, 128, 128)
    out = preprocess_rgb_batch(x, target_size=224)
    assert out.shape == (2, 3, 224, 224)
    expected_r = (1.0 - 0.485) / 0.229
    assert abs(out[:, 0].mean().item() - expected_r) < 1e-4


def test_preprocess_rejects_wrong_shape():
    try:
        preprocess_rgb_batch(torch.ones(3, 128, 128), target_size=224)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_preprocess_rejects_non_float():
    try:
        preprocess_rgb_batch(torch.zeros(1, 3, 128, 128, dtype=torch.uint8), target_size=224)
        raise AssertionError("should have raised")
    except ValueError:
        pass
