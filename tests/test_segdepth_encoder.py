import numpy as np
import torch

from bsp_surgical.models.segdepth_encoder import _SegDepthCNN, seg_to_onehot, normalize_depth


def test_segdepth_cnn_output_shape():
    cnn = _SegDepthCNN(num_seg_channels=8, out_dim=256)
    seg = torch.rand(4, 8, 128, 128)
    depth = torch.rand(4, 1, 128, 128)
    out = cnn(seg, depth)
    assert out.shape == (4, 256)


def test_seg_to_onehot_picks_top_k_ids():
    seg = np.zeros((4, 4), dtype=np.int32)
    seg[0, 0] = 5   # 1 pixel
    seg[0, 1:] = 3  # 3 pixels
    seg[1:] = 1      # 12 pixels (most common)
    out = seg_to_onehot(seg, num_channels=3)
    assert out.shape == (3, 4, 4)
    # channel 0 = most common (id=1, 12 pixels)
    assert out[0].sum() == 12
    # channel 1 = id=3, 3 pixels
    assert out[1].sum() == 3
    # channel 2 = id=5, 1 pixel
    assert out[2].sum() == 1


def test_seg_to_onehot_decodes_body_id_from_upper_bits():
    """PyBullet encodes link_id in upper 24 bits, body_id in lower 24."""
    seg = np.array([[(2 << 24) | 1, (5 << 24) | 1], [1, (2 << 24) | 7]], dtype=np.int32)
    out = seg_to_onehot(seg, num_channels=2)
    # three pixels have body_id=1, one has body_id=7
    assert out.shape == (2, 2, 2)
    assert out[0].sum() == 3  # most common body_id
    assert out[1].sum() == 1


def test_normalize_depth_shape():
    d = np.random.rand(128, 128).astype(np.float32)
    out = normalize_depth(d)
    assert out.shape == (1, 128, 128)
