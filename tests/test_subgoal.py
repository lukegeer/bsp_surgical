import torch

from bsp_surgical.models.subgoal import SubgoalGenerator
from bsp_surgical.models.losses import subgoal_dual_supervision_loss


def test_subgoal_generator_shapes():
    mlp = SubgoalGenerator(latent_dim=128)
    z_now = torch.randn(4, 128)
    z_goal = torch.randn(4, 128)

    sg = mlp(z_now, z_goal)

    assert sg.shape == (4, 128)


def test_subgoal_generator_is_order_sensitive():
    """h(a, b) != h(b, a) — the two inputs have different roles."""
    torch.manual_seed(0)
    mlp = SubgoalGenerator(latent_dim=16).eval()
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)

    assert not torch.allclose(mlp(a, b), mlp(b, a))


def test_dual_supervision_loss_two_terms_are_nonneg_scalars():
    torch.manual_seed(0)
    mlp = SubgoalGenerator(latent_dim=16)
    z_start = torch.randn(4, 16)
    z_quarter = torch.randn(4, 16)
    z_mid = torch.randn(4, 16)
    z_end = torch.randn(4, 16)

    total, parts = subgoal_dual_supervision_loss(
        mlp, z_start=z_start, z_quarter=z_quarter, z_mid=z_mid, z_end=z_end
    )

    assert total.ndim == 0
    assert total.item() >= 0
    assert set(parts.keys()) == {"gt", "pred"}
    assert parts["gt"].item() >= 0 and parts["pred"].item() >= 0


def test_dual_supervision_uses_model_prediction_for_second_level():
    """The L_pred term must feed the model's own sg_1 prediction back in,
    not the ground-truth z_mid. Verify by substituting z_mid and seeing
    that the loss would be *different*."""
    torch.manual_seed(0)
    mlp = SubgoalGenerator(latent_dim=16)
    z_start = torch.randn(4, 16)
    z_quarter = torch.randn(4, 16)
    z_mid = torch.randn(4, 16)
    z_end = torch.randn(4, 16)

    total, parts = subgoal_dual_supervision_loss(
        mlp, z_start=z_start, z_quarter=z_quarter, z_mid=z_mid, z_end=z_end
    )

    # alternative: L_pred with GT midpoint as input — different value
    sg1_pred = mlp(z_start, z_end)
    sg2_via_pred = mlp(z_start, sg1_pred)
    sg2_via_gt = mlp(z_start, z_mid)
    assert not torch.allclose(sg2_via_pred, sg2_via_gt)
    # parts["pred"] must use sg2_via_pred, not sg2_via_gt
    expected_pred = ((sg2_via_pred - z_quarter) ** 2).mean()
    assert torch.allclose(parts["pred"], expected_pred, atol=1e-5)
