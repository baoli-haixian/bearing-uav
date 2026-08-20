import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    MultiScaleRotationMarginalizedPositionHead,
    PARCASGM_v5a_GlobalRST_PosPrior,
    PARCASGM_v5a_MSRDCP_P5,
)
from cvphr.train.cvphr_train import load_initial_model_weights


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class ExperimentP5Test(unittest.TestCase):
    def test_dense_head_shape_loss_and_gradient(self):
        torch.manual_seed(41)
        head = MultiScaleRotationMarginalizedPositionHead(
            input_dim=32,
            feature_dim=16,
            num_rotations=4,
            output_size=12,
            scale_sizes=(8, 12),
            template_sizes=(3, 4),
        )
        rst_maps = torch.randn(2, 4, 32, 8, 8)
        uav_map = torch.randn(2, 32, 8, 8)
        base_position = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
        position, auxiliary = head(rst_maps, uav_map, base_position)
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['position_probability'].shape), (2, 12, 12))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.allclose(
            auxiliary['position_probability'].sum(dim=(-2, -1)),
            torch.ones(2), atol=1e-5,
        ))
        position.square().mean().backward()
        gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def _build(self, model_class):
        key = (
            'PARCASGM_v5a_MSRDCP_P5'
            if model_class is PARCASGM_v5a_MSRDCP_P5
            else 'PARCASGM_v5a_GlobalRST_PosPrior'
        )
        kwargs = dict(MODEL_KEYWARDS_DICT[key])
        if model_class is PARCASGM_v5a_MSRDCP_P5:
            kwargs.update(
                dense_feature_dim=16,
                dense_num_rotations=4,
                dense_output_size=12,
                dense_scale_sizes=(8, 12),
                dense_template_sizes=(3, 4),
            )
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return model_class(**kwargs)

    def test_f_checkpoint_and_stage_one_freezing(self):
        self.assertIs(MODEL_CLASS_DICT['PARCASGM_v5a_MSRDCP_P5'], PARCASGM_v5a_MSRDCP_P5)
        experiment_f = self._build(PARCASGM_v5a_GlobalRST_PosPrior).eval()
        model = self._build(PARCASGM_v5a_MSRDCP_P5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f'{directory}/experiment_f.pth'
            torch.save({'model_state_dict': experiment_f.state_dict()}, checkpoint_path)
            incompatible = load_initial_model_weights(model, checkpoint_path)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            key.startswith('dense_position_head.') for key in incompatible.missing_keys
        ))
        self.assertTrue(all(
            parameter.requires_grad == name.startswith('dense_position_head.')
            for name, parameter in model.named_parameters()
        ))

        patches = torch.randn(2, 5, 3, 64, 64)
        model.train()
        position, heading, auxiliary = model(patches, return_aux=True)
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        targets = torch.tensor([[0.1, -0.2], [-0.25, 0.3]])
        losses = model.compute_position_losses(auxiliary, targets)
        total = nn.SmoothL1Loss()(position, targets) + losses['total']
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith('dense_position_head.') and parameter.grad is not None
        ), 0.0)
        self.assertTrue(all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not name.startswith('dense_position_head.')
        ))


if __name__ == '__main__':
    unittest.main()
