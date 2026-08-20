import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100,
)


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class ExperimentFWithH3DJoint100Test(unittest.TestCase):
    def _build(self):
        kwargs = dict(
            MODEL_KEYWARDS_DICT[
                'PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100'
            ]
        )
        kwargs.update(
            heading_feature_dim=16,
            heading_spatial_size=4,
            heading_candidate_grid_size=3,
            heading_angle_bins=8,
        )
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100(**kwargs)

    def test_registration_schedule_and_training_policy(self):
        cls = PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100
        self.assertIs(
            MODEL_CLASS_DICT[
                'PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100'
            ],
            cls,
        )
        self.assertFalse(cls.requires_init_checkpoint)
        self.assertEqual(cls.default_optimizer, 'Adam')
        self.assertEqual(cls.default_learning_rate, 5e-5)
        model = self._build()
        self.assertFalse(model.freeze_base)
        self.assertAlmostEqual(model.heading_loss_scale(0), 0.1)
        self.assertAlmostEqual(model.heading_loss_scale(4), 0.5)
        self.assertAlmostEqual(model.heading_loss_scale(9), 1.0)
        self.assertAlmostEqual(model.heading_loss_scale(99), 1.0)
        self.assertTrue(any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith('rst_global_fusion.')
        ))
        self.assertTrue(any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith('gprv_head.')
        ))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.backbone.parameters()
        ))

    def test_joint_forward_and_both_branches_receive_gradients(self):
        torch.manual_seed(53)
        model = self._build().train()
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)
        target_position = torch.tensor([[0.2, -0.1], [-0.25, 0.3]])
        target_heading = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        position_loss = nn.SmoothL1Loss()(position, target_position)
        heading_loss = model.compute_heading_losses(
            auxiliary, target_heading, target_position=target_position
        )['total']
        loss = 0.8 * position_loss + 0.2 * heading_loss
        loss.backward()
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertTrue(torch.isfinite(loss))
        global_gradient = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith('rst_global_fusion.') and parameter.grad is not None
        )
        heading_gradient = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith('gprv_head.') and parameter.grad is not None
        )
        self.assertGreater(global_gradient, 0.0)
        self.assertGreater(heading_gradient, 0.0)


if __name__ == '__main__':
    unittest.main()
