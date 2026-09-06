import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    GeometryAdaptiveRotationMarginalizedHeadingHead,
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior_PRMC_GeometryAdaptive,
    ParallelRotationMarginalizedHeadingHead,
)


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class GeometryAdaptivePRMCTest(unittest.TestCase):
    def _build_model(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_GeometryAdaptive'
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return MODEL_CLASS_DICT[name](**MODEL_KEYWARDS_DICT[name])

    @staticmethod
    def _grad_sum(module):
        return sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    def test_identity_initialization_matches_rigid_prmc(self):
        torch.manual_seed(7)
        rigid = ParallelRotationMarginalizedHeadingHead()
        adaptive = GeometryAdaptiveRotationMarginalizedHeadingHead()
        adaptive.rst_projector.load_state_dict(rigid.rst_projector.state_dict())
        adaptive.uav_projector.load_state_dict(rigid.uav_projector.state_dict())
        rst = torch.randn(2, 4, 256, 16, 16)
        uav = torch.randn(2, 256, 16, 16)
        rigid_heading, rigid_aux = rigid(rst, uav)
        adaptive_heading, adaptive_aux = adaptive(rst, uav)
        self.assertTrue(torch.allclose(rigid_heading, adaptive_heading, atol=1e-6))
        self.assertTrue(
            torch.allclose(rigid_aux['response'], adaptive_aux['response'], atol=1e-6)
        )
        identity = torch.eye(2).expand(2, -1, -1)
        self.assertTrue(
            torch.allclose(adaptive_aux['deformation_matrix'], identity, atol=1e-7)
        )

    def test_stretch_is_symmetric_positive(self):
        head = GeometryAdaptiveRotationMarginalizedHeadingHead()
        rst = torch.randn(3, 4, 256, 16, 16)
        uav = torch.randn(3, 256, 16, 16)
        _, auxiliary = head(rst, uav)
        stretch = auxiliary['deformation_matrix']
        self.assertTrue(torch.allclose(stretch, stretch.transpose(-1, -2), atol=1e-6))
        self.assertTrue((torch.linalg.eigvalsh(stretch) > 0).all())
        self.assertEqual(tuple(auxiliary['deformation_parameters'].shape), (3, 3))

    def test_registered_model_forward_backward(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_GeometryAdaptive'
        self.assertIs(
            MODEL_CLASS_DICT[name],
            PARCASGM_v5a_GlobalRST_PosPrior_PRMC_GeometryAdaptive,
        )
        model = self._build_model()
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['response'].shape), (2, 36, 9, 9))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())

        target = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
        losses = model.compute_heading_losses(auxiliary, target)
        self.assertIn('deformation', losses)
        losses['total'].backward()
        self.assertGreater(self._grad_sum(model.prmc_head.deformation_predictor), 0.0)


if __name__ == '__main__':
    unittest.main()
