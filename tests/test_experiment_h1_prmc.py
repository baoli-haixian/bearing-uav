import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_PRMC_H1,
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


class ExperimentH1PRMCTest(unittest.TestCase):
    def _build_model(self, detach_shared=False):
        kwargs = dict(MODEL_KEYWARDS_DICT['PARCASGM_v5a_PRMC_H1'])
        kwargs.update(
            heading_feature_dim=16,
            heading_spatial_size=4,
            heading_num_rotations=8,
            heading_detach_shared=detach_shared,
        )
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return PARCASGM_v5a_PRMC_H1(**kwargs)

    def test_registration_preserves_legacy_h1(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_PRMC_H1'],
            PARCASGM_v5a_PRMC_H1,
        )
        self.assertIn('PARCASGM_v5a_H1', MODEL_CLASS_DICT)
        self.assertEqual(PARCASGM_v5a_PRMC_H1.default_loss_type, 'prmc')

    def test_head_shapes_mosaic_windows_and_gradients(self):
        torch.manual_seed(31)
        head = ParallelRotationMarginalizedHeadingHead(
            input_dim=32,
            feature_dim=16,
            spatial_size=4,
            num_rotations=8,
        )
        rst_maps = torch.randn(2, 4, 32, 8, 8, requires_grad=True)
        uav_map = torch.randn(2, 32, 8, 8, requires_grad=True)
        heading, auxiliary = head(rst_maps, uav_map)

        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['response'].shape), (2, 8, 5, 5))
        self.assertEqual(
            tuple(auxiliary['orientation_probability'].shape), (2, 8)
        )
        self.assertTrue(torch.isfinite(heading).all())
        self.assertTrue(torch.isfinite(auxiliary['response']).all())
        self.assertTrue(
            torch.allclose(
                auxiliary['orientation_probability'].sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )

        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        (heading - target).square().mean().backward()
        self.assertGreater(rst_maps.grad.abs().sum().item(), 0.0)
        self.assertGreater(uav_map.grad.abs().sum().item(), 0.0)
        parameter_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(parameter_gradient, 0.0)

    def test_default_head_builds_36_global_response_maps(self):
        torch.manual_seed(33)
        head = ParallelRotationMarginalizedHeadingHead(
            input_dim=32,
            feature_dim=16,
            spatial_size=8,
            num_rotations=36,
        )
        rst_maps = torch.randn(1, 4, 32, 12, 12)
        uav_map = torch.randn(1, 32, 12, 12)
        with torch.no_grad():
            _, auxiliary = head(rst_maps, uav_map)
        self.assertEqual(tuple(auxiliary['response'].shape), (1, 36, 9, 9))

    def test_normalized_correlation_finds_an_inserted_template(self):
        torch.manual_seed(35)
        head = ParallelRotationMarginalizedHeadingHead(
            input_dim=8,
            feature_dim=8,
            spatial_size=4,
            num_rotations=4,
        )
        templates = torch.randn(1, 4, 8, 4, 4)
        validity = torch.ones(1, 4, 1, 4, 4)
        mosaic = torch.zeros(1, 8, 8, 8)
        mosaic[:, :, 2:6, 3:7] = templates[:, 0]
        response = head._normalized_correlation(mosaic, templates, validity)
        peak_index = response[0, 0].flatten().argmax().item()
        self.assertEqual(divmod(peak_index, 5), (2, 3))
        self.assertTrue(torch.allclose(response[0, 0, 2, 3], torch.tensor(1.0)))

    def test_model_parallel_outputs_losses_and_no_official_heading_gradient(self):
        torch.manual_seed(37)
        model = self._build_model().train()
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)
        target_heading = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        losses = model.compute_heading_losses(auxiliary, target_heading)
        losses['total'].backward()

        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertTrue(torch.isfinite(losses['total']))
        self.assertGreater(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in model.prmc_head.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.dir_regressor.parameters())
        )
        shared_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.csmg.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(shared_gradient, 0.0)

    def test_optional_shared_gradient_detach(self):
        torch.manual_seed(41)
        model = self._build_model(detach_shared=True).train()
        patches = torch.randn(2, 5, 3, 64, 64)
        _, _, auxiliary = model(patches, return_aux=True)
        target_heading = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        model.compute_heading_losses(auxiliary, target_heading)['total'].backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.csmg.parameters())
        )
        self.assertGreater(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in model.prmc_head.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )


if __name__ == '__main__':
    unittest.main()
