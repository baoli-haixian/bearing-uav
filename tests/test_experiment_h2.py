import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    CyclicOrientationCorrelation,
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    MSPCOCHeadingHead,
    PARCASGM_v5a,
    PARCASGM_v5a_MSPCOC,
    PositionConditionedPolarSampler,
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


class ExperimentH2Test(unittest.TestCase):
    def _build(self, model_class):
        key = (
            'PARCASGM_v5a_MSPCOC'
            if model_class is PARCASGM_v5a_MSPCOC
            else 'PARCASGM_v5a'
        )
        kwargs = dict(MODEL_KEYWARDS_DICT[key])
        if model_class is PARCASGM_v5a_MSPCOC:
            kwargs.update(
                heading_feature_dim=16,
                heading_radial_bins=4,
                heading_angle_bins=12,
                heading_crop_scales=(1.0,),
                heading_num_heads=4,
                heading_dropout=0.0,
            )
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return model_class(**kwargs)

    def test_registration_and_coordinate_convention(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_MSPCOC'],
            PARCASGM_v5a_MSPCOC,
        )
        positions = torch.tensor(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
        )
        centers = PositionConditionedPolarSampler.position_to_grid_center(
            positions
        )
        expected = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
        )
        self.assertTrue(torch.equal(centers, expected))

    def test_polar_sampler_shape_and_finite_gradients(self):
        sampler = PositionConditionedPolarSampler(
            num_radial_bins=4,
            num_angle_bins=12,
            crop_scales=(0.75, 1.0, 1.25),
        )
        mosaic = torch.randn(2, 8, 16, 16, requires_grad=True)
        positions = torch.tensor([[0.0, 0.0], [0.25, -0.5]])
        sampled = sampler(mosaic, positions)
        self.assertEqual(tuple(sampled.shape), (2, 3, 8, 4, 12))
        sampled.square().mean().backward()
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertGreater(mosaic.grad.abs().sum().item(), 0.0)

    def test_rst_centers_sample_the_expected_mosaic_quadrants(self):
        rst_maps = torch.stack(
            [torch.full((1, 1, 8, 8), float(index)) for index in range(1, 5)],
            dim=1,
        )
        mosaic = PositionConditionedPolarSampler.build_mosaic(rst_maps)
        sampler = PositionConditionedPolarSampler(
            num_radial_bins=1,
            num_angle_bins=4,
            crop_scales=(1.0,),
            base_radius=1e-4,
        )
        centers = torch.tensor(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
        )
        sampled = sampler(mosaic.expand(4, -1, -1, -1), centers)
        values = sampled.mean(dim=(1, 2, 3, 4))
        self.assertTrue(
            torch.allclose(values, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )

    def test_cyclic_correlation_recovers_known_shift(self):
        torch.manual_seed(5)
        angle_bins = 12
        correlation = CyclicOrientationCorrelation(
            num_angle_bins=angle_bins,
            orientation_temperature=0.1,
            scale_temperature=0.2,
        )
        rst = torch.randn(2, 8, 4, angle_bins)
        heading_shift = 3
        uav = torch.roll(rst, shifts=-heading_shift, dims=-1)
        rst = rst.unsqueeze(1)
        _, probability, _ = correlation(uav, rst)
        self.assertTrue(
            torch.equal(
                probability.argmax(dim=-1),
                torch.full((2,), heading_shift),
            )
        )

    def test_head_outputs_and_losses_are_finite(self):
        torch.manual_seed(7)
        head = MSPCOCHeadingHead(
            input_dim=32,
            feature_dim=16,
            num_radial_bins=4,
            num_angle_bins=12,
            crop_scales=(1.0,),
            num_heads=4,
            dropout=0.0,
        )
        rst_maps = torch.randn(2, 4, 32, 8, 8)
        uav_map = torch.randn(2, 32, 8, 8)
        position = torch.zeros(2, 2, requires_grad=True)
        base_heading = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        heading, auxiliary = head(
            rst_maps, uav_map, position, base_heading
        )
        heading.square().mean().backward()
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(
            tuple(auxiliary['orientation_probability'].shape), (2, 12)
        )
        self.assertTrue(torch.isfinite(heading).all())
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(heading, dim=-1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertIsNone(position.grad)
        gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)

    def test_baseline_checkpoint_forward_backward_and_freezing(self):
        baseline = self._build(PARCASGM_v5a).eval()
        h2 = self._build(PARCASGM_v5a_MSPCOC).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f'{directory}/baseline.pth'
            torch.save(
                {'model_state_dict': baseline.state_dict()}, checkpoint_path
            )
            incompatible = load_initial_model_weights(h2, checkpoint_path)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(key.startswith('ms_pcoc_head.') for key in incompatible.missing_keys)
        )

        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.no_grad():
            baseline_position, _ = baseline(patches)
        h2.train()
        position, heading, auxiliary = h2(patches, return_aux=True)
        losses = h2.compute_heading_losses(
            auxiliary, torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        )
        losses['total'].backward()

        self.assertTrue(torch.equal(position, baseline_position))
        self.assertTrue(torch.isfinite(heading).all())
        self.assertTrue(torch.isfinite(losses['total']))
        base_gradients = [
            parameter.grad
            for name, parameter in h2.named_parameters()
            if not name.startswith('ms_pcoc_head.')
        ]
        self.assertTrue(all(gradient is None for gradient in base_gradients))
        head_gradient = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in h2.named_parameters()
            if name.startswith('ms_pcoc_head.') and parameter.grad is not None
        )
        self.assertGreater(head_gradient, 0.0)


if __name__ == '__main__':
    unittest.main()
