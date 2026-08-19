import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    GeometryPoseVolumeHeadingHead,
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a,
    PARCASGM_v5a_GPRVH,
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


class ExperimentH3DTest(unittest.TestCase):
    def _build(self, model_class):
        key = (
            'PARCASGM_v5a_GPRVH'
            if model_class is PARCASGM_v5a_GPRVH
            else 'PARCASGM_v5a'
        )
        kwargs = dict(MODEL_KEYWARDS_DICT[key])
        if model_class is PARCASGM_v5a_GPRVH:
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
            return model_class(**kwargs)

    def test_registration_and_candidate_grid(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_GPRVH'], PARCASGM_v5a_GPRVH
        )
        head = GeometryPoseVolumeHeadingHead(
            input_dim=32,
            feature_dim=16,
            spatial_size=4,
            candidate_grid_size=3,
            candidate_radius=0.5,
            num_angle_bins=8,
        )
        self.assertEqual(tuple(head.candidate_offsets.shape), (9, 2))
        self.assertTrue(
            torch.allclose(head.candidate_offsets.min(dim=0).values,
                           torch.tensor([-0.5, -0.5]))
        )
        self.assertTrue(
            torch.allclose(head.candidate_offsets.max(dim=0).values,
                           torch.tensor([0.5, 0.5]))
        )

    def test_head_shapes_finite_and_detached_position(self):
        torch.manual_seed(13)
        head = GeometryPoseVolumeHeadingHead(
            input_dim=32,
            feature_dim=16,
            spatial_size=4,
            candidate_grid_size=3,
            num_angle_bins=8,
        )
        rst_maps = torch.randn(2, 4, 32, 8, 8)
        uav_map = torch.randn(2, 32, 8, 8)
        position = torch.zeros(2, 2, requires_grad=True)
        heading, auxiliary = head(rst_maps, uav_map, position)
        heading.square().mean().backward()
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['pose_volume_logits'].shape), (2, 9, 8))
        self.assertEqual(
            tuple(auxiliary['orientation_probability'].shape), (2, 8)
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

    def test_baseline_checkpoint_losses_and_freezing(self):
        torch.manual_seed(17)
        baseline = self._build(PARCASGM_v5a).eval()
        h3 = self._build(PARCASGM_v5a_GPRVH).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f'{directory}/baseline.pth'
            torch.save(
                {'model_state_dict': baseline.state_dict()}, checkpoint_path
            )
            incompatible = load_initial_model_weights(h3, checkpoint_path)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(key.startswith('gprv_head.') for key in incompatible.missing_keys)
        )

        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.no_grad():
            baseline_position, _ = baseline(patches)
        h3.train()
        position, heading, auxiliary = h3(patches, return_aux=True)
        target_position = torch.tensor([[0.1, -0.2], [-0.3, 0.25]])
        target_heading = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        losses = h3.compute_heading_losses(
            auxiliary,
            target_heading,
            target_position=target_position,
        )
        losses['total'].backward()

        self.assertTrue(torch.equal(position, baseline_position))
        self.assertTrue(torch.isfinite(heading).all())
        self.assertTrue(torch.isfinite(losses['total']))
        self.assertIn('opposite', losses)
        base_gradients = [
            parameter.grad
            for name, parameter in h3.named_parameters()
            if not name.startswith('gprv_head.')
        ]
        self.assertTrue(all(gradient is None for gradient in base_gradients))
        head_gradient = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in h3.named_parameters()
            if name.startswith('gprv_head.') and parameter.grad is not None
        )
        self.assertGreater(head_gradient, 0.0)


if __name__ == '__main__':
    unittest.main()
