import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    CyclicEquivariantPoseVolume,
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior_C8Pose,
    load_config_and_model,
)


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class CyclicPoseVolumeTest(unittest.TestCase):
    def test_orbit_pose_volume_shapes_and_probabilities(self):
        branch = CyclicEquivariantPoseVolume(
            feature_dim=8,
            spatial_size=4,
            num_rotations=8,
            residual_hidden_dim=8,
        )
        rst = torch.randn(2, 4, 3, 64, 64)
        uav = torch.randn(2, 3, 64, 64)
        position, heading, auxiliary = branch(rst, uav)

        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['pose_volume'].shape), (2, 8, 5, 5))
        self.assertEqual(
            tuple(auxiliary['position_probability'].shape), (2, 5, 5)
        )
        self.assertEqual(
            tuple(auxiliary['orientation_probability'].shape), (2, 8)
        )
        self.assertTrue(
            torch.allclose(
                auxiliary['position_probability'].sum(dim=(1, 2)),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                auxiliary['orientation_probability'].sum(dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(heading.norm(dim=-1), torch.ones(2), atol=1e-5)
        )

    def test_group_mosaic_layout(self):
        tiles = torch.zeros(1, 4, 1, 2, 2, 2)
        for index in range(4):
            tiles[:, index].fill_(float(index + 1))
        mosaic = CyclicEquivariantPoseVolume.build_group_mosaic(tiles)
        self.assertEqual(tuple(mosaic.shape), (1, 1, 2, 4, 4))
        self.assertTrue(torch.equal(mosaic[..., :2, :2], torch.ones(1, 1, 2, 2, 2)))
        self.assertTrue(torch.equal(mosaic[..., :2, 2:], 2 * torch.ones(1, 1, 2, 2, 2)))
        self.assertTrue(torch.equal(mosaic[..., 2:, :2], 3 * torch.ones(1, 1, 2, 2, 2)))
        self.assertTrue(torch.equal(mosaic[..., 2:, 2:], 4 * torch.ones(1, 1, 2, 2, 2)))


class ExperimentFC8PoseTest(unittest.TestCase):
    def _build_model(self):
        kwargs = dict(
            MODEL_KEYWARDS_DICT['PARCASGM_v5a_GlobalRST_PosPrior_C8Pose']
        )
        kwargs.update(
            c8_feature_dim=8,
            c8_spatial_size=4,
            c8_residual_hidden_dim=8,
        )
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return PARCASGM_v5a_GlobalRST_PosPrior_C8Pose(**kwargs)

    def test_registration_forward_loss_and_backward(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_C8Pose'
        self.assertIs(
            MODEL_CLASS_DICT[name], PARCASGM_v5a_GlobalRST_PosPrior_C8Pose
        )
        model = self._build_model()
        patches = torch.randn(1, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)

        self.assertEqual(model.model_name, 'phr5_f_c8pose')
        self.assertEqual(model.default_loss_type, 'c8pose')
        self.assertEqual(tuple(position.shape), (1, 2))
        self.assertEqual(tuple(heading.shape), (1, 2))
        self.assertEqual(tuple(auxiliary['pose_volume'].shape), (1, 8, 5, 5))
        self.assertGreater(float(auxiliary['c8_position_gate']), 0.0)
        self.assertLess(float(auxiliary['c8_position_gate']), 0.5)

        target_heading = nn.functional.normalize(
            torch.randn(1, 2), dim=-1
        )
        losses = model.compute_heading_losses(auxiliary, target_heading)
        total = position.square().mean() + losses['total']
        total.backward()
        branch_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.c8_pose_branch.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(branch_gradient, 0.0)
        self.assertTrue(torch.isfinite(total))

    def test_training_config_can_restore_model_class(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_C8Pose'
        kwargs = MODEL_KEYWARDS_DICT[name]
        with tempfile.TemporaryDirectory() as config_dir:
            with open(
                f'{config_dir}/training_configure.json',
                'w',
                encoding='utf-8',
            ) as config_file:
                json.dump(
                    {'model_class': name, 'model_kwargs': kwargs}, config_file
                )
            model_class, restored_kwargs = load_config_and_model(config_dir)

        self.assertIs(model_class, PARCASGM_v5a_GlobalRST_PosPrior_C8Pose)
        self.assertEqual(restored_kwargs, kwargs)


if __name__ == '__main__':
    unittest.main()
