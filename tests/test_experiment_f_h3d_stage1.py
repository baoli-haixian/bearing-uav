import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior,
    PARCASGM_v5a_GlobalRST_PosPrior_GPRVH,
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


class ExperimentFWithH3DStage1Test(unittest.TestCase):
    def _build(self, model_class):
        key = (
            'PARCASGM_v5a_GlobalRST_PosPrior_GPRVH'
            if model_class is PARCASGM_v5a_GlobalRST_PosPrior_GPRVH
            else 'PARCASGM_v5a_GlobalRST_PosPrior'
        )
        kwargs = dict(MODEL_KEYWARDS_DICT[key])
        if model_class is PARCASGM_v5a_GlobalRST_PosPrior_GPRVH:
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

    def test_registration(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_GlobalRST_PosPrior_GPRVH'],
            PARCASGM_v5a_GlobalRST_PosPrior_GPRVH,
        )
        self.assertEqual(
            PARCASGM_v5a_GlobalRST_PosPrior_GPRVH.default_loss_type, 'gprvh'
        )

    def test_f_checkpoint_preserves_position_and_only_head_trains(self):
        torch.manual_seed(29)
        experiment_f = self._build(PARCASGM_v5a_GlobalRST_PosPrior).eval()
        combined = self._build(PARCASGM_v5a_GlobalRST_PosPrior_GPRVH).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f'{directory}/experiment_f.pth'
            torch.save(
                {'model_state_dict': experiment_f.state_dict()}, checkpoint_path
            )
            incompatible = load_initial_model_weights(combined, checkpoint_path)

        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(key.startswith('gprv_head.') for key in incompatible.missing_keys)
        )
        self.assertTrue(
            all(
                parameter.requires_grad == name.startswith('gprv_head.')
                for name, parameter in combined.named_parameters()
            )
        )

        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.no_grad():
            f_position, _ = experiment_f(patches)
        combined.train()
        position, heading, auxiliary = combined(patches, return_aux=True)
        self.assertTrue(torch.equal(position, f_position))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['pose_volume_logits'].shape), (2, 9, 8))

        target_position = torch.tensor([[0.2, -0.1], [-0.25, 0.3]])
        target_heading = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        losses = combined.compute_heading_losses(
            auxiliary,
            target_heading,
            target_position=target_position,
        )
        losses['total'].backward()
        self.assertTrue(torch.isfinite(losses['total']))
        head_gradient = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in combined.named_parameters()
            if name.startswith('gprv_head.') and parameter.grad is not None
        )
        self.assertGreater(head_gradient, 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in combined.named_parameters()
                if not name.startswith('gprv_head.')
            )
        )


if __name__ == '__main__':
    unittest.main()
