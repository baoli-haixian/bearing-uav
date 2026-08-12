import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST,
    PARCASGM_v5a_GlobalRST_Aux,
)


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class ExperimentETest(unittest.TestCase):
    def _build_model(self):
        kwargs = MODEL_KEYWARDS_DICT['PARCASGM_v5a_GlobalRST_Aux']
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return PARCASGM_v5a_GlobalRST_Aux(**kwargs)

    def test_registration_and_default_inference_interface(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_GlobalRST_Aux'],
            PARCASGM_v5a_GlobalRST_Aux,
        )
        model = self._build_model()
        patches = torch.randn(1, 5, 3, 64, 64)

        inference_output = model(patches)
        training_output = model(patches, return_aux=True)

        self.assertEqual(len(inference_output), 2)
        self.assertEqual(len(training_output), 3)
        self.assertEqual(tuple(inference_output[0].shape), (1, 2))
        self.assertEqual(tuple(inference_output[1].shape), (1, 2))
        self.assertEqual(
            tuple(training_output[2]['attention_weights'].shape), (1, 4)
        )

    def test_auxiliary_loss_warmup_and_gradients(self):
        model = self._build_model()
        patches = torch.randn(2, 5, 3, 64, 64)
        coords = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])

        position, heading, auxiliary = model(patches, return_aux=True)
        losses_epoch_0 = model.compute_auxiliary_losses(auxiliary, coords, epoch=0)
        losses_epoch_4 = model.compute_auxiliary_losses(auxiliary, coords, epoch=4)
        pose_loss = position.square().mean() + heading.square().mean()
        (pose_loss + losses_epoch_0['total']).backward()

        self.assertAlmostEqual(losses_epoch_0['warmup_scale'], 0.2)
        self.assertAlmostEqual(losses_epoch_4['warmup_scale'], 1.0)
        self.assertTrue(torch.isfinite(losses_epoch_0['quad']))
        self.assertTrue(torch.isfinite(losses_epoch_0['attention']))
        fusion_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.rst_global_fusion.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(fusion_gradient, 0.0)

    def test_geometry_attention_loss_matches_corner_order(self):
        model = self._build_model()
        original = F.normalize(torch.randn(4, 4, 1024), dim=-1)
        attention = torch.eye(4)
        coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        )
        auxiliary = {
            'original_descriptors': original,
            'context_descriptors': original.clone(),
            'attention_weights': attention,
        }

        losses = model.compute_auxiliary_losses(auxiliary, coords, epoch=4)

        self.assertLess(losses['quad'].item(), 1e-6)
        self.assertLess(losses['attention'].item(), 1e-5)

    def test_experiment_b_interface_remains_unchanged(self):
        kwargs = MODEL_KEYWARDS_DICT['PARCASGM_v5a_GlobalRST']
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            model = PARCASGM_v5a_GlobalRST(**kwargs)

        output = model(torch.randn(1, 5, 3, 64, 64))
        self.assertEqual(len(output), 2)
        self.assertFalse(hasattr(model, 'uses_auxiliary_losses'))


if __name__ == '__main__':
    unittest.main()
