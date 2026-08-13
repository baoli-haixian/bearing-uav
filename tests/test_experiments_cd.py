import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_Attn,
    PARCASGM_v5a_GlobalRST_Quad,
)


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class ExperimentsCDTest(unittest.TestCase):
    def _build_model(self, model_key):
        model_class = MODEL_CLASS_DICT[model_key]
        model_kwargs = MODEL_KEYWARDS_DICT[model_key]
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return model_class(**model_kwargs)

    def _forward_and_losses(self, model_key):
        model = self._build_model(model_key)
        patches = torch.randn(2, 5, 3, 64, 64)
        coords = torch.tensor([[-0.5, -0.25], [0.25, 0.75]])
        position, heading, auxiliary = model(patches, return_aux=True)
        losses = model.compute_auxiliary_losses(auxiliary, coords, epoch=4)
        return model, position, heading, losses

    def test_experiment_c_registration_and_loss_isolation(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_GlobalRST_Quad'],
            PARCASGM_v5a_GlobalRST_Quad,
        )
        model, position, heading, losses = self._forward_and_losses(
            'PARCASGM_v5a_GlobalRST_Quad'
        )

        self.assertEqual(model.model_name, 'phr5_globalrst_c')
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(model.attention_loss_weight, 0.0)
        self.assertEqual(losses['weighted_attention'].item(), 0.0)
        self.assertTrue(
            torch.allclose(losses['total'], losses['weighted_quad'])
        )

    def test_experiment_d_registration_and_loss_isolation(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_GlobalRST_Attn'],
            PARCASGM_v5a_GlobalRST_Attn,
        )
        model, position, heading, losses = self._forward_and_losses(
            'PARCASGM_v5a_GlobalRST_Attn'
        )

        self.assertEqual(model.model_name, 'phr5_globalrst_d')
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(model.quad_loss_weight, 0.0)
        self.assertEqual(losses['weighted_quad'].item(), 0.0)
        self.assertTrue(
            torch.allclose(losses['total'], losses['weighted_attention'])
        )

    def test_wrong_auxiliary_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            PARCASGM_v5a_GlobalRST_Quad(attention_loss_weight=0.1)
        with self.assertRaises(ValueError):
            PARCASGM_v5a_GlobalRST_Attn(quad_loss_weight=0.05)


if __name__ == '__main__':
    unittest.main()
