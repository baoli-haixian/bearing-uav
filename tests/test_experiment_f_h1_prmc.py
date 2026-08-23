import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1,
    PARCASGM_v5a_PRMC_H1,
)


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class ExperimentFH1PRMCTest(unittest.TestCase):
    def _build_model(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1'
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return MODEL_CLASS_DICT[name](**MODEL_KEYWARDS_DICT[name])

    @staticmethod
    def _gradient_sum(module):
        return sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    def test_old_prmc_is_preserved_and_combined_model_is_registered(self):
        self.assertIs(MODEL_CLASS_DICT['PARCASGM_v5a_PRMC_H1'], PARCASGM_v5a_PRMC_H1)
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1'
        self.assertIs(MODEL_CLASS_DICT[name], PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1)
        self.assertEqual(PARCASGM_v5a_PRMC_H1.model_name, 'phr5_h1_prmc_parallel')
        self.assertEqual(
            PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1.model_name,
            'phr5_f_h1_prmc',
        )
        self.assertEqual(
            PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1.default_loss_type, 'prmc'
        )

    def test_forward_loss_and_branch_gradient_isolation(self):
        model = self._build_model()
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['orientation_probability'].shape), (2, 36))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())

        target_heading = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
        model.compute_heading_losses(auxiliary, target_heading)['total'].backward()
        self.assertGreater(self._gradient_sum(model.prmc_head), 0.0)
        self.assertEqual(self._gradient_sum(model.rst_global_fusion), 0.0)

        model.zero_grad(set_to_none=True)
        position, _ = model(patches)
        position.square().mean().backward()
        self.assertGreater(self._gradient_sum(model.rst_global_fusion), 0.0)
        self.assertEqual(self._gradient_sum(model.prmc_head), 0.0)


if __name__ == '__main__':
    unittest.main()
