import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvphr.models.posaglreg.models import (
    ContentDirectionMatching,
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior,
    PARCASGM_v5a_GlobalRST_PosPrior_CDM,
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


class ExperimentFCDMTest(unittest.TestCase):
    @staticmethod
    def _build(cls):
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return cls(**MODEL_KEYWARDS_DICT[cls.__name__])

    def test_matching_shapes_and_gradients(self):
        module = ContentDirectionMatching()
        rst = torch.randn(2, 4, 256, 16, 16, requires_grad=True)
        uav = torch.randn(2, 256, 16, 16, requires_grad=True)
        prior = torch.full((2, 4), 0.25)
        residual, heading, auxiliary = module(rst, uav, prior)
        self.assertEqual(tuple(residual.shape), (2, 4, 1024))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['tile_relation'].shape), (2, 4, 16, 2))
        self.assertTrue(torch.isfinite(heading).all())
        self.assertTrue(torch.isfinite(residual).all())
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual)))
        (heading.square().mean() + residual.sum()).backward()
        self.assertGreater(module.content_projector[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.direction_projector[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.residual_scale.grad.abs().sum().item(), 0)

    def test_f_baseline_initially_preserved_and_training_loss(self):
        torch.manual_seed(8)
        baseline = self._build(PARCASGM_v5a_GlobalRST_PosPrior)
        model = self._build(PARCASGM_v5a_GlobalRST_PosPrior_CDM)
        missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(all(name.startswith('cdm.') for name in missing))
        self.assertEqual(unexpected, [])
        baseline.eval()
        model.eval()
        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.no_grad():
            baseline_pos, baseline_head = baseline(patches)
        position, heading, auxiliary = model(patches, return_aux=True)
        self.assertTrue(torch.allclose(position, baseline_pos, atol=1e-6))
        self.assertTrue(torch.allclose(heading, baseline_head, atol=1e-6))
        self.assertEqual(tuple(auxiliary['heading_aux'].shape), (2, 2))
        self.assertEqual(tuple(auxiliary['attention_weights'].shape), (2, 4))

        target_heading = F.normalize(torch.randn(2, 2), dim=-1)
        losses = model.compute_heading_losses(auxiliary, target_heading)
        expected = F.smooth_l1_loss(heading.float(), target_heading)
        expected += 0.1 * F.smooth_l1_loss(auxiliary['heading_aux'].float(), target_heading)
        self.assertTrue(torch.allclose(losses['total'], expected))
        (position.square().mean() + 0.2 * losses['total']).backward()
        self.assertGreater(model.cdm.residual_scale.grad.abs().sum().item(), 0)
        self.assertGreater(model.cdm.direction_projector[0].weight.grad.abs().sum().item(), 0)

    def test_registry_and_config_restore(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_CDM'
        self.assertIs(MODEL_CLASS_DICT[name], PARCASGM_v5a_GlobalRST_PosPrior_CDM)
        kwargs = MODEL_KEYWARDS_DICT[name]
        with tempfile.TemporaryDirectory() as directory:
            with open(f'{directory}/training_configure.json', 'w', encoding='utf-8') as stream:
                json.dump({'model_class': name, 'model_kwargs': kwargs}, stream)
            restored_class, restored_kwargs = load_config_and_model(directory)
        self.assertIs(restored_class, PARCASGM_v5a_GlobalRST_PosPrior_CDM)
        self.assertEqual(restored_kwargs, kwargs)


if __name__ == '__main__':
    unittest.main()
