import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    ImplicitDirectionalRelationAdapter,
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior,
    PARCASGM_v5a_GlobalRST_PosPrior_IDRA,
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


class ExperimentH6DIDRATest(unittest.TestCase):
    @staticmethod
    def _gradient_sum(module):
        return sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    def _build(self, model_class):
        name = (
            'PARCASGM_v5a_GlobalRST_PosPrior_IDRA'
            if model_class is PARCASGM_v5a_GlobalRST_PosPrior_IDRA
            else 'PARCASGM_v5a_GlobalRST_PosPrior'
        )
        kwargs = dict(MODEL_KEYWARDS_DICT[name])
        if model_class is PARCASGM_v5a_GlobalRST_PosPrior_IDRA:
            kwargs.update(
                heading_relation_dim=32,
                heading_spatial_size=4,
                heading_num_queries=2,
                heading_num_heads=4,
                heading_dropout=0.0,
            )
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return model_class(**kwargs)

    def test_adapter_shapes_gate_and_rotation_convention(self):
        adapter = ImplicitDirectionalRelationAdapter(
            input_dim=16,
            descriptor_dim=32,
            relation_dim=32,
            spatial_size=4,
            num_queries=2,
            num_heads=4,
            dropout=0.0,
        ).eval()
        heading, auxiliary = adapter(
            torch.randn(2, 4, 16, 8, 8),
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 64),
            torch.randn(2, 2),
        )
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['heading_relation'].shape), (2, 32))
        self.assertEqual(tuple(auxiliary['heading_gate'].shape), (2, 1))
        self.assertTrue(torch.isfinite(auxiliary['rotation_consistency']))
        self.assertTrue(torch.allclose(heading.norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertGreater(float(auxiliary['heading_gate'].mean()), 0.0)
        self.assertLess(float(auxiliary['heading_gate'].mean()), 0.5)
        right = torch.tensor([[1.0, 0.0]])
        up = adapter.rotate_image_heading(right, 1)
        self.assertTrue(torch.equal(up, torch.tensor([[0.0, -1.0]])))

    def test_position_matches_experiment_f_and_gradients_are_separated(self):
        torch.manual_seed(7)
        baseline = self._build(PARCASGM_v5a_GlobalRST_PosPrior).eval()
        model = self._build(PARCASGM_v5a_GlobalRST_PosPrior_IDRA).eval()
        incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(
            all(key.startswith('idra_heading.') for key in incompatible.missing_keys)
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.no_grad():
            baseline_position, _ = baseline(patches)
            position, heading, auxiliary = model(patches, return_aux=True)
        torch.testing.assert_close(position, baseline_position, atol=1e-6, rtol=1e-5)
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(model.model_name, 'phr5_f_h6d_idra')
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())

        model.train()
        position, heading, auxiliary = model(patches, return_aux=True)
        target = nn.functional.normalize(torch.randn_like(heading), dim=-1)
        losses = model.compute_heading_losses(auxiliary, target)
        self.assertEqual(
            set(losses), {'distribution', 'correlation', 'final', 'total'}
        )
        losses['total'].backward()
        self.assertGreater(self._gradient_sum(model.idra_heading), 0.0)
        self.assertEqual(self._gradient_sum(model.rst_global_fusion), 0.0)

        model.zero_grad(set_to_none=True)
        position, _ = model(patches)
        position.square().mean().backward()
        self.assertGreater(self._gradient_sum(model.rst_global_fusion), 0.0)
        self.assertEqual(self._gradient_sum(model.idra_heading), 0.0)

    def test_registration_and_config_restore(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_IDRA'
        self.assertIs(
            MODEL_CLASS_DICT[name], PARCASGM_v5a_GlobalRST_PosPrior_IDRA
        )
        kwargs = MODEL_KEYWARDS_DICT[name]
        with tempfile.TemporaryDirectory() as config_dir:
            with open(
                f'{config_dir}/training_configure.json', 'w', encoding='utf-8'
            ) as config_file:
                json.dump({'model_class': name, 'model_kwargs': kwargs}, config_file)
            model_class, restored_kwargs = load_config_and_model(config_dir)
        self.assertIs(model_class, PARCASGM_v5a_GlobalRST_PosPrior_IDRA)
        self.assertEqual(restored_kwargs, kwargs)


if __name__ == '__main__':
    unittest.main()
