import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA,
    SpatialGuidedCyclicDirectionCA,
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


class ExperimentFSGCDCATest(unittest.TestCase):
    def _build_model(self):
        kwargs = MODEL_KEYWARDS_DICT[
            'PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA'
        ]
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA(**kwargs)

    @staticmethod
    def _gradient_sum(module):
        return sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    def test_direction_ca_shapes_and_probabilities(self):
        module = SpatialGuidedCyclicDirectionCA(
            input_dim=32, descriptor_dim=64, direction_dim=16,
            num_directions=16,
        )
        heading_context, auxiliary = module(
            torch.randn(2, 4, 32, 16, 16),
            torch.randn(2, 32, 16, 16),
            torch.randn(2, 4),
            torch.randn(2, 64),
            torch.randn(2, 64),
        )
        self.assertEqual(tuple(heading_context.shape), (2, 64))
        self.assertEqual(tuple(auxiliary['direction_logits'].shape), (2, 4, 16))
        self.assertEqual(tuple(auxiliary['joint_attention'].shape), (2, 4, 16))
        self.assertEqual(
            tuple(auxiliary['orientation_probability'].shape), (2, 16)
        )
        self.assertTrue(
            torch.allclose(
                auxiliary['joint_attention'].sum(dim=(1, 2)),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                auxiliary['orientation_probability'].sum(dim=1),
                torch.ones(2),
                atol=1e-6,
            )
        )

    def test_model_forward_loss_and_gradient_isolation(self):
        model = self._build_model()
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)
        self.assertEqual(model.model_name, 'phr5_f_sgcdca')
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['joint_logits'].shape), (2, 4, 16))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())

        target_heading = torch.nn.functional.normalize(
            torch.randn(2, 2), dim=-1
        )
        losses = model.compute_heading_losses(auxiliary, target_heading)
        self.assertEqual(
            set(losses), {'distribution', 'correlation', 'final', 'total'}
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses['total'].backward()
        self.assertGreater(self._gradient_sum(model.sgcdca_heading), 0.0)
        self.assertEqual(self._gradient_sum(model.rst_global_fusion), 0.0)

        model.zero_grad(set_to_none=True)
        position, _ = model(patches)
        position.square().mean().backward()
        self.assertGreater(self._gradient_sum(model.rst_global_fusion), 0.0)
        self.assertEqual(self._gradient_sum(model.sgcdca_heading), 0.0)

    def test_registration_and_config_restore(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA'
        self.assertIs(MODEL_CLASS_DICT[name], PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA)
        self.assertEqual(PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA.default_loss_type, 'sgcdca')
        kwargs = MODEL_KEYWARDS_DICT[name]
        with tempfile.TemporaryDirectory() as config_dir:
            with open(
                f'{config_dir}/training_configure.json', 'w', encoding='utf-8'
            ) as config_file:
                json.dump({'model_class': name, 'model_kwargs': kwargs}, config_file)
            model_class, restored_kwargs = load_config_and_model(config_dir)
        self.assertIs(model_class, PARCASGM_v5a_GlobalRST_PosPrior_SGCDCA)
        self.assertEqual(restored_kwargs, kwargs)


if __name__ == '__main__':
    unittest.main()
