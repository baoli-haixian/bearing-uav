import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    LocalCandidatePositionRefiner,
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior_LCPR1,
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


class ExperimentFLCPR1Test(unittest.TestCase):
    def _build_model(self):
        kwargs = MODEL_KEYWARDS_DICT[
            'PARCASGM_v5a_GlobalRST_PosPrior_LCPR1'
        ]
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return PARCASGM_v5a_GlobalRST_PosPrior_LCPR1(**kwargs)

    @staticmethod
    def _gradient_sum(module):
        return sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    def test_refiner_shapes_bounds_and_backward(self):
        refiner = LocalCandidatePositionRefiner(
            input_dim=16, feature_dim=8, radius=0.15
        )
        rst_maps = torch.randn(2, 4, 16, 8, 8, requires_grad=True)
        uav_map = torch.randn(2, 16, 8, 8, requires_grad=True)
        coarse = torch.tensor([[0.0, 0.0], [0.95, -0.95]], requires_grad=True)

        position, auxiliary = refiner(rst_maps, uav_map, coarse)
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['candidate_positions'].shape), (2, 9, 2))
        self.assertEqual(tuple(auxiliary['candidate_weights'].shape), (2, 9))
        self.assertEqual(tuple(auxiliary['refinement_gate'].shape), (2, 1))
        self.assertTrue(torch.all(position >= -1.0))
        self.assertTrue(torch.all(position <= 1.0))
        self.assertTrue(torch.allclose(
            auxiliary['candidate_weights'].sum(dim=-1), torch.ones(2)
        ))
        self.assertTrue(torch.isfinite(position).all())

        position.square().mean().backward()
        self.assertGreater(self._gradient_sum(refiner), 0.0)
        self.assertIsNotNone(rst_maps.grad)
        self.assertIsNotNone(uav_map.grad)
        self.assertIsNotNone(coarse.grad)

    def test_registration_forward_and_gradient_isolation(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_GlobalRST_PosPrior_LCPR1'],
            PARCASGM_v5a_GlobalRST_PosPrior_LCPR1,
        )
        model = self._build_model()
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)

        self.assertEqual(model.model_name, 'phr5_globalrst_f_lcpr1_joint100')
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['coarse_position'].shape), (2, 2))
        self.assertEqual(tuple(auxiliary['candidate_weights'].shape), (2, 9))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())

        heading.square().mean().backward()
        self.assertEqual(
            self._gradient_sum(model.local_position_refiner), 0.0
        )

        model.zero_grad(set_to_none=True)
        position, _ = model(patches)
        position.square().mean().backward()
        self.assertGreater(
            self._gradient_sum(model.local_position_refiner), 0.0
        )

    def test_training_config_restores_model(self):
        key = 'PARCASGM_v5a_GlobalRST_PosPrior_LCPR1'
        kwargs = MODEL_KEYWARDS_DICT[key]
        with tempfile.TemporaryDirectory() as config_dir:
            with open(
                f'{config_dir}/training_configure.json', 'w', encoding='utf-8'
            ) as config_file:
                json.dump({'model_class': key, 'model_kwargs': kwargs}, config_file)
            model_class, restored_kwargs = load_config_and_model(config_dir)

        self.assertIs(model_class, PARCASGM_v5a_GlobalRST_PosPrior_LCPR1)
        self.assertEqual(restored_kwargs, kwargs)


if __name__ == '__main__':
    unittest.main()
