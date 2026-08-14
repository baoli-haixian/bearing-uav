import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior,
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


class ExperimentFTest(unittest.TestCase):
    def _build_model(self):
        kwargs = MODEL_KEYWARDS_DICT['PARCASGM_v5a_GlobalRST_PosPrior']
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return PARCASGM_v5a_GlobalRST_PosPrior(**kwargs)

    @staticmethod
    def _fusion_gradient(model):
        return sum(
            parameter.grad.abs().sum().item()
            for parameter in model.rst_global_fusion.parameters()
            if parameter.grad is not None
        )

    def test_registration_forward_and_auxiliary_shapes(self):
        self.assertIs(
            MODEL_CLASS_DICT['PARCASGM_v5a_GlobalRST_PosPrior'],
            PARCASGM_v5a_GlobalRST_PosPrior,
        )
        model = self._build_model()
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(
            patches, return_aux=True
        )

        self.assertEqual(model.model_name, 'phr5_globalrst_f')
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(
            tuple(auxiliary['global_rst_descriptors'].shape),
            (2, 4, 1024),
        )
        self.assertEqual(tuple(auxiliary['position_prior'].shape), (2, 2))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())

    def test_global_fusion_receives_only_position_gradient(self):
        model = self._build_model()
        patches = torch.randn(2, 5, 3, 64, 64)

        _, heading = model(patches)
        heading.square().mean().backward()
        self.assertEqual(self._fusion_gradient(model), 0.0)

        model.zero_grad(set_to_none=True)
        position, _ = model(patches)
        position.square().mean().backward()
        self.assertGreater(self._fusion_gradient(model), 0.0)

    def test_training_config_can_restore_model_class(self):
        kwargs = MODEL_KEYWARDS_DICT['PARCASGM_v5a_GlobalRST_PosPrior']
        with tempfile.TemporaryDirectory() as config_dir:
            with open(
                f'{config_dir}/training_configure.json',
                'w',
                encoding='utf-8',
            ) as config_file:
                json.dump(
                    {
                        'model_class': 'PARCASGM_v5a_GlobalRST_PosPrior',
                        'model_kwargs': kwargs,
                    },
                    config_file,
                )
            model_class, restored_kwargs = load_config_and_model(config_dir)

        self.assertIs(model_class, PARCASGM_v5a_GlobalRST_PosPrior)
        self.assertEqual(restored_kwargs, kwargs)


if __name__ == '__main__':
    unittest.main()
