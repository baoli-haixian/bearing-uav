import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST,
    RSTGlobalContextFusion,
    load_config_and_model,
)


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        # models.py removes the final eight children. The retained stride-8
        # convolution produces the same 32x32 backbone map for a 256x256 patch.
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class RSTGlobalContextFusionTest(unittest.TestCase):
    def test_mosaic_round_trip_preserves_patch_order(self):
        rst_maps = torch.stack(
            [torch.full((1, 2, 3, 5), float(index)) for index in range(1, 5)],
            dim=1,
        )
        mosaic = RSTGlobalContextFusion.build_mosaic(rst_maps)
        restored = RSTGlobalContextFusion.split_mosaic(mosaic, 3, 5)

        self.assertEqual(tuple(mosaic.shape), (1, 2, 6, 10))
        self.assertTrue(torch.equal(restored, rst_maps))

    def test_fusion_shape_finiteness_and_gradients(self):
        module = RSTGlobalContextFusion(
            feature_dim=16,
            descriptor_dim=64,
            token_grid_size=4,
            num_heads=4,
            feedforward_dim=32,
            dropout=0.0,
        )
        rst_maps = torch.randn(2, 4, 16, 8, 8, requires_grad=True)
        rst_descriptors = torch.randn(2, 4, 64, requires_grad=True)

        fused = module(rst_maps, rst_descriptors)
        gradient_probe = torch.linspace(-1.0, 1.0, fused.size(-1)).view(1, 1, -1)
        (fused * gradient_probe).sum().backward()

        self.assertEqual(tuple(fused.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(fused).all())
        baseline_descriptors = torch.nn.functional.normalize(
            rst_descriptors.detach(), p=2, dim=-1
        )
        initial_similarity = torch.nn.functional.cosine_similarity(
            fused.detach(), baseline_descriptors, dim=-1
        ).mean()
        self.assertGreater(initial_similarity.item(), 0.98)
        grad_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_model_registration_and_end_to_end_forward(self):
        self.assertIs(MODEL_CLASS_DICT['PARCASGM_v5a_GlobalRST'], PARCASGM_v5a_GlobalRST)
        model_kwargs = MODEL_KEYWARDS_DICT['PARCASGM_v5a_GlobalRST']

        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            model = PARCASGM_v5a_GlobalRST(**model_kwargs)

        patches = torch.randn(1, 5, 3, 64, 64)
        pos_pred, dir_pred = model(patches)
        (pos_pred.square().mean() + dir_pred.square().mean()).backward()

        self.assertEqual(tuple(pos_pred.shape), (1, 2))
        self.assertEqual(tuple(dir_pred.shape), (1, 2))
        self.assertTrue(torch.isfinite(pos_pred).all())
        self.assertTrue(torch.isfinite(dir_pred).all())
        fusion_grad_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.rst_global_fusion.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(fusion_grad_sum, 0.0)

    def test_training_config_can_restore_model_class(self):
        model_kwargs = MODEL_KEYWARDS_DICT['PARCASGM_v5a_GlobalRST']
        with tempfile.TemporaryDirectory() as config_dir:
            config_path = f'{config_dir}/training_configure.json'
            with open(config_path, 'w', encoding='utf-8') as config_file:
                json.dump(
                    {
                        'model_class': 'PARCASGM_v5a_GlobalRST',
                        'model_kwargs': model_kwargs,
                    },
                    config_file,
                )

            model_class, restored_kwargs = load_config_and_model(config_dir)

        self.assertIs(model_class, PARCASGM_v5a_GlobalRST)
        self.assertEqual(restored_kwargs, model_kwargs)


if __name__ == '__main__':
    unittest.main()
