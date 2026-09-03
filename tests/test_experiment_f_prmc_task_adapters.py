import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior_PRMC_TaskAdapters,
    SpatialTaskAdapter,
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


class ExperimentFPRMCTaskAdaptersTest(unittest.TestCase):
    def _build(self, name):
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

    def test_adapter_is_exact_identity_at_initialization(self):
        adapter = SpatialTaskAdapter(dim=16, bottleneck=8, groups=4)
        feature = torch.randn(2, 16, 7, 7)
        self.assertTrue(torch.equal(adapter(feature), feature))

    def test_model_registration_forward_and_statistics(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_TaskAdapters'
        self.assertIs(
            MODEL_CLASS_DICT[name],
            PARCASGM_v5a_GlobalRST_PosPrior_PRMC_TaskAdapters,
        )
        model = self._build(name)
        patches = torch.randn(2, 5, 3, 64, 64)
        position, heading, auxiliary = model(patches, return_aux=True)
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(auxiliary['orientation_probability'].shape), (2, 36))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())
        for key in (
            'adapter_alpha_position',
            'adapter_alpha_heading_rst',
            'adapter_alpha_heading_uav',
        ):
            self.assertEqual(float(auxiliary[key]), 0.0)

    def test_task_losses_update_only_their_adapter(self):
        model = self._build(
            'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_TaskAdapters'
        )
        for adapter in (
            model.position_adapter,
            model.heading_rst_adapter,
            model.heading_uav_adapter,
        ):
            adapter.alpha.data.fill_(0.1)
        patches = torch.randn(2, 5, 3, 64, 64)

        position, _ = model(patches)
        position.square().mean().backward()
        self.assertGreater(self._gradient_sum(model.position_adapter), 0.0)
        self.assertEqual(self._gradient_sum(model.heading_rst_adapter), 0.0)
        self.assertEqual(self._gradient_sum(model.heading_uav_adapter), 0.0)

        model.zero_grad(set_to_none=True)
        _, heading, auxiliary = model(patches, return_aux=True)
        target = torch.nn.functional.normalize(torch.randn_like(heading), dim=-1)
        model.compute_heading_losses(auxiliary, target)['total'].backward()
        self.assertEqual(self._gradient_sum(model.position_adapter), 0.0)
        self.assertGreater(self._gradient_sum(model.heading_rst_adapter), 0.0)
        self.assertGreater(self._gradient_sum(model.heading_uav_adapter), 0.0)
        self.assertGreater(self._gradient_sum(model.sgm), 0.0)

    def test_f_prmc_checkpoint_initializes_only_new_adapters(self):
        baseline = self._build('PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1')
        adapted = self._build(
            'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_TaskAdapters'
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'baseline.pth'
            torch.save({'model_state_dict': baseline.state_dict()}, checkpoint)
            incompatible = load_initial_model_weights(adapted, checkpoint)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            key.startswith(adapted.initialization_missing_prefixes)
            for key in incompatible.missing_keys
        ))


if __name__ == '__main__':
    unittest.main()
