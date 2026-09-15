import unittest

import torch
import torch.nn.functional as F

from tests.test_experiment_f_cdm import ExperimentFCDMTest
from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior,
    PARCASGM_v5a_GlobalRST_PosPrior_CDMAuxOnly,
    PARCASGM_v5a_GlobalRST_PosPrior_CDMResidualOnly,
)


class ExperimentFCDMAblationTest(unittest.TestCase):
    def test_residual_only_uses_original_heading_loss(self):
        model = ExperimentFCDMTest._build(
            PARCASGM_v5a_GlobalRST_PosPrior_CDMResidualOnly
        ).eval()
        self.assertEqual(model.cdm_aux_heading_weight, 0.0)
        patches = torch.randn(2, 5, 3, 64, 64)
        model.cdm.residual_scale.data.fill_(0.1)
        pos, heading, aux = model(patches, return_aux=True)
        target = torch.randn_like(heading)
        losses = model.compute_heading_losses(aux, target)
        torch.testing.assert_close(
            losses['total'], F.smooth_l1_loss(heading, target)
        )
        (pos.square().mean() + losses['total']).backward()
        self.assertGreater(model.cdm.residual_scale.grad.abs().sum().item(), 0)
        self.assertIsNone(model.cdm.aux_heading[0].weight.grad)

    def test_aux_only_preserves_f_forward_and_trains_aux(self):
        torch.manual_seed(13)
        baseline = ExperimentFCDMTest._build(
            PARCASGM_v5a_GlobalRST_PosPrior
        ).eval()
        model = ExperimentFCDMTest._build(
            PARCASGM_v5a_GlobalRST_PosPrior_CDMAuxOnly
        ).eval()
        missing, unexpected = model.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertTrue(all(key.startswith('cdm.') for key in missing))
        self.assertEqual(unexpected, [])
        model.cdm.residual_scale.data.fill_(0.7)
        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.no_grad():
            baseline_pos, baseline_heading = baseline(patches)
        pos, heading, aux = model(patches, return_aux=True)
        torch.testing.assert_close(pos, baseline_pos, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(heading, baseline_heading, atol=1e-6, rtol=1e-5)
        target = F.normalize(torch.randn_like(heading), dim=-1)
        losses = model.compute_heading_losses(aux, target)
        expected = F.smooth_l1_loss(heading, target)
        expected += 0.1 * F.smooth_l1_loss(aux['heading_aux'], target)
        torch.testing.assert_close(losses['total'], expected)
        losses['total'].backward()
        self.assertIsNone(model.cdm.residual_scale.grad)
        self.assertGreater(model.cdm.aux_heading[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.cdm.content_projector[0].weight.grad.abs().sum().item(), 0)

    def test_registry(self):
        for cls in (
            PARCASGM_v5a_GlobalRST_PosPrior_CDMAuxOnly,
            PARCASGM_v5a_GlobalRST_PosPrior_CDMResidualOnly,
        ):
            self.assertIs(MODEL_CLASS_DICT[cls.__name__], cls)
            self.assertIn(cls.__name__, MODEL_KEYWARDS_DICT)
        self.assertEqual(
            MODEL_KEYWARDS_DICT[
                PARCASGM_v5a_GlobalRST_PosPrior_CDMResidualOnly.__name__
            ]['cdm_aux_heading_weight'], 0.0
        )


if __name__ == '__main__':
    unittest.main()
