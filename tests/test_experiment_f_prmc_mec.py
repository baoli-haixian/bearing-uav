import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    MatchingEvidenceCalibration,
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


class ExperimentFPRMCMECTest(unittest.TestCase):
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

    def test_calibrator_is_identity_at_initialization(self):
        module = MatchingEvidenceCalibration(spatial_temperature=0.05)
        logits = torch.randn(2, 4)
        response = torch.randn(2, 36, 9, 9)
        probability = torch.softmax(torch.randn(2, 36), dim=-1)
        resultant = torch.rand(2)
        calibrated, auxiliary = module(
            logits, response, probability, resultant
        )
        self.assertTrue(torch.equal(calibrated, logits))
        self.assertTrue(torch.allclose(
            auxiliary['mec_rst_probability'].sum(-1), torch.ones(2)
        ))
        self.assertTrue(torch.isfinite(auxiliary['mec_confidence']).all())

    def test_corner_membership_matches_official_rst_order(self):
        membership = MatchingEvidenceCalibration._corner_membership(
            3, torch.device('cpu'), torch.float32
        )
        self.assertTrue(torch.equal(membership[0], torch.tensor([1., 0., 0., 0.])))
        self.assertTrue(torch.equal(membership[2], torch.tensor([0., 1., 0., 0.])))
        self.assertTrue(torch.equal(membership[6], torch.tensor([0., 0., 1., 0.])))
        self.assertTrue(torch.equal(membership[8], torch.tensor([0., 0., 0., 1.])))
        self.assertTrue(torch.allclose(membership.sum(-1), torch.ones(9)))

    def test_detach_and_joint_position_gradient_paths(self):
        patches = torch.randn(2, 5, 3, 64, 64)
        detach = self._build(
            'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECDetach'
        )
        detach.matching_evidence_calibration.alpha.data.fill_(0.25)
        position, _ = detach(patches)
        position.square().mean().backward()
        self.assertEqual(self._gradient_sum(detach.prmc_head), 0.0)
        self.assertGreater(
            self._gradient_sum(detach.matching_evidence_calibration), 0.0
        )

        joint = self._build(
            'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECJoint'
        )
        joint.matching_evidence_calibration.alpha.data.fill_(0.25)
        position, _ = joint(patches)
        position.square().mean().backward()
        self.assertGreater(self._gradient_sum(joint.prmc_head), 0.0)
        self.assertGreater(
            self._gradient_sum(joint.matching_evidence_calibration), 0.0
        )

    def test_registered_models_forward_and_shuffle_control(self):
        patches = torch.randn(2, 5, 3, 64, 64)
        for name in (
            'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECDetach',
            'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECJoint',
            'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECShuffle',
        ):
            model = self._build(name)
            model.matching_evidence_calibration.alpha.data.fill_(0.25)
            position, heading, auxiliary = model(patches, return_aux=True)
            self.assertEqual(tuple(position.shape), (2, 2))
            self.assertEqual(tuple(heading.shape), (2, 2))
            self.assertEqual(tuple(auxiliary['mec_spatial_probability'].shape), (2, 81))
            self.assertEqual(tuple(auxiliary['mec_rst_probability'].shape), (2, 4))
            self.assertTrue(torch.isfinite(position).all())
            self.assertTrue(torch.isfinite(heading).all())

    def test_f_prmc_checkpoint_initializes_only_mec(self):
        baseline = self._build('PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1')
        mec = self._build('PARCASGM_v5a_GlobalRST_PosPrior_PRMC_MECJoint')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'baseline.pth'
            torch.save({'model_state_dict': baseline.state_dict()}, checkpoint)
            incompatible = load_initial_model_weights(mec, checkpoint)
        self.assertEqual(
            incompatible.missing_keys,
            ['matching_evidence_calibration.alpha'],
        )
        self.assertFalse(incompatible.unexpected_keys)

        baseline.eval()
        mec.eval()
        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.inference_mode():
            baseline_outputs = baseline(patches)
            mec_outputs = mec(patches)
        for baseline_output, mec_output in zip(baseline_outputs, mec_outputs):
            self.assertTrue(torch.allclose(
                baseline_output, mec_output, atol=1e-6, rtol=1e-6
            ))


if __name__ == '__main__':
    unittest.main()
