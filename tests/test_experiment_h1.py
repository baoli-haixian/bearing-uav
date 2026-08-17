import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from cvphr.models.posaglreg.models import (
    MODEL_CLASS_DICT,
    MODEL_KEYWARDS_DICT,
    PARCASGM_v5a,
    PARCASGM_v5a_H1,
    UnitL2Normalize,
)
from cvphr.utils.utils import CircularDirectionLoss


class _FakeVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            *[nn.Identity() for _ in range(8)],
        )


class ExperimentH1Test(unittest.TestCase):
    def _build_model(self, model_class):
        kwargs = MODEL_KEYWARDS_DICT['PARCASGM_v5a_H1']
        with patch(
            'cvphr.models.posaglreg.models.models.vgg16',
            return_value=_FakeVGG16(),
        ):
            return model_class(**kwargs)

    def test_registration_and_default_loss(self):
        self.assertIs(MODEL_CLASS_DICT['PARCASGM_v5a_H1'], PARCASGM_v5a_H1)
        self.assertEqual(PARCASGM_v5a_H1.default_loss_type, 'circular')

    def test_unit_projection_and_known_circular_losses(self):
        raw = torch.tensor([[3.0, 4.0], [-2.0, 5.0]], requires_grad=True)
        prediction = UnitL2Normalize()(raw)
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(prediction, dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )

        canonical_prediction = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )
        canonical_target = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        losses = CircularDirectionLoss(reduction='none')(
            canonical_prediction, canonical_target
        )
        self.assertTrue(
            torch.allclose(losses, torch.tensor([0.0, 1.0, 2.0]), atol=1e-6)
        )

    def test_loss_has_finite_nonzero_gradient(self):
        raw = torch.tensor([[0.3, 0.7], [-0.4, 0.2]], requires_grad=True)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = CircularDirectionLoss()(UnitL2Normalize()(raw), target)
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(raw.grad.abs().sum().item(), 0.0)

    def test_forward_shape_finiteness_and_unit_heading(self):
        model = self._build_model(PARCASGM_v5a_H1).eval()
        with torch.no_grad():
            position, heading = model(torch.randn(2, 5, 3, 64, 64))
        self.assertEqual(tuple(position.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertTrue(torch.isfinite(position).all())
        self.assertTrue(torch.isfinite(heading).all())
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(heading, dim=-1),
                torch.ones(2),
                atol=1e-5,
            )
        )

    def test_official_parameter_keys_remain_compatible(self):
        baseline = self._build_model(PARCASGM_v5a)
        h1 = self._build_model(PARCASGM_v5a_H1)
        self.assertEqual(set(baseline.state_dict()), set(h1.state_dict()))
        h1.load_state_dict(baseline.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
