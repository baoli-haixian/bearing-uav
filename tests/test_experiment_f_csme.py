import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torchvision.models.vgg import make_layers, cfgs

from tests.test_experiment_f_cdm import ExperimentFCDMTest
from cvphr.models.posaglreg.models import (
    ContentSpatialMomentEncoding, MODEL_CLASS_DICT, MODEL_KEYWARDS_DICT,
    PARCASGM_v5a_GlobalRST_PosPrior,
    PARCASGM_v5a_GlobalRST_PosPrior_CSME, load_config_and_model,
)


class ExperimentFCSMETest(unittest.TestCase):
    def test_real_vgg_features_standard_input(self):
        # Real VGG-16 convolutional architecture, random weights, no download.
        backbone = torch.nn.Module()
        backbone.features = make_layers(cfgs['D'])
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_CSME'
        with patch('cvphr.models.posaglreg.models.models.vgg16', return_value=backbone):
            model = MODEL_CLASS_DICT[name](**MODEL_KEYWARDS_DICT[name]).eval()
        with torch.no_grad():
            position, heading, aux = model(torch.randn(1, 5, 3, 256, 256), return_aux=True)
        self.assertEqual(tuple(position.shape), (1, 2))
        self.assertEqual(tuple(heading.shape), (1, 2))
        self.assertEqual(tuple(aux['spatial_moments'].shape), (1, 4, 4, 5))
        self.assertTrue(torch.isfinite(position).all() and torch.isfinite(heading).all())

    def test_known_moments_and_empty_clusters(self):
        weights = torch.zeros(1, 3, 2, 2)
        weights[0, 0, 0, 1] = 1
        weights[0, 1] = 1
        moments, valid = ContentSpatialMomentEncoding.spatial_moments(weights)
        torch.testing.assert_close(moments[0, 0], torch.tensor([.5, -.5, 0., 0., 0.]))
        torch.testing.assert_close(moments[0, 1], torch.tensor([0., 0., .25, .25, 0.]))
        self.assertFalse(valid[0, 2])
        module = ContentSpatialMomentEncoding()
        module.scale.data.fill_(.2)
        desc = F.normalize(torch.randn(1, 3, 256), dim=-1)
        out, _ = module(desc, weights)
        torch.testing.assert_close(out[:, 2], desc[:, 2], rtol=0, atol=0)

    def test_module_gradients(self):
        module = ContentSpatialMomentEncoding()
        module.scale.data.fill_(.1)
        weights = torch.rand(2, 4, 16, 16, requires_grad=True)
        desc = F.normalize(torch.randn(2, 4, 256), dim=-1)
        out, _ = module(desc, weights)
        (out * torch.randn_like(out)).sum().backward()
        for gradient in (weights.grad, module.projector[0].weight.grad, module.scale.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_baseline_identity_psg_isolation_and_backward(self):
        build = ExperimentFCDMTest._build
        torch.manual_seed(17)
        baseline = build(PARCASGM_v5a_GlobalRST_PosPrior).eval()
        model = build(PARCASGM_v5a_GlobalRST_PosPrior_CSME).eval()
        missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(missing and all(key.startswith('csme.') for key in missing))
        self.assertEqual(unexpected, [])
        patches = torch.randn(2, 5, 3, 64, 64)
        with torch.no_grad():
            bp, bh, ba = baseline(patches, return_aux=True)
        pos, heading, aux = model(patches, return_aux=True)
        torch.testing.assert_close(pos, bp, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(heading, bh, atol=1e-6, rtol=1e-5)
        self.assertEqual(tuple(pos.shape), (2, 2))
        self.assertEqual(tuple(heading.shape), (2, 2))
        self.assertEqual(tuple(aux['spatial_moments'].shape), (2, 4, 4, 5))
        (.8 * F.smooth_l1_loss(pos, torch.randn_like(pos)) +
         .2 * F.smooth_l1_loss(heading, torch.randn_like(heading))).backward()
        self.assertGreater(model.csme.scale.grad.abs().item(), 0)
        model.zero_grad()
        model.csme.scale.data.fill_(.1)
        pos, heading, aux = model(patches, return_aux=True)
        torch.testing.assert_close(aux['position_prior'], ba['position_prior'])
        torch.testing.assert_close(aux['global_rst_descriptors'], ba['global_rst_descriptors'])
        (.8 * pos.square().mean() + .2 * heading.square().mean()).backward()
        self.assertGreater(model.csme.projector[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_config_and_checkpoint_roundtrip(self):
        name = 'PARCASGM_v5a_GlobalRST_PosPrior_CSME'
        self.assertIs(MODEL_CLASS_DICT[name], PARCASGM_v5a_GlobalRST_PosPrior_CSME)
        with tempfile.TemporaryDirectory() as directory:
            with open(f'{directory}/training_configure.json', 'w') as stream:
                json.dump({'model_class': name, 'model_kwargs': MODEL_KEYWARDS_DICT[name]}, stream)
            cls, kwargs = load_config_and_model(directory)
            self.assertEqual(kwargs, MODEL_KEYWARDS_DICT[name])
            model = ExperimentFCDMTest._build(cls)
            torch.save(model.state_dict(), f'{directory}/model.pth')
            restored = ExperimentFCDMTest._build(cls)
            restored.load_state_dict(torch.load(f'{directory}/model.pth', map_location='cpu'), strict=True)


if __name__ == '__main__':
    unittest.main()
