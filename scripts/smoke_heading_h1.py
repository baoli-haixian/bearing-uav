"""Fast correctness checks for the H1 circular-heading experiment."""

import argparse
import json

import torch

from config.base_info import PATCH_SIZE
from cvphr.models.posaglreg.models import MODEL_CLASS_DICT, UnitL2Normalize
from cvphr.utils.utils import CircularDirectionLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-model', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--check-entrypoints', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(7)
    loss_fn = CircularDirectionLoss()

    raw = torch.tensor(
        [[3.0, 4.0], [-2.0, 5.0], [0.2, -0.7]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred = UnitL2Normalize()(raw)
    target = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    loss = loss_fn(pred, target)
    loss.backward()

    canonical_pred = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    canonical_target = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    canonical_losses = CircularDirectionLoss(reduction='none')(
        canonical_pred, canonical_target
    )

    norms = torch.linalg.vector_norm(pred.detach(), dim=-1)
    report = {
        'registered_model': 'PARCASGM_v5a_H1' in MODEL_CLASS_DICT,
        'model_default_loss': MODEL_CLASS_DICT['PARCASGM_v5a_H1'].default_loss_type,
        'output_norms': norms.tolist(),
        'max_unit_norm_error': float((norms - 1.0).abs().max()),
        'canonical_losses_aligned_orthogonal_opposite': canonical_losses.tolist(),
        'loss': float(loss.detach()),
        'finite_gradients': bool(torch.isfinite(raw.grad).all()),
        'nonzero_gradient_norm': float(torch.linalg.vector_norm(raw.grad)),
    }

    if args.check_entrypoints:
        from cvphr.test.cvphr_test import test_par
        from cvphr.train.cvphr_train import train_par

        report['entrypoints'] = {
            'train_imported': callable(train_par),
            'test_imported': callable(test_par),
        }

    if args.full_model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MODEL_CLASS_DICT['PARCASGM_v5a_H1']().to(device).eval()
        checkpoint_loaded = False
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            checkpoint_loaded = True
        patches = torch.randn(1, 5, 3, PATCH_SIZE, PATCH_SIZE, device=device)
        with torch.no_grad():
            pos_pred, dir_pred = model(patches)
        full_norms = torch.linalg.vector_norm(dir_pred.float(), dim=-1)
        report['full_model'] = {
            'device': str(device),
            'checkpoint_loaded_strictly': checkpoint_loaded,
            'position_shape': list(pos_pred.shape),
            'heading_shape': list(dir_pred.shape),
            'heading_norms': full_norms.cpu().tolist(),
            'finite_outputs': bool(
                torch.isfinite(pos_pred).all() and torch.isfinite(dir_pred).all()
            ),
        }
    print(json.dumps(report, indent=2))

    assert report['registered_model']
    assert report['model_default_loss'] == 'circular'
    assert report['max_unit_norm_error'] < 1e-6
    assert torch.allclose(
        canonical_losses,
        torch.tensor([0.0, 1.0, 2.0]),
        atol=1e-6,
    )
    assert report['finite_gradients']
    assert report['nonzero_gradient_norm'] > 0.0
    if args.full_model:
        assert report['full_model']['position_shape'] == [1, 2]
        assert report['full_model']['heading_shape'] == [1, 2]
        assert report['full_model']['finite_outputs']
        assert abs(report['full_model']['heading_norms'][0] - 1.0) < 1e-5
    if args.check_entrypoints:
        assert report['entrypoints']['train_imported']
        assert report['entrypoints']['test_imported']


if __name__ == '__main__':
    main()
