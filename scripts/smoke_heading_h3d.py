"""CUDA/CPU smoke check for experiment H3-D without loading the dataset."""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.base_info import PATCH_SIZE
from cvphr.models.posaglreg.models import MODEL_CLASS_DICT, MODEL_KEYWARDS_DICT
from cvphr.train.cvphr_train import load_initial_model_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(23)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = MODEL_CLASS_DICT['PARCASGM_v5a_GPRVH']
    model = model_class(**MODEL_KEYWARDS_DICT['PARCASGM_v5a_GPRVH']).to(device)
    checkpoint_report = None
    if args.checkpoint:
        incompatible = load_initial_model_weights(model, args.checkpoint)
        checkpoint_report = {
            'missing_count': len(incompatible.missing_keys),
            'missing_only_h3d': all(
                key.startswith('gprv_head.') for key in incompatible.missing_keys
            ),
            'unexpected': list(incompatible.unexpected_keys),
        }

    model.train()
    patches = torch.randn(
        args.batch_size, 5, 3, PATCH_SIZE, PATCH_SIZE, device=device
    )
    target_angle = torch.linspace(0.0, 1.0, args.batch_size, device=device)
    target_heading = torch.stack(
        (target_angle.cos(), target_angle.sin()), dim=-1
    )
    target_position = torch.zeros(args.batch_size, 2, device=device)
    position, heading, auxiliary = model(patches, return_aux=True)
    losses = model.compute_heading_losses(
        auxiliary, target_heading, target_position=target_position
    )
    losses['total'].backward()

    head_gradient = sum(
        parameter.grad.float().abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith('gprv_head.') and parameter.grad is not None
    )
    base_has_gradient = any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith('gprv_head.')
    )
    report = {
        'device': str(device),
        'registered': model_class.__name__,
        'checkpoint': checkpoint_report,
        'position_shape': list(position.shape),
        'heading_shape': list(heading.shape),
        'pose_volume_shape': list(auxiliary['pose_volume_logits'].shape),
        'probability_shape': list(auxiliary['orientation_probability'].shape),
        'heading_norm': torch.linalg.vector_norm(
            heading.float(), dim=-1
        ).detach().cpu().tolist(),
        'losses': {
            key: value.detach().float().item() for key, value in losses.items()
        },
        'finite_outputs': bool(
            torch.isfinite(position).all()
            and torch.isfinite(heading).all()
            and torch.isfinite(losses['total'])
        ),
        'head_gradient_sum': head_gradient,
        'base_has_gradient': base_has_gradient,
    }
    print(json.dumps(report, indent=2))

    assert report['position_shape'] == [args.batch_size, 2]
    assert report['heading_shape'] == [args.batch_size, 2]
    assert report['pose_volume_shape'] == [args.batch_size, 81, 36]
    assert report['probability_shape'] == [args.batch_size, 36]
    assert report['finite_outputs']
    assert max(abs(value - 1.0) for value in report['heading_norm']) < 1e-5
    assert report['head_gradient_sum'] > 0.0
    assert not report['base_has_gradient']
    if checkpoint_report is not None:
        assert checkpoint_report['missing_only_h3d']
        assert not checkpoint_report['unexpected']


if __name__ == '__main__':
    main()
