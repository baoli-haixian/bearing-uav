"""Measure localization evidence and complementarity in a trained P-RMC head."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.05)
    return parser.parse_args()


def candidate_coordinates(size, device):
    axis = torch.linspace(-1.0, 1.0, size, device=device)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def spatial_probability(response, orientation_probability, temperature):
    joint_logits = (
        response.flatten(2).float() / temperature
        + orientation_probability.float().clamp_min(1e-8).log().unsqueeze(-1)
    )
    return torch.softmax(torch.logsumexp(joint_logits, dim=1), dim=-1)


def pearson(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def rank(values):
    order = np.argsort(values, kind='stable')
    output = np.empty_like(order, dtype=np.float64)
    output[order] = np.arange(len(values), dtype=np.float64)
    return output


def error_summary(error):
    return {
        'mean_norm_distance': float(error.mean()),
        'median_norm_distance': float(error.median()),
        'within_0.25': float((error <= 0.25).float().mean()),
        'within_0.50': float((error <= 0.50).float().mean()),
    }


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError('temperature must be positive')
    code_root = args.code_root.resolve()
    data_root = args.data_root.resolve()
    metadata = args.metadata.resolve()
    checkpoint_path = args.checkpoint.resolve()
    sys.path.insert(0, str(code_root))

    from cvphr.models.posaglreg import models

    original_vgg16 = tv_models.vgg16
    models.models.vgg16 = lambda pretrained=True: original_vgg16(weights=None)
    name = 'PARCASGM_v5a_GlobalRST_PosPrior_PRMC_H1'
    model = models.MODEL_CLASS_DICT[name](**models.MODEL_KEYWARDS_DICT[name])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)

    # Metadata paths are relative to the dataset repository root.
    os.chdir(data_root)
    dataset = models.RSBlockDatasetPA_v3q(str(metadata), is_train=False)
    generator = torch.Generator().manual_seed(42)
    _, validation, _ = torch.utils.data.random_split(
        range(len(dataset)), [76500, 4500, 9000], generator=generator
    )
    indices = validation.indices[:min(args.limit, len(validation.indices))]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    stored = {
        key: [] for key in (
            'target', 'model_position', 'prmc_expected', 'prmc_argmax',
            'model_error', 'prmc_error', 'entropy', 'resultant',
            'target_window', 'top_indices',
        )
    }
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            patches = batch['patches'].to(device, non_blocking=True)
            target = batch['coords'].to(device)
            position, _, auxiliary = model(patches, return_aux=True)
            probability = spatial_probability(
                auxiliary['response'],
                auxiliary['orientation_probability'],
                args.temperature,
            )
            grid_size = auxiliary['response'].size(-1)
            coordinates = candidate_coordinates(grid_size, device)
            expected = probability @ coordinates
            top_indices = probability.topk(k=3, dim=-1).indices
            argmax = coordinates[top_indices[:, 0]]
            target_window = torch.cdist(
                target[:, None], coordinates[None]
            ).squeeze(1).argmin(-1)
            entropy = -(
                probability * probability.clamp_min(1e-8).log()
            ).sum(-1) / math.log(probability.size(-1))

            values = {
                'target': target,
                'model_position': position,
                'prmc_expected': expected,
                'prmc_argmax': argmax,
                'model_error': (position - target).norm(dim=-1),
                'prmc_error': (argmax - target).norm(dim=-1),
                'entropy': entropy,
                'resultant': auxiliary['resultant'],
                'target_window': target_window,
                'top_indices': top_indices,
            }
            for key, value in values.items():
                stored[key].append(value.detach().cpu())
            if (batch_number + 1) % 50 == 0:
                print(
                    f'processed={min((batch_number + 1) * args.batch_size, len(indices))}',
                    flush=True,
                )

    merged = {key: torch.cat(value) for key, value in stored.items()}
    model_error = merged['model_error']
    prmc_error = merged['prmc_error']
    target_window = merged['target_window']
    top_indices = merged['top_indices']
    target_quadrant = (merged['target'] >= 0).to(torch.long)
    prmc_quadrant = (merged['prmc_argmax'] >= 0).to(torch.long)
    quadrant_accuracy = (target_quadrant == prmc_quadrant).all(-1).float().mean()

    model_np = model_error.numpy()
    prmc_np = prmc_error.numpy()
    report = {
        'samples': len(indices),
        'device': str(device),
        'temperature': args.temperature,
        'model_position': error_summary(model_error),
        'prmc_argmax': error_summary(prmc_error),
        'prmc_expected': error_summary(
            (merged['prmc_expected'] - merged['target']).norm(dim=-1)
        ),
        'window_retrieval': {
            'top1': float((top_indices[:, 0] == target_window).float().mean()),
            'top3': float((top_indices == target_window[:, None]).any(-1).float().mean()),
            'quadrant_accuracy': float(quadrant_accuracy),
        },
        'complementarity': {
            'prmc_better_fraction': float((prmc_error < model_error).float().mean()),
            'oracle_min_mean_norm_distance': float(
                torch.minimum(model_error, prmc_error).mean()
            ),
            'pearson_error_correlation': pearson(model_np, prmc_np),
            'spearman_error_correlation': pearson(rank(model_np), rank(prmc_np)),
            'both_outside_0.25': float(
                ((model_error > 0.25) & (prmc_error > 0.25)).float().mean()
            ),
        },
        'confidence': {
            'mean_spatial_entropy': float(merged['entropy'].mean()),
            'mean_heading_resultant': float(merged['resultant'].mean()),
            'error_entropy_correlation': pearson(
                prmc_np, merged['entropy'].numpy()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
