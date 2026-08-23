"""Minimal feasibility checks for the proposed SG-CDCA heading branch.

Check 1 is self-contained. Checks 2 and 3 additionally require a Bearing-UAV
metadata CSV and a trained checkpoint; they intentionally skip otherwise.
This script is diagnostic only and is not imported by the training code.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


class FixedPolarDSE(torch.nn.Module):
    """Encode a feature map into fixed, north-referenced angular sectors."""

    def __init__(self, num_directions=16, concentration=12.0):
        super().__init__()
        self.num_directions = int(num_directions)
        self.concentration = float(concentration)

    def forward(self, feature):
        if feature.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feature.shape)}")
        _, _, height, width = feature.shape
        dtype, device = feature.dtype, feature.device
        y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        pixel_angle = torch.atan2(yy, xx)
        radius = torch.sqrt(xx.square() + yy.square()).clamp_max(1.0)
        angles = torch.arange(
            self.num_directions, dtype=dtype, device=device
        ) * (2.0 * math.pi / self.num_directions)
        weights = torch.exp(
            self.concentration
            * (torch.cos(pixel_angle.unsqueeze(0) - angles[:, None, None]) - 1.0)
        )
        weights = weights * radius.unsqueeze(0)
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return torch.einsum("bchw,ahw->bac", feature, weights)


def rotate_map(feature, degrees):
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    matrix = feature.new_tensor(
        [[[cosine, -sine, 0.0], [sine, cosine, 0.0]]]
    ).expand(feature.size(0), -1, -1)
    grid = F.affine_grid(matrix, feature.shape, align_corners=False)
    return F.grid_sample(
        feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def circular_bin_delta(after, before, num_bins):
    raw = (after - before) % num_bins
    return raw - num_bins if raw > num_bins // 2 else raw


def validate_rotation_equivariance(device):
    num_bins = 16
    encoder = FixedPolarDSE(num_bins).to(device)
    axis = torch.linspace(-1.0, 1.0, 65, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    # A single off-centre target has an unambiguous 360-degree direction. Keep
    # this unit check single-modal; multi-modal stability belongs in the real
    # sample statistics below.
    feature = torch.exp(-((xx - 0.58).square() + yy.square()) / 0.025)
    feature = feature[None, None]
    reference = encoder(feature).square().sum(dim=-1).argmax(dim=-1).item()
    rows = []
    passed = True
    for degrees in (22.5, 45.0, 90.0):
        rotated = rotate_map(feature, degrees)
        peak = encoder(rotated).square().sum(dim=-1).argmax(dim=-1).item()
        observed = circular_bin_delta(peak, reference, num_bins)
        # affine_grid samples input coordinates, so positive matrix angles
        # rotate visible content clockwise (negative mathematical direction).
        expected = -round(degrees / (360.0 / num_bins))
        ok = observed == expected
        passed &= ok
        rows.append(
            {
                "rotation_degrees": degrees,
                "expected_bin_shift": expected,
                "observed_bin_shift": observed,
                "pass": ok,
            }
        )
    return {"name": "rotation_equivariance", "pass": passed, "details": rows}


def cyclic_scores(rst_tokens, uav_tokens):
    rst_tokens = F.normalize(rst_tokens, dim=-1, eps=1e-6)
    uav_tokens = F.normalize(uav_tokens, dim=-1, eps=1e-6)
    scores = []
    for shift in range(rst_tokens.size(2)):
        # Positive heading bins align the UAV token at r with the RST token at
        # r + shift. torch.roll(+shift) implements that lookup convention.
        aligned = torch.roll(rst_tokens, shifts=shift, dims=2)
        scores.append(
            (aligned * uav_tokens[:, None]).sum(dim=-1).mean(dim=-1)
        )
    return torch.stack(scores, dim=-1)


def load_checkpoint(model, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key[7:]: value for key, value in payload.items()}
    missing, unexpected = model.load_state_dict(payload, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )


@torch.no_grad()
def validate_real_samples(args, device):
    from torch.utils.data import DataLoader, Subset

    from cvphr.models.posaglreg.models import (
        MODEL_CLASS_DICT,
        MODEL_KEYWARDS_DICT,
        RSBlockDatasetPA_v3q,
    )

    dataset = RSBlockDatasetPA_v3q(args.metadata_csv, is_train=False)
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(dataset), generator=generator)
    if args.sample_pool == "test":
        train_size = int(0.85 * len(dataset))
        val_size = int(0.05 * len(dataset))
        pool = permutation[train_size + val_size :]
    else:
        pool = permutation
    count = min(args.max_samples, len(pool))
    indices = pool[:count].tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False
    )
    model = MODEL_CLASS_DICT[args.model_class](
        **MODEL_KEYWARDS_DICT[args.model_class]
    ).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    encoder = FixedPolarDSE(args.num_directions).to(device)

    heading_margins = []
    heading_hits = []
    predicted_bins = []
    target_bins = []
    semantic_hits = []
    joint_hits = {weight: [] for weight in (0.1, 0.25, 0.5, 1.0)}
    for batch in loader:
        patches = batch["patches"].to(device)
        outputs = [model.sgm(patches[:, index]) for index in range(5)]
        descriptors = torch.stack(
            [output["descriptor_flatten"] for output in outputs[:4]], dim=1
        )
        uav_descriptor = outputs[4]["descriptor_flatten"]
        rst_maps = torch.stack([output["nl_feat"] for output in outputs[:4]], dim=1)
        uav_map = outputs[4]["nl_feat"]

        rst_tokens = torch.stack(
            [encoder(rst_maps[:, index]) for index in range(4)], dim=1
        )
        uav_tokens = encoder(uav_map)
        direction_scores = cyclic_scores(rst_tokens, uav_tokens)
        heading_scores = torch.logsumexp(direction_scores, dim=1)

        target = batch["agl_coords"].to(device)
        target_angle = torch.atan2(target[:, 1], target[:, 0]) % (2.0 * math.pi)
        target_bin = torch.round(
            target_angle * args.num_directions / (2.0 * math.pi)
        ).long() % args.num_directions
        correct = heading_scores.gather(1, target_bin[:, None]).squeeze(1)
        wrong_mean = (
            heading_scores.sum(dim=1) - correct
        ) / (args.num_directions - 1)
        heading_margins.extend((correct - wrong_mean).cpu().tolist())
        predicted_bin = heading_scores.argmax(dim=1)
        heading_hits.extend((predicted_bin == target_bin).cpu().tolist())
        predicted_bins.extend(predicted_bin.cpu().tolist())
        target_bins.extend(target_bin.cpu().tolist())

        known_coords = patches.new_tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        )
        coords = batch["coords"].to(device)
        true_tile = (coords[:, None] - known_coords[None]).square().sum(-1).argmin(1)
        coord_embs = model.coord_encoder(known_coords).unsqueeze(0)
        keys = descriptors + coord_embs if model.add_patch_coord else descriptors
        query = model.neighbors_cross_attn.q_proj(uav_descriptor).unsqueeze(1)
        key = model.neighbors_cross_attn.k_proj(keys)
        semantic = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        semantic = semantic / math.sqrt(query.size(-1))
        direction_tile = torch.logsumexp(direction_scores, dim=-1)

        def standardize(value):
            return (value - value.mean(1, keepdim=True)) / value.std(
                1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)

        semantic_hits.extend((semantic.argmax(1) == true_tile).cpu().tolist())
        for weight in joint_hits:
            joint = standardize(semantic) + weight * standardize(direction_tile)
            joint_hits[weight].extend((joint.argmax(1) == true_tile).cpu().tolist())

    mean_margin = float(torch.tensor(heading_margins).mean())
    predicted_bins = torch.tensor(predicted_bins)
    target_bins = torch.tensor(target_bins)
    calibration = []
    for reverse in (False, True):
        oriented = -predicted_bins if reverse else predicted_bins
        for offset in range(args.num_directions):
            accuracy = float(
                (((oriented + offset) % args.num_directions) == target_bins)
                .float()
                .mean()
            )
            calibration.append((accuracy, reverse, offset))
    best_accuracy, best_reverse, best_offset = max(calibration)
    semantic_accuracy = float(torch.tensor(semantic_hits).float().mean())
    joint_accuracies = {
        str(weight): float(torch.tensor(hits).float().mean())
        for weight, hits in joint_hits.items()
    }
    best_weight = max(joint_accuracies, key=joint_accuracies.get)
    best_joint_accuracy = joint_accuracies[best_weight]
    return [
        {
            "name": "real_heading_bin_separation",
            "samples": count,
            "pass": mean_margin > 0.0 and float(
                torch.tensor(heading_hits).float().mean()
            ) > 1.0 / args.num_directions,
            "mean_correct_minus_wrong_score": mean_margin,
            "top1_bin_accuracy": float(torch.tensor(heading_hits).float().mean()),
            "random_top1_accuracy": 1.0 / args.num_directions,
            "best_global_calibrated_accuracy": best_accuracy,
            "best_global_offset_bins": best_offset,
            "best_global_reverses_direction": best_reverse,
        },
        {
            "name": "joint_ca_true_tile_focus",
            "samples": count,
            "pass": best_joint_accuracy > semantic_accuracy,
            "semantic_tile_accuracy": semantic_accuracy,
            "joint_tile_accuracy_by_weight": joint_accuracies,
            "best_direction_weight": float(best_weight),
            "best_joint_tile_accuracy": best_joint_accuracy,
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model_class", default="PARCASGM_v5a_GlobalRST_PosPrior")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_directions", type=int, default=16)
    parser.add_argument("--direction_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_pool", choices=("test", "full"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    results = [validate_rotation_equivariance(device)]
    metadata_ok = bool(args.metadata_csv and Path(args.metadata_csv).is_file())
    checkpoint_ok = bool(args.checkpoint and Path(args.checkpoint).is_file())
    if metadata_ok and checkpoint_ok:
        results.extend(validate_real_samples(args, device))
    else:
        reason = (
            "requires both --metadata_csv and --checkpoint; no local dataset/"
            "checkpoint was found under /mnt/d/Bearing-UAV_char"
        )
        results.extend(
            [
                {"name": "real_heading_bin_separation", "status": "SKIP", "reason": reason},
                {"name": "joint_ca_true_tile_focus", "status": "SKIP", "reason": reason},
            ]
        )
    report = {"device": str(device), "results": results}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
