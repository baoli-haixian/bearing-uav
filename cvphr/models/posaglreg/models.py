# Bearing-UAV model.
import os
import math
import cv2
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from importlib import import_module
from typing import Literal, Tuple, List, Dict


from cvphr.sceneGraphEncodingNet.nets import CSMG, JointNet
from cvphr.sceneGraphEncodingNet.nets import CSMG_soft, JointNet_soft  #for phr5

from cvphr.utils.utils_transform import transform_pipeline3
from cvphr.utils.utils_transform import transform_pipeline1_gentle
from cvphr.utils.utils_transform import transform_pipeline_weather


"""****************************************************************************
*                                                                             *
*                                  CVPHR-model                                *
*                                                                             *
****************************************************************************"""
class NeighborsCrossAttention(nn.Module):
    def __init__(self, feat_dim=256, reduction_ratio=1):
        """
        Lightweight cross attention module
        Core idea: Use target feature fi as Query to perform attention-weighted aggregation on neighbor_feats to extract relevant context.
        """
        super().__init__()
        reduced_dim = feat_dim // reduction_ratio
        
        self.q_proj = nn.Linear(feat_dim, reduced_dim)
        self.k_proj = nn.Linear(feat_dim, reduced_dim)
        self.v_proj = nn.Linear(feat_dim, reduced_dim)
        
    def forward(self, query, keys):
        # query: [B, num_clusters * D]
        # keys:  [B, 4, num_clusters * D]
        assert query.shape[-1] == self.q_proj.in_features, \
            f"Expected query with dim {self.q_proj.in_features}, got {query.shape[-1]}"

        # query==center_feat，keys==neighbor_feats
        q = self.q_proj(query).unsqueeze(1)  # [B, 1, num_clusters * D/r]
        k = self.k_proj(keys)                # [B, 4, num_clusters * D/r]
        v = self.v_proj(keys)                # [B, 4, num_clusters * D/r]
        
        attn = torch.bmm(q, k.transpose(1,2)) / np.sqrt(q.size(-1))  # [B, 1, 4]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).squeeze(1)  # [B, num_clusters * D/r]
        return out


class UnitL2Normalize(nn.Module):
    """Project a 2-D heading vector onto the unit circle."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, vector):
        return F.normalize(vector, p=2, dim=-1, eps=self.eps)

class SimilarityPositionPrior(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.cos = nn.CosineSimilarity(dim=-1)

        # Fixed neighborhood relative coordinates (same as position encoding)
        self.register_buffer('rel_coords', torch.tensor(
            [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            dtype=torch.float32
        ))

    def forward(self, ft, neighbor_feats, temperature=1.0, return_weights=False):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        B, D = ft.size()
        fi_expand = ft.unsqueeze(1).expand(-1, 4, -1)  # [B, 4, D]
        sim = self.cos(fi_expand, neighbor_feats)  # [B, 4]
        weights = torch.softmax(sim / temperature, dim=1)  # [B, 4]

        # Weighted relative position
        pos_prior = weights.unsqueeze(2) * self.rel_coords.unsqueeze(0)  # [B, 4, 2]
        pos_prior = pos_prior.sum(dim=1)  # [B, 2]
        if return_weights:
            return pos_prior, weights
        return pos_prior


class RSTGlobalContextFusion(nn.Module):
    """Fuse the four contiguous RST feature maps and inject global context."""

    def __init__(
        self,
        feature_dim=256,
        descriptor_dim=1024,
        token_grid_size=4,
        num_heads=8,
        feedforward_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.feature_dim = feature_dim
        self.descriptor_dim = descriptor_dim
        self.token_grid_size = token_grid_size

        self.cross_boundary = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.token_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, token_grid_size * token_grid_size, feature_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.context_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, descriptor_dim),
            nn.GELU(),
            nn.LayerNorm(descriptor_dim),
        )
        self.gate = nn.Linear(descriptor_dim * 2, 1)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    @staticmethod
    def build_mosaic(rst_maps):
        """Arrange p1-p4 in their real crop order: [[p1, p2], [p3, p4]]."""
        if rst_maps.dim() != 5 or rst_maps.size(1) != 4:
            raise ValueError(
                f"Expected RST maps shaped [B, 4, C, H, W], got {tuple(rst_maps.shape)}"
            )
        top = torch.cat((rst_maps[:, 0], rst_maps[:, 1]), dim=-1)
        bottom = torch.cat((rst_maps[:, 2], rst_maps[:, 3]), dim=-1)
        return torch.cat((top, bottom), dim=-2)

    @staticmethod
    def split_mosaic(mosaic, patch_height, patch_width):
        """Split a 2x2 mosaic back into p1-p4 order."""
        expected_hw = (patch_height * 2, patch_width * 2)
        if mosaic.shape[-2:] != expected_hw:
            raise ValueError(
                f"Expected mosaic spatial size {expected_hw}, got {tuple(mosaic.shape[-2:])}"
            )
        return torch.stack(
            (
                mosaic[:, :, :patch_height, :patch_width],
                mosaic[:, :, :patch_height, patch_width:],
                mosaic[:, :, patch_height:, :patch_width],
                mosaic[:, :, patch_height:, patch_width:],
            ),
            dim=1,
        )

    def forward(self, rst_maps, rst_descriptors, return_aux=False):
        if rst_descriptors.dim() != 3 or rst_descriptors.size(1) != 4:
            raise ValueError(
                "Expected RST descriptors shaped [B, 4, descriptor_dim], "
                f"got {tuple(rst_descriptors.shape)}"
            )
        if rst_maps.size(0) != rst_descriptors.size(0):
            raise ValueError("RST maps and descriptors must have the same batch size")
        if rst_maps.size(2) != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} map channels, got {rst_maps.size(2)}"
            )
        if rst_descriptors.size(2) != self.descriptor_dim:
            raise ValueError(
                f"Expected descriptor dim {self.descriptor_dim}, got {rst_descriptors.size(2)}"
            )

        patch_height, patch_width = rst_maps.shape[-2:]
        mosaic = self.build_mosaic(rst_maps)
        boundary_features = self.cross_boundary(mosaic)

        token_features = self.token_pool(boundary_features)
        tokens = token_features.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens + self.position_embedding)
        token_features = tokens.transpose(1, 2).reshape(
            rst_maps.size(0),
            self.feature_dim,
            self.token_grid_size,
            self.token_grid_size,
        )
        token_features = F.interpolate(
            token_features,
            size=boundary_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        global_map = boundary_features + token_features

        quadrant_maps = self.split_mosaic(global_map, patch_height, patch_width)
        quadrant_context = quadrant_maps.mean(dim=(-2, -1))
        global_context = global_map.mean(dim=(-2, -1)).unsqueeze(1).expand(-1, 4, -1)
        context_descriptors = F.normalize(
            self.context_projection(
                torch.cat((quadrant_context, global_context), dim=-1)
            ),
            p=2,
            dim=-1,
        )

        gate_values = torch.sigmoid(
            self.gate(torch.cat((rst_descriptors, context_descriptors), dim=-1))
        )
        fused_descriptors = F.normalize(
            rst_descriptors + gate_values * context_descriptors,
            p=2,
            dim=-1,
        )
        if return_aux:
            return fused_descriptors, {
                'original_descriptors': rst_descriptors,
                'context_descriptors': context_descriptors,
                'gate_values': gate_values,
            }
        return fused_descriptors


class LocalCandidatePositionRefiner(nn.Module):
    """Refine a coarse RSB position with local RST-to-UAV feature matching."""

    def __init__(
        self,
        input_dim=256,
        feature_dim=64,
        radius=0.15,
        temperature=0.1,
        gate_bias=-2.0,
    ):
        super().__init__()
        if radius <= 0:
            raise ValueError("LCPR radius must be positive")
        if temperature <= 0:
            raise ValueError("LCPR temperature must be positive")

        self.radius = float(radius)
        self.temperature = float(temperature)
        self.rst_projection = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )
        self.uav_projection = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

        axis = torch.tensor((-1.0, 0.0, 1.0))
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        offsets = torch.stack((xx, yy), dim=-1).reshape(-1, 2) * self.radius
        self.register_buffer('candidate_offsets', offsets)

    def forward(self, rst_maps, uav_map, coarse_position):
        if rst_maps.dim() != 5 or rst_maps.size(1) != 4:
            raise ValueError("LCPR requires four RST feature maps")
        if uav_map.dim() != 4 or coarse_position.shape != (rst_maps.size(0), 2):
            raise ValueError("Invalid UAV map or coarse position shape for LCPR")

        rst_mosaic = RSTGlobalContextFusion.build_mosaic(rst_maps)
        rst_feature = self.rst_projection(rst_mosaic)
        uav_query = F.adaptive_avg_pool2d(
            self.uav_projection(uav_map), 1
        ).flatten(1)

        offsets = self.candidate_offsets.to(
            device=coarse_position.device, dtype=coarse_position.dtype
        )
        candidate_positions = (
            coarse_position[:, None, :] + offsets[None, :, :]
        ).clamp(-1.0, 1.0)
        effective_offsets = candidate_positions - coarse_position[:, None, :]

        # The 2x2 RST mosaic spans twice the normalized target search region.
        sampling_grid = (candidate_positions / 2.0).unsqueeze(2)
        candidate_features = F.grid_sample(
            rst_feature,
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        ).squeeze(-1).transpose(1, 2)

        normalized_query = F.normalize(uav_query, p=2, dim=-1, eps=1e-6)
        normalized_candidates = F.normalize(
            candidate_features, p=2, dim=-1, eps=1e-6
        )
        candidate_logits = torch.einsum(
            'bd,bnd->bn', normalized_query, normalized_candidates
        ) / self.temperature
        candidate_weights = torch.softmax(candidate_logits, dim=-1)
        position_delta = torch.einsum(
            'bn,bnd->bd', candidate_weights, effective_offsets
        )
        matched_feature = torch.einsum(
            'bn,bnd->bd', candidate_weights, candidate_features
        )

        gate_input = torch.cat(
            (uav_query, matched_feature, coarse_position, position_delta), dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input))
        refined_position = (
            coarse_position + gate * position_delta
        ).clamp(-1.0, 1.0)
        return refined_position, {
            'coarse_position': coarse_position,
            'candidate_positions': candidate_positions,
            'candidate_weights': candidate_weights,
            'position_delta': position_delta,
            'refinement_gate': gate,
        }


class MultiScaleRotationMarginalizedPositionHead(nn.Module):
    """Dense RSB localization with rotation-marginalized multi-scale matching."""

    def __init__(
        self,
        input_dim=256,
        feature_dim=64,
        num_rotations=16,
        output_size=32,
        scale_sizes=(16, 24, 32),
        template_sizes=(6, 8, 10),
        rotation_temperature=0.15,
        heatmap_temperature=0.10,
        gate_bias=-2.0,
        refine_radius=0.125,
    ):
        super().__init__()
        if num_rotations < 1 or output_size < 2:
            raise ValueError("Rotation count and output size must be positive")
        if len(scale_sizes) != len(template_sizes) or not scale_sizes:
            raise ValueError("Each dense matching scale requires a template size")
        if any(k >= size for size, k in zip(scale_sizes, template_sizes)):
            raise ValueError("Template sizes must be smaller than RST scale sizes")
        if rotation_temperature <= 0 or heatmap_temperature <= 0:
            raise ValueError("Dense position temperatures must be positive")

        self.num_rotations = int(num_rotations)
        self.output_size = int(output_size)
        self.scale_sizes = tuple(int(size) for size in scale_sizes)
        self.template_sizes = tuple(int(size) for size in template_sizes)
        self.rotation_temperature = float(rotation_temperature)
        self.heatmap_temperature = float(heatmap_temperature)
        self.refine_radius = float(refine_radius)

        self.rst_projector = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )
        self.uav_projector = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scale_sizes)))
        quality_dim = 4
        self.refiner = nn.Sequential(
            nn.Linear(feature_dim * 3 + quality_dim + 2, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        self.gate = nn.Sequential(
            nn.Linear(quality_dim + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

        angles = torch.arange(self.num_rotations, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / self.num_rotations)
        self.register_buffer('rotation_angles', angles)
        coords = torch.linspace(-1.0, 1.0, self.output_size)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('position_grid', torch.stack((xx, yy), dim=-1))

    @staticmethod
    def _build_mosaic(rst_maps):
        return RSTGlobalContextFusion.build_mosaic(rst_maps)

    def _rotation_bank(self, feature, template_size):
        feature = F.adaptive_avg_pool2d(feature, (template_size, template_size))
        batch_size = feature.size(0)
        angles = self.rotation_angles.to(device=feature.device, dtype=feature.dtype)
        cosine, sine = angles.cos(), angles.sin()
        theta = torch.zeros(
            self.num_rotations, 2, 3, device=feature.device, dtype=feature.dtype
        )
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta = theta.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, 2, 3)
        expanded = feature[:, None].expand(
            -1, self.num_rotations, -1, -1, -1
        ).reshape(-1, feature.size(1), template_size, template_size)
        grid = F.affine_grid(theta, expanded.shape, align_corners=False)
        rotated = F.grid_sample(
            expanded, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        return rotated.reshape(
            batch_size, self.num_rotations, feature.size(1), template_size, template_size
        )

    def _align_response(self, response, map_size, template_size):
        """Resample valid correlation centers onto the common [-1, 1] RSB grid."""
        response_size = map_size - template_size + 1
        target = self.position_grid.to(device=response.device, dtype=response.dtype)
        mosaic_center = target / 2.0
        pixel_center = (mosaic_center + 1.0) * map_size / 2.0
        response_index = pixel_center - template_size / 2.0
        source_grid = (response_index + 0.5) * 2.0 / response_size - 1.0
        source_grid = source_grid.unsqueeze(0).expand(response.size(0), -1, -1, -1)
        aligned = F.grid_sample(
            response, source_grid, mode='bilinear', padding_mode='border',
            align_corners=False,
        )
        return aligned

    def _correlate_scale(self, rst_feature, uav_feature, map_size, template_size):
        rst_scaled = F.interpolate(
            rst_feature, size=(map_size, map_size), mode='bilinear', align_corners=False
        )
        templates = self._rotation_bank(uav_feature, template_size)
        batch_size, rotations = templates.shape[:2]
        patches = F.unfold(rst_scaled, kernel_size=template_size)
        patches = F.normalize(patches, p=2, dim=1, eps=1e-6)
        templates = templates.flatten(2)
        templates = F.normalize(templates, p=2, dim=-1, eps=1e-6)
        scores = torch.einsum('brd,bdn->brn', templates, patches)
        response_size = map_size - template_size + 1
        scores = scores.reshape(batch_size, rotations, response_size, response_size)
        return self._align_response(scores, map_size, template_size)

    def _soft_argmax(self, probability):
        grid = self.position_grid.to(device=probability.device, dtype=probability.dtype)
        return torch.einsum('bhw,hwc->bc', probability, grid)

    def forward(self, rst_maps, uav_map, base_position):
        if rst_maps.dim() != 5 or rst_maps.size(1) != 4:
            raise ValueError("Dense positioning requires four RST feature maps")
        rst_feature = self.rst_projector(self._build_mosaic(rst_maps))
        uav_feature = self.uav_projector(uav_map)
        scale_volumes = [
            self._correlate_scale(rst_feature, uav_feature, size, template)
            for size, template in zip(self.scale_sizes, self.template_sizes)
        ]
        scale_volume = torch.stack(scale_volumes, dim=1)
        scale_weights = torch.softmax(self.scale_logits, dim=0)
        volume = torch.einsum('s,bsrhw->brhw', scale_weights, scale_volume)
        position_logits = self.rotation_temperature * torch.logsumexp(
            volume / self.rotation_temperature, dim=1
        )
        probability = torch.softmax(
            position_logits.flatten(1) / self.heatmap_temperature, dim=-1
        ).reshape_as(position_logits)
        coarse_position = self._soft_argmax(probability)

        flat_probability = probability.flatten(1)
        top_values = flat_probability.topk(k=2, dim=-1).values
        entropy = -(
            flat_probability * flat_probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(flat_probability.size(-1))
        scale_positions = []
        for scores in scale_volumes:
            logits = self.rotation_temperature * torch.logsumexp(
                scores / self.rotation_temperature, dim=1
            )
            scale_probability = torch.softmax(
                logits.flatten(1) / self.heatmap_temperature, dim=-1
            ).reshape_as(logits)
            scale_positions.append(self._soft_argmax(scale_probability))
        scale_positions = torch.stack(scale_positions, dim=1)
        scale_spread = scale_positions.std(dim=1, unbiased=False).norm(dim=-1)
        quality = torch.stack(
            (top_values[:, 0], top_values[:, 0] - top_values[:, 1],
             1.0 - entropy, scale_spread), dim=-1
        )

        grid_center = (coarse_position / 2.0).view(-1, 1, 1, 2)
        sampled_rst = F.grid_sample(
            rst_feature, grid_center, mode='bilinear', padding_mode='border',
            align_corners=False,
        ).flatten(1)
        uav_summary = F.adaptive_avg_pool2d(uav_feature, 1).flatten(1)
        feature_delta = sampled_rst - uav_summary
        refine_input = torch.cat(
            (sampled_rst, uav_summary, feature_delta, quality, coarse_position), dim=-1
        )
        offset = torch.tanh(self.refiner(refine_input)) * self.refine_radius
        dense_position = (coarse_position + offset).clamp(-1.0, 1.0)
        gate_input = torch.cat((quality, dense_position - base_position.detach()), dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        final_position = base_position + gate * (dense_position - base_position)
        return final_position, {
            'position_logits': position_logits,
            'position_probability': probability,
            'coarse_position': coarse_position,
            'dense_position': dense_position,
            'position_gate': gate,
            'scale_weights': scale_weights,
            'scale_positions': scale_positions,
        }


class PositionConditionedPolarSampler(nn.Module):
    """Sample multi-scale polar RST features around a predicted RSB position."""

    def __init__(
        self,
        num_radial_bins=8,
        num_angle_bins=72,
        crop_scales=(0.75, 1.0, 1.25),
        base_radius=0.5,
    ):
        super().__init__()
        if num_radial_bins <= 0 or num_angle_bins <= 1:
            raise ValueError("Polar bin counts must be positive")
        if base_radius <= 0:
            raise ValueError("base_radius must be positive")
        if not crop_scales or any(scale <= 0 for scale in crop_scales):
            raise ValueError("crop_scales must contain positive values")

        self.num_radial_bins = int(num_radial_bins)
        self.num_angle_bins = int(num_angle_bins)
        self.base_radius = float(base_radius)
        self.register_buffer(
            'crop_scales', torch.tensor(crop_scales, dtype=torch.float32)
        )
        angles = torch.arange(num_angle_bins, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / num_angle_bins)
        self.register_buffer(
            'angle_basis', torch.stack((angles.cos(), angles.sin()), dim=-1)
        )

    @staticmethod
    def build_mosaic(rst_maps):
        return RSTGlobalContextFusion.build_mosaic(rst_maps)

    @staticmethod
    def position_to_grid_center(position):
        """Convert RSB x/y units to normalized 2x2-mosaic grid coordinates."""
        if position.dim() != 2 or position.size(-1) != 2:
            raise ValueError(
                f"Expected positions shaped [B, 2], got {tuple(position.shape)}"
            )
        # Metadata uses 128 px per coordinate unit. The complete RSB is
        # 512x512, so its grid_sample half-extent (256 px) equals two units.
        return position / 2.0

    def forward(self, rst_mosaic, position):
        if rst_mosaic.dim() != 4:
            raise ValueError(
                f"Expected mosaic shaped [B, C, H, W], got {tuple(rst_mosaic.shape)}"
            )
        if rst_mosaic.size(0) != position.size(0):
            raise ValueError("Mosaic and position batch sizes must match")

        batch_size, channels = rst_mosaic.shape[:2]
        dtype = rst_mosaic.dtype
        device = rst_mosaic.device
        center = self.position_to_grid_center(position).to(dtype=dtype)
        scales = self.crop_scales.to(device=device, dtype=dtype)
        angle_basis = self.angle_basis.to(device=device, dtype=dtype)
        radius = (
            torch.arange(self.num_radial_bins, device=device, dtype=dtype) + 0.5
        ) / self.num_radial_bins
        radius = radius * self.base_radius

        offsets = (
            scales[:, None, None, None]
            * radius[None, :, None, None]
            * angle_basis[None, None, :, :]
        )
        grid = center[:, None, None, None, :] + offsets[None, :, :, :, :]
        num_scales = scales.numel()
        sample_grid = grid.reshape(
            batch_size * num_scales,
            self.num_radial_bins,
            self.num_angle_bins,
            2,
        )
        expanded_mosaic = rst_mosaic[:, None].expand(
            -1, num_scales, -1, -1, -1
        ).reshape(
            batch_size * num_scales,
            channels,
            rst_mosaic.size(-2),
            rst_mosaic.size(-1),
        )
        sampled = F.grid_sample(
            expanded_mosaic,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        return sampled.reshape(
            batch_size,
            num_scales,
            channels,
            self.num_radial_bins,
            self.num_angle_bins,
        )


class UAVDirectionQueryEncoder(nn.Module):
    """Learn UAV direction tokens without imposing a planar polar warp."""

    def __init__(
        self,
        feature_dim=64,
        num_radial_bins=8,
        num_angle_bins=72,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.num_radial_bins = num_radial_bins
        self.num_angle_bins = num_angle_bins
        self.radial_queries = nn.Parameter(
            torch.zeros(1, num_radial_bins, 1, feature_dim)
        )
        angles = torch.arange(num_angle_bins, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / num_angle_bins)
        self.register_buffer(
            'angle_basis', torch.stack((angles.cos(), angles.sin()), dim=-1)
        )
        self.angle_projection = nn.Linear(2, feature_dim)
        self.spatial_projection = nn.Linear(2, feature_dim)
        self.query_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        nn.init.trunc_normal_(self.radial_queries, std=0.02)

    @staticmethod
    def _normalized_coordinates(height, width, device, dtype):
        rows = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        cols = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
        return torch.stack((row_grid, col_grid), dim=-1).reshape(-1, 2)

    def forward(self, uav_map):
        if uav_map.dim() != 4 or uav_map.size(1) != self.feature_dim:
            raise ValueError(
                "Expected UAV map shaped [B, feature_dim, H, W], "
                f"got {tuple(uav_map.shape)}"
            )
        batch_size, _, height, width = uav_map.shape
        memory = uav_map.flatten(2).transpose(1, 2)
        coordinates = self._normalized_coordinates(
            height, width, uav_map.device, uav_map.dtype
        )
        memory = memory + self.spatial_projection(coordinates).unsqueeze(0)

        angle_queries = self.angle_projection(
            self.angle_basis.to(device=uav_map.device, dtype=uav_map.dtype)
        ).view(1, 1, self.num_angle_bins, self.feature_dim)
        queries = self.radial_queries.to(dtype=uav_map.dtype) + angle_queries
        queries = queries.reshape(
            1, self.num_radial_bins * self.num_angle_bins, self.feature_dim
        ).expand(batch_size, -1, -1)
        encoded, _ = self.query_attention(
            queries, memory, memory, need_weights=False
        )
        encoded = self.output_norm(encoded + queries)
        return encoded.reshape(
            batch_size,
            self.num_radial_bins,
            self.num_angle_bins,
            self.feature_dim,
        ).permute(0, 3, 1, 2).contiguous()


class SharedRadialAlignment(nn.Module):
    """Map UAV-height and RST-radius features to a shared learned radial axis."""

    def __init__(self, feature_dim=64, num_radial_bins=8, num_heads=4, dropout=0.1):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.num_radial_bins = num_radial_bins
        self.virtual_queries = nn.Parameter(
            torch.zeros(1, num_radial_bins, feature_dim)
        )
        self.radial_embedding = nn.Parameter(
            torch.zeros(1, num_radial_bins, feature_dim)
        )
        self.uav_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.rst_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.uav_norm = nn.LayerNorm(feature_dim)
        self.rst_norm = nn.LayerNorm(feature_dim)
        nn.init.trunc_normal_(self.virtual_queries, std=0.02)
        nn.init.trunc_normal_(self.radial_embedding, std=0.02)

    def _align(self, feature, attention, output_norm):
        batch_size, channels, radial_bins, angle_bins = feature.shape
        if channels != self.feature_dim or radial_bins != self.num_radial_bins:
            raise ValueError("Unexpected radial alignment feature shape")
        memory = feature.permute(0, 3, 2, 1).reshape(
            batch_size * angle_bins, radial_bins, channels
        )
        memory = memory + self.radial_embedding.to(dtype=feature.dtype)
        queries = self.virtual_queries.to(dtype=feature.dtype).expand(
            batch_size * angle_bins, -1, -1
        )
        aligned, _ = attention(queries, memory, memory, need_weights=False)
        aligned = output_norm(aligned + queries)
        return aligned.reshape(
            batch_size, angle_bins, radial_bins, channels
        ).permute(0, 3, 2, 1).contiguous()

    def forward(self, uav_feature, rst_features):
        aligned_uav = self._align(
            uav_feature, self.uav_attention, self.uav_norm
        )
        batch_size, num_scales, channels, radial_bins, angle_bins = (
            rst_features.shape
        )
        flat_rst = rst_features.reshape(
            batch_size * num_scales, channels, radial_bins, angle_bins
        )
        aligned_rst = self._align(
            flat_rst, self.rst_attention, self.rst_norm
        ).reshape(
            batch_size, num_scales, channels, radial_bins, angle_bins
        )
        return aligned_uav, aligned_rst


class CyclicOrientationCorrelation(nn.Module):
    """Build a full circular heading distribution from aligned view features."""

    def __init__(self, num_angle_bins=72, orientation_temperature=0.1,
                 scale_temperature=0.2):
        super().__init__()
        if orientation_temperature <= 0 or scale_temperature <= 0:
            raise ValueError("Correlation temperatures must be positive")
        self.num_angle_bins = num_angle_bins
        self.orientation_temperature = orientation_temperature
        self.scale_temperature = scale_temperature

    def forward(self, uav_feature, rst_features):
        if rst_features.size(-1) != self.num_angle_bins:
            raise ValueError("RST angle dimension does not match num_angle_bins")
        uav_feature = F.normalize(uav_feature, p=2, dim=1)
        rst_features = F.normalize(rst_features, p=2, dim=2)
        scores = []
        for shift in range(self.num_angle_bins):
            # Shift the UAV-relative direction sequence so logit k directly
            # represents the candidate absolute heading theta_k.
            shifted_uav = torch.roll(uav_feature, shifts=shift, dims=-1)
            score = (
                shifted_uav[:, None] * rst_features
            ).sum(dim=2).mean(dim=(-2, -1))
            scores.append(score)
        scale_scores = torch.stack(scores, dim=-1)
        logits = self.scale_temperature * torch.logsumexp(
            scale_scores / self.scale_temperature, dim=1
        )
        probability = torch.softmax(
            logits / self.orientation_temperature, dim=-1
        )
        return logits, probability, scale_scores


class MSPCOCHeadingHead(nn.Module):
    """Multi-scale position-conditioned orientation correlation head."""

    def __init__(
        self,
        input_dim=256,
        feature_dim=64,
        num_radial_bins=8,
        num_angle_bins=72,
        crop_scales=(0.75, 1.0, 1.25),
        base_radius=0.5,
        num_heads=4,
        dropout=0.1,
        orientation_temperature=0.1,
        scale_temperature=0.2,
        gate_bias=-3.0,
    ):
        super().__init__()
        self.num_angle_bins = num_angle_bins
        projector = lambda: nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
        )
        self.rst_projector = projector()
        self.uav_projector = projector()
        self.polar_sampler = PositionConditionedPolarSampler(
            num_radial_bins=num_radial_bins,
            num_angle_bins=num_angle_bins,
            crop_scales=crop_scales,
            base_radius=base_radius,
        )
        self.uav_direction_encoder = UAVDirectionQueryEncoder(
            feature_dim=feature_dim,
            num_radial_bins=num_radial_bins,
            num_angle_bins=num_angle_bins,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.radial_alignment = SharedRadialAlignment(
            feature_dim=feature_dim,
            num_radial_bins=num_radial_bins,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.correlation = CyclicOrientationCorrelation(
            num_angle_bins=num_angle_bins,
            orientation_temperature=orientation_temperature,
            scale_temperature=scale_temperature,
        )
        angles = torch.arange(num_angle_bins, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / num_angle_bins)
        self.register_buffer(
            'heading_basis', torch.stack((angles.cos(), angles.sin()), dim=-1)
        )
        self.confidence_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.confidence_gate[-1].weight)
        nn.init.constant_(self.confidence_gate[-1].bias, gate_bias)

    def forward(self, rst_maps, uav_map, position, base_heading):
        batch_size, num_rst, channels, height, width = rst_maps.shape
        if num_rst != 4:
            raise ValueError("MS-PCOC requires exactly four RST maps")
        projected_rst = self.rst_projector(
            rst_maps.reshape(batch_size * num_rst, channels, height, width)
        ).reshape(batch_size, num_rst, -1, height, width)
        rst_mosaic = PositionConditionedPolarSampler.build_mosaic(projected_rst)
        rst_polar = self.polar_sampler(rst_mosaic, position.detach())
        projected_uav = self.uav_projector(uav_map)
        uav_direction = self.uav_direction_encoder(projected_uav)
        aligned_uav, aligned_rst = self.radial_alignment(
            uav_direction, rst_polar
        )
        logits, probability, scale_scores = self.correlation(
            aligned_uav, aligned_rst
        )

        basis = self.heading_basis.to(
            device=probability.device, dtype=probability.dtype
        )
        raw_heading = probability @ basis
        resultant = torch.sqrt(
            raw_heading.square().sum(dim=-1) + 1e-12
        )
        corr_heading = raw_heading / resultant.clamp_min(1e-6).unsqueeze(-1)
        max_probability = probability.max(dim=-1).values
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(self.num_angle_bins)
        confidence = torch.stack(
            (max_probability, 1.0 - entropy, resultant), dim=-1
        )
        gate = torch.sigmoid(self.confidence_gate(confidence))
        base_heading = F.normalize(base_heading, p=2, dim=-1, eps=1e-6)
        final_heading = F.normalize(
            (1.0 - gate) * base_heading + gate * corr_heading,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        return final_heading, {
            'orientation_logits': logits,
            'orientation_probability': probability,
            'scale_scores': scale_scores,
            'correlation_heading': corr_heading,
            'base_heading': base_heading,
            'final_heading': final_heading,
            'gate': gate,
            'resultant': resultant,
            'rst_polar': rst_polar,
            'uav_direction': uav_direction,
        }


class GeometryPoseVolumeHeadingHead(nn.Module):
    """Geometry-preserving joint translation-orientation heading head."""

    def __init__(
        self,
        input_dim=256,
        feature_dim=64,
        spatial_size=8,
        candidate_grid_size=9,
        candidate_radius=0.5,
        num_angle_bins=36,
        crop_half_extent=0.5,
        orientation_temperature=0.1,
        translation_temperature=0.2,
        residual_limit_degrees=5.0,
    ):
        super().__init__()
        if candidate_grid_size < 1 or candidate_grid_size % 2 == 0:
            raise ValueError("candidate_grid_size must be a positive odd number")
        if num_angle_bins < 4 or num_angle_bins % 2 != 0:
            raise ValueError("num_angle_bins must be even and at least four")
        if min(
            feature_dim,
            spatial_size,
            candidate_radius,
            crop_half_extent,
            orientation_temperature,
            translation_temperature,
        ) <= 0:
            raise ValueError("GPRV-H dimensions, radii, and temperatures must be positive")

        self.feature_dim = int(feature_dim)
        self.spatial_size = int(spatial_size)
        self.candidate_grid_size = int(candidate_grid_size)
        self.num_angle_bins = int(num_angle_bins)
        self.crop_half_extent = float(crop_half_extent)
        self.orientation_temperature = float(orientation_temperature)
        self.translation_temperature = float(translation_temperature)
        self.residual_limit = math.radians(float(residual_limit_degrees))

        def projector():
            return nn.Sequential(
                nn.Conv2d(input_dim, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, feature_dim),
                nn.GELU(),
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size=3,
                    padding=1,
                    groups=feature_dim,
                    bias=False,
                ),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, feature_dim),
                nn.GELU(),
            )

        self.rst_projector = projector()
        self.uav_projector = projector()
        self.uav_confidence = nn.Conv2d(feature_dim, 1, kernel_size=1)
        self.geometry_bias = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        phase_input_dim = 2 * feature_dim + 5
        self.phase_refiner = nn.Sequential(
            nn.Linear(phase_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        axis = torch.linspace(
            -float(candidate_radius),
            float(candidate_radius),
            candidate_grid_size,
        )
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer(
            'candidate_offsets',
            torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2),
        )
        angles = torch.arange(num_angle_bins, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / num_angle_bins)
        self.register_buffer('angles', angles)
        self.register_buffer(
            'heading_basis', torch.stack((angles.cos(), angles.sin()), dim=-1)
        )

    @staticmethod
    def build_mosaic(rst_maps):
        return RSTGlobalContextFusion.build_mosaic(rst_maps)

    @staticmethod
    def position_to_grid_center(position):
        return PositionConditionedPolarSampler.position_to_grid_center(position)

    @staticmethod
    def _rotate_heading(heading, angle):
        cosine = angle.cos()
        sine = angle.sin()
        x_coord, y_coord = heading.unbind(dim=-1)
        return torch.stack(
            (cosine * x_coord - sine * y_coord,
             sine * x_coord + cosine * y_coord),
            dim=-1,
        )

    def _rotation_bank(self, feature, confidence=None):
        batch_size = feature.size(0)
        angles = self.angles.to(device=feature.device, dtype=feature.dtype)
        cosine = angles.cos()
        sine = angles.sin()
        matrices = torch.zeros(
            self.num_angle_bins, 2, 3, device=feature.device, dtype=feature.dtype
        )
        # affine_grid maps output coordinates to input coordinates. This sign
        # makes bin k align a UAV feature rotated by -theta_k with the RST map.
        matrices[:, 0, 0] = cosine
        matrices[:, 0, 1] = -sine
        matrices[:, 1, 0] = sine
        matrices[:, 1, 1] = cosine
        matrices = matrices.unsqueeze(0).expand(batch_size, -1, -1, -1)
        matrices = matrices.reshape(batch_size * self.num_angle_bins, 2, 3)
        expanded = feature[:, None].expand(
            -1, self.num_angle_bins, -1, -1, -1
        ).reshape(
            batch_size * self.num_angle_bins,
            feature.size(1),
            feature.size(2),
            feature.size(3),
        )
        grid = F.affine_grid(matrices, expanded.shape, align_corners=False)
        rotated = F.grid_sample(
            expanded,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        ).reshape(
            batch_size,
            self.num_angle_bins,
            feature.size(1),
            feature.size(2),
            feature.size(3),
        )
        if confidence is None:
            return rotated
        expanded_confidence = confidence[:, None].expand(
            -1, self.num_angle_bins, -1, -1, -1
        ).reshape(
            batch_size * self.num_angle_bins,
            1,
            confidence.size(2),
            confidence.size(3),
        )
        rotated_confidence = F.grid_sample(
            expanded_confidence,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        ).reshape(
            batch_size,
            self.num_angle_bins,
            confidence.size(1),
            confidence.size(2),
            confidence.size(3),
        )
        return rotated, rotated_confidence

    def _candidate_crops(self, mosaic, position):
        batch_size, channels = mosaic.shape[:2]
        center = self.position_to_grid_center(position).to(dtype=mosaic.dtype)
        offsets = self.candidate_offsets.to(
            device=mosaic.device, dtype=mosaic.dtype
        )
        candidate_centers = center[:, None] + offsets[None] / 2.0

        sample_axis = (
            (torch.arange(self.spatial_size, device=mosaic.device, dtype=mosaic.dtype)
             + 0.5)
            / self.spatial_size
            * 2.0
            - 1.0
        ) * self.crop_half_extent
        sample_y, sample_x = torch.meshgrid(sample_axis, sample_axis, indexing='ij')
        local_grid = torch.stack((sample_x, sample_y), dim=-1)
        grid = candidate_centers[:, :, None, None, :] + local_grid[None, None]
        num_candidates = offsets.size(0)
        expanded = mosaic[:, None].expand(
            -1, num_candidates, -1, -1, -1
        ).reshape(
            batch_size * num_candidates,
            channels,
            mosaic.size(-2),
            mosaic.size(-1),
        )
        crops = F.grid_sample(
            expanded,
            grid.reshape(
                batch_size * num_candidates,
                self.spatial_size,
                self.spatial_size,
                2,
            ),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        return crops.reshape(
            batch_size,
            num_candidates,
            channels,
            self.spatial_size,
            self.spatial_size,
        )

    def _pose_volume(self, rst_mosaic, uav_feature, position):
        crops = self._candidate_crops(rst_mosaic, position)
        confidence = torch.sigmoid(self.uav_confidence(uav_feature))
        rotated_uav, rotated_confidence = self._rotation_bank(
            uav_feature, confidence
        )
        rotated_uav = F.normalize(rotated_uav, p=2, dim=2, eps=1e-6)
        crops = F.normalize(crops, p=2, dim=2, eps=1e-6)
        weights = rotated_confidence.squeeze(2)
        scores = torch.einsum(
            'bkchw,bnchw,bkhw->bnk', rotated_uav, crops, weights
        ) / weights.sum(dim=(-2, -1)).clamp_min(1e-6)[:, None, :]

        offsets = self.candidate_offsets.to(
            device=scores.device, dtype=scores.dtype
        )
        angles = self.angles.to(device=scores.device, dtype=scores.dtype)
        geometry = torch.cat(
            (
                offsets[:, None].expand(-1, self.num_angle_bins, -1),
                angles.cos()[None, :, None].expand(offsets.size(0), -1, -1),
                angles.sin()[None, :, None].expand(offsets.size(0), -1, -1),
            ),
            dim=-1,
        )
        scores = scores + self.geometry_bias(geometry).squeeze(-1).unsqueeze(0)
        return scores, crops, uav_feature

    def forward(self, rst_maps, uav_map, position, base_heading=None):
        batch_size, num_rst, channels, height, width = rst_maps.shape
        if num_rst != 4:
            raise ValueError("GPRV-H requires exactly four RST feature maps")
        projected_rst = self.rst_projector(
            rst_maps.reshape(batch_size * num_rst, channels, height, width)
        )
        projected_rst = F.adaptive_avg_pool2d(
            projected_rst, (self.spatial_size, self.spatial_size)
        ).reshape(
            batch_size,
            num_rst,
            self.feature_dim,
            self.spatial_size,
            self.spatial_size,
        )
        projected_uav = F.adaptive_avg_pool2d(
            self.uav_projector(uav_map),
            (self.spatial_size, self.spatial_size),
        )
        rst_mosaic = self.build_mosaic(projected_rst)
        volume_logits, crops, uav_feature = self._pose_volume(
            rst_mosaic, projected_uav, position.detach()
        )

        orientation_logits = self.translation_temperature * torch.logsumexp(
            volume_logits / self.translation_temperature, dim=1
        )
        orientation_probability = torch.softmax(
            orientation_logits / self.orientation_temperature, dim=-1
        )
        basis = self.heading_basis.to(
            device=orientation_probability.device,
            dtype=orientation_probability.dtype,
        )
        raw_heading = orientation_probability @ basis
        resultant = torch.sqrt(
            raw_heading.square().sum(dim=-1) + 1e-12
        )
        coarse_heading = raw_heading / resultant.clamp_min(1e-6).unsqueeze(-1)

        position_logits = torch.logsumexp(volume_logits, dim=-1)
        position_probability = torch.softmax(position_logits, dim=-1)
        rst_summary = torch.einsum(
            'bn,bnc->bc', position_probability, crops.mean(dim=(-2, -1))
        )
        uav_summary = uav_feature.mean(dim=(-2, -1))
        max_probability = orientation_probability.max(dim=-1).values
        entropy = -(
            orientation_probability
            * orientation_probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(self.num_angle_bins)
        phase_input = torch.cat(
            (
                uav_summary,
                rst_summary,
                coarse_heading,
                max_probability.unsqueeze(-1),
                (1.0 - entropy).unsqueeze(-1),
                resultant.unsqueeze(-1),
            ),
            dim=-1,
        )
        phase_output = self.phase_refiner(phase_input)
        residual = torch.tanh(phase_output[:, 0]) * self.residual_limit
        final_heading = F.normalize(
            self._rotate_heading(coarse_heading, residual),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        phase2_heading = F.normalize(
            phase_output[:, 1:3], p=2, dim=-1, eps=1e-6
        )
        volume_probability = torch.softmax(
            volume_logits.flatten(1) / self.orientation_temperature, dim=-1
        ).reshape_as(volume_logits)
        return final_heading, {
            'pose_volume_logits': volume_logits,
            'pose_volume_probability': volume_probability,
            'orientation_logits': orientation_logits,
            'orientation_probability': orientation_probability,
            'position_probability': position_probability,
            'coarse_heading': coarse_heading,
            'final_heading': final_heading,
            'phase2_heading': phase2_heading,
            'phase_residual': residual,
            'resultant': resultant,
            'candidate_offsets': self.candidate_offsets,
        }

class PositionAngleRegressionSGM(nn.Module):
    def __init__(self, 
                 feature_dim=256, 
                 coord_enc_dims=[64, 256],  # Multiply by n_cluster of csmg for each layer
                 regressor_dims=[256, 64],  # Regression scale
                 reduction_ratio=1,
                 backbone_name='vgg16',
                 num_clusters=4,
                 freeze_backbone=True,
                 partial_unfreeze=False,
                 add_patch_coord=True
                 ):
        """
        PositionAngleRegressionSGM is based on PositionAngleRegressionModel with csmg replacement.
        'par_ca_sgm':  # Replace feat-extract of par_ca with csmg feature extractor of sgm
        """
        super().__init__()
        
        # Save config params as instance attributes
        self.model_name = 'par_ca_sgm'
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters  # Number of scene graph clusters
        self.coord_enc_dims = [x*num_clusters for x in coord_enc_dims]  # Dimension sequence of coordinate encoding
        self.regressor_dims = regressor_dims
        self.reduction_ratio = reduction_ratio
        self.is_pretrained = True
        self.freeze_backbone = freeze_backbone
        self.partial_unfreeze = partial_unfreeze
        self.add_patch_coord = add_patch_coord  # Whether to add neighborhood coordinates
        
        # Backbone config
        if backbone_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=self.is_pretrained)
            # VGG16 features include 13 conv layers and 5 pooling layers
            # Truncate to first 18 layers (remove last 8 layers), output feature map size 28x28, channels 512
            layers = list(self.backbone.features.children())[:-8]
            self.backbone = nn.Sequential(*layers)
            self.backbone_out_dim = 512  # Output channels of VGG16
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=self.is_pretrained)
            layers = list(backbone.children())[:-2]
            self.backbone = nn.Sequential(*layers)
            self.backbone_out_dim = 512  # Output channels of ResNet50
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=self.is_pretrained)
            # ResNet50 features, output feature map size 7x7, channels 2048
            layers = list(backbone.children())[:-2]
            self.backbone = nn.Sequential(*layers)
            self.backbone_out_dim = 2048  # Output channels of ResNet50
        
        # === Freeze or partially unfreeze backbone params ===
        if self.freeze_backbone:
            if self.partial_unfreeze:
                if self.backbone_name == 'vgg16':
                    # Unfreeze last several layers of VGG16 (e.g., 10~17)
                    for name, module in self.backbone.named_children():
                        if int(name) >= 14:
                            for p in module.parameters():
                                p.requires_grad = True
                        else:
                            for p in module.parameters():
                                p.requires_grad = False

                elif self.backbone_name.startswith('resnet'):
                    # ResNet unfreeze layer4 (or layer3 + layer4)
                    for name, module in self.backbone.named_children():
                        if any(k in name for k in ['layer4']):
                            for p in module.parameters():
                                p.requires_grad = True
                        else:
                            for p in module.parameters():
                                p.requires_grad = False
            else:
                # Fully freeze
                for p in self.backbone.parameters():
                    p.requires_grad = False

        # Scene graph encoding module
        # Input dim: backbone output dim, Output dim: num_clusters*feature_dim
        self.csmg = CSMG(input_channel=self.backbone_out_dim, output_channel=self.feature_dim, num_clusters=num_clusters)

        # Coordinate encoder
        coord_enc_layers = []
        coord_in_dim = 2  # Input coordinate dim
        # coord_out_dim = self.feature_dim * num_clusters
        for out_dim in self.coord_enc_dims:
            coord_enc_layers.append(nn.Linear(coord_in_dim, out_dim))
            coord_enc_layers.append(nn.ReLU())
            coord_in_dim = out_dim
        # coord_enc_layers.append(nn.Linear(coord_in_dim, coord_out_dim))
        self.coord_encoder = nn.Sequential(*coord_enc_layers)
        
        # Cross attention module
        self.neighbors_cross_attn = NeighborsCrossAttention(
            feat_dim=self.feature_dim*num_clusters, 
            reduction_ratio=reduction_ratio
        )
        
        # Regression head
        self.reg_in_dim = num_clusters * (self.feature_dim + (self.feature_dim // reduction_ratio))

        # Position branch
        #Scheme 1: Initial scheme
        self.pos_in_dim = self.reg_in_dim
        pos_regressor_layers = []
        for out_dim_ in regressor_dims:
            pos_regressor_layers.append(nn.Linear(self.pos_in_dim, out_dim_))
            pos_regressor_layers.append(nn.ReLU())
            self.pos_in_dim = out_dim_
        pos_regressor_layers.append(nn.Linear(self.pos_in_dim, 2))  # Output x, y
        self.pos_regressor = nn.Sequential(*pos_regressor_layers)
        
        # Direction branch
        self.dir_in_dim = self.reg_in_dim
        dir_regressor_layers = []
        for out_dim_ in regressor_dims:
            dir_regressor_layers.append(nn.Linear(self.dir_in_dim, out_dim_))
            dir_regressor_layers.append(nn.ReLU())
            self.dir_in_dim = out_dim_
        dir_regressor_layers.append(nn.Linear(self.dir_in_dim, 2))  # Output cos, sin
        self.dir_regressor = nn.Sequential(*dir_regressor_layers)
    
    def forward(self, patches):
        # patches: [B, 5, C, H, W]
        B = patches.size(0)
        
        # Extract features (shared weights)
        feats = []
        for i in range(5):
            patch = patches[:, i]
            visual_feat = self.backbone(patch)  # torch.Size([32, 512, 32, 32])
            # Pass through scene graph encoding module
            sim_scores, d, d_flatten, _ = self.csmg(visual_feat)  # d_flatten: [B, num_clusters*feature_dim]
            feats.append(d_flatten)

        f1, f2, f3, f4, fi = feats
        # 4 neighborhood coordinates, origin at block center, u as unit length, according to translation image processing coordinate system
        u = 1.0
        known_coords = torch.tensor([[-u,-u], [-u,u], [u,-u], [u,u]],
                                   dtype=torch.float32, device=patches.device) 

        # [4,D] --> [1,4,D] --> [B,4,D]
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).repeat(B,1,1)  # [B,4,D]

        # Enhance neighborhood features
        if self.add_patch_coord:
            neighbor_feats = torch.stack([f1, f2, f3, f4], dim=1) + coord_embs  # [B,4,D]
        else:
            neighbor_feats = torch.stack([f1, f2, f3, f4], dim=1)  # [B,4,D]
        
        # Cross attention
        ctx_feat = self.neighbors_cross_attn(fi, neighbor_feats)  # [B, 32]
        
        # Feature fusion
        combined = torch.cat([fi, ctx_feat], dim=1)  # [B, 160]
        
        # Regression
        pos_pred = self.pos_regressor(combined)
        dir_pred = self.dir_regressor(combined)

        return pos_pred, dir_pred
    
    @classmethod
    def get_model_name(cls):
        """
        # Get via class method
        model_name = PositionAngleRegressionSGM.get_model_name()
        print(model_name)  # Output: par_sgmdca
        """
        # Create temporary instance and return attribute value
        return cls().model_name

    #  The following three member functions are added for debugging
    # Add numerical stability check function
    def _check_tensor_validity(self, tensor, name="tensor"):
        """Check if tensor contains NaN or infinity values"""
        if torch.isnan(tensor).any():
            warnings.warn(f"Warning: {name} contains NaN values")
            return False
        if torch.isinf(tensor).any():
            warnings.warn(f"Warning: {name} contains Inf values")
            return False
        return True   
        
    def _save_forword_tensor(self, dct_content, debug_dir):
        """Save current important variables when position and angle are NaN"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = f'{debug_dir}/debug_por_forward'
        os.makedirs(debug_dir, exist_ok=True)
        temp_path = f'{debug_dir}/por_forward_tensors_{timestamp}.pt'
        torch.save(dct_content, temp_path)  # Save multiple Tensors (dictionary form)
        print(f"🎯Warning: save err info to {temp_path}")

class PARCASGM_v5(PositionAngleRegressionSGM):
    def __init__(self, 
                 backbone_name='vgg16',
                 feature_dim=256, 
                 coord_enc_dims=[16, 64, 256],   # Coordinate encoding layer dimension sequence
                 regressor_dims=[1024, 256, 64],  # Modified to 3-layer structure
                 reduction_ratio=1,
                 num_clusters=4,
                 freeze_backbone=True,
                 partial_unfreeze=False,
                 add_patch_coord=True):
        """
        PARSGM_v4a：-->v5.
        a Try to replace original csmg encoding with Joint (real sgm)
        b Add "feature similarity" to guide regression accuracy,
        c Basic network settings are fixed to deepened state
            coord_enc_dims=[16, 64, 256],   # Coordinate encoding layer dimension sequence
            regressor_dims=[1024, 256, 64],  # Modified to 3-layer structure
        Code organized more concisely
        """
        super().__init__(
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            backbone_name=backbone_name,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord
        )
        
        # Update model name
        self.model_name = 'par_ca_sgm_v5'
        self.sgm = JointNet(None, self.csmg)
        self.sgm.backbone = self.backbone

        # Add auxiliary position guidance module v4a
        self.sim_pos_prior = SimilarityPositionPrior(feat_dim=self.feature_dim)
        self.pos_in_dim = self.reg_in_dim + 2  #Note: 2 nodes added here
        # Position branch v5
        pos_regressor_layers = []
        for out_dim_ in regressor_dims:
            pos_regressor_layers.append(nn.Linear(self.pos_in_dim, out_dim_))
            pos_regressor_layers.append(nn.ReLU())
            self.pos_in_dim = out_dim_
        pos_regressor_layers.append(nn.Linear(self.pos_in_dim, 2))  # Output x, y
        self.pos_regressor = nn.Sequential(*pos_regressor_layers)

    def _model_device(self):
        """Ensure tensors move to the same device as the model parameters."""
        return next(self.parameters()).device

    def _model_dtype(self):
        """Ensure tensors match the model parameter dtype."""
        return next(self.parameters()).dtype

    # Put here: define alias directly in class body after forward_full
    # ===== A: Single patch feature (available for training; no no_grad) =====
    def encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Input: [C,H,W] or [B,C,H,W]
        Output: [B, K*D]
        No patch normalization pipeline here, as it cooperates with data loading class which is responsible for normalization
        """
        # if patch.dim() == 3:
        #     patch = patch.unsqueeze(0)
        # out = self.sgm(patch)
        # Key: move input to same device and dtype as model
        # patch = patch.to(device=self._model_device(), dtype=self._model_dtype(), non_blocking=True)
    
        sgm_output = self.sgm(patch)
        return sgm_output['descriptor_flatten']  # d_flatten,[B, K*D]
    
    def forward(self, patches, debug_dir=''):
        # patches: [B, 5, C, H, W]        
        # Extract features (shared weights)
        B = patches.size(0)
        sgm_outputs = []
        _5patch_features = []
        for i in range(5):
            patch = patches[:, i]  # 3, H, W
            sgm_output = self.sgm(patch)  #return_nl=True indicates debug mode
            d_flatten = sgm_output['descriptor_flatten']  # [B, K*D]
            _5patch_features.append(d_flatten)
            sgm_outputs.append(sgm_output)

        f1, f2, f3, f4, uav_patch_feature = _5patch_features
        neighbor_feats = torch.stack([f1, f2, f3, f4], dim=1)  # [B,K,D]

        # 4 neighborhood coordinates, origin at block center, u as unit length, according to translation image processing coordinate system
        u = 1.0
        known_coords = torch.tensor([[-u,-u], [-u,u], [u,-u], [u,u]],
                                   dtype=torch.float32, device=patches.device) 
        # [4,D] --> [1,4,D] --> [B,4,D]
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).repeat(B,1,1)  # [B,4,D]

        # Enhance neighborhood features
        
        pos_soft_prior = self.sim_pos_prior(uav_patch_feature, neighbor_feats)  # [B, 2]  v5

        if self.add_patch_coord:
            neighbor_feats = neighbor_feats + coord_embs  # [B,4,D]

        # Cross attention
        ctx_feat = self.neighbors_cross_attn(uav_patch_feature, neighbor_feats)  # [B, 32]

        # Feature fusion + position info par_ca_sgm_v4a
        combined = torch.cat([uav_patch_feature, ctx_feat], dim=1)  # [B, D + D']
        combined_with_prior = torch.cat([combined, pos_soft_prior], dim=1)  # [B, D + D' + 2]  v5

        # Regression
        pos_pred = self.pos_regressor(combined_with_prior)
        dir_pred = self.dir_regressor(combined)

        return pos_pred, dir_pred

class PARCASGM_v5a(PARCASGM_v5):
    """
    PARCASGM_v5 class with softmax normalization
    Inherits from PARCASGM_v5, only difference is using CSMG_soft and JointNet_soft
    """
    def __init__(self, 
                 backbone_name='vgg16',
                 feature_dim=256, 
                 coord_enc_dims=[16, 64, 256],   # Coordinate encoding layer dimension sequence
                 regressor_dims=[1024, 256, 64],  # Modified to 3-layer structure
                 reduction_ratio=1,
                 num_clusters=4,
                 freeze_backbone=True,
                 partial_unfreeze=False,
                 add_patch_coord=True):
        """
        PARCASGM_v5a: Version with softmax normalization
        Only difference from PARCASGM_v5 is using CSMG_soft and JointNet_soft
        """
        # Call parent class initialization
        super().__init__(
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord
        )
        
        # Update model name
        self.model_name = 'phr5'
        
        # Redefine csmg and sgm with soft version
        self.csmg = CSMG_soft(input_channel=self.backbone_out_dim, output_channel=self.feature_dim, num_clusters=num_clusters)
        self.sgm = JointNet_soft(None, self.csmg)
        self.sgm.backbone = self.backbone


class PARCASGM_v5a_GlobalRST(PARCASGM_v5a):
    """Experiment B: add global context among four contiguous RSTs."""

    def __init__(
        self,
        backbone_name='vgg16',
        feature_dim=256,
        coord_enc_dims=[16, 64, 256],
        regressor_dims=[1024, 256, 64],
        reduction_ratio=1,
        num_clusters=4,
        freeze_backbone=True,
        partial_unfreeze=False,
        add_patch_coord=True,
        global_token_grid_size=4,
        global_num_heads=8,
        global_feedforward_dim=512,
        global_dropout=0.1,
    ):
        super().__init__(
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord,
        )
        self.model_name = 'phr5_globalrst_b'
        descriptor_dim = feature_dim * num_clusters
        self.rst_global_fusion = RSTGlobalContextFusion(
            feature_dim=feature_dim,
            descriptor_dim=descriptor_dim,
            token_grid_size=global_token_grid_size,
            num_heads=global_num_heads,
            feedforward_dim=global_feedforward_dim,
            dropout=global_dropout,
        )

    def forward(
        self,
        patches,
        debug_dir='',
        return_aux=False,
        attention_temperature=1.0,
    ):
        # patches: [B, 5, C, H, W], ordered as p1, p2, p3, p4, UAV.
        if patches.dim() != 5 or patches.size(1) != 5:
            raise ValueError(
                f"Expected patches shaped [B, 5, C, H, W], got {tuple(patches.shape)}"
            )

        batch_size = patches.size(0)
        sgm_outputs = [self.sgm(patches[:, i]) for i in range(5)]

        rst_descriptors = torch.stack(
            [output['descriptor_flatten'] for output in sgm_outputs[:4]],
            dim=1,
        )
        rst_maps = torch.stack(
            [output['nl_feat'] for output in sgm_outputs[:4]],
            dim=1,
        )
        fusion_output = self.rst_global_fusion(
            rst_maps, rst_descriptors, return_aux=return_aux
        )
        if return_aux:
            neighbor_feats, auxiliary = fusion_output
        else:
            neighbor_feats = fusion_output
        uav_patch_feature = sgm_outputs[4]['descriptor_flatten']

        known_coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=patches.dtype,
            device=patches.device,
        )
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        prior_output = self.sim_pos_prior(
            uav_patch_feature,
            neighbor_feats,
            temperature=attention_temperature,
            return_weights=return_aux,
        )
        if return_aux:
            pos_soft_prior, attention_weights = prior_output
            auxiliary['attention_weights'] = attention_weights
            auxiliary['position_prior'] = pos_soft_prior
        else:
            pos_soft_prior = prior_output
        cross_attn_feats = neighbor_feats
        if self.add_patch_coord:
            cross_attn_feats = cross_attn_feats + coord_embs

        ctx_feat = self.neighbors_cross_attn(uav_patch_feature, cross_attn_feats)
        combined = torch.cat((uav_patch_feature, ctx_feat), dim=1)
        combined_with_prior = torch.cat((combined, pos_soft_prior), dim=1)

        pos_pred = self.pos_regressor(combined_with_prior)
        dir_pred = self.dir_regressor(combined)
        if return_aux:
            return pos_pred, dir_pred, auxiliary
        return pos_pred, dir_pred


class PARCASGM_v5a_GlobalRST_PosPrior(PARCASGM_v5a_GlobalRST):
    """Experiment F: use global RST context only in the position prior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'phr5_globalrst_f'

    def forward(
        self,
        patches,
        debug_dir='',
        return_aux=False,
        attention_temperature=1.0,
    ):
        # patches: [B, 5, C, H, W], ordered as p1, p2, p3, p4, UAV.
        if patches.dim() != 5 or patches.size(1) != 5:
            raise ValueError(
                f"Expected patches shaped [B, 5, C, H, W], got {tuple(patches.shape)}"
            )

        batch_size = patches.size(0)
        sgm_outputs = [self.sgm(patches[:, i]) for i in range(5)]
        original_rst_descriptors = torch.stack(
            [output['descriptor_flatten'] for output in sgm_outputs[:4]],
            dim=1,
        )
        rst_maps = torch.stack(
            [output['nl_feat'] for output in sgm_outputs[:4]],
            dim=1,
        )
        fusion_output = self.rst_global_fusion(
            rst_maps, original_rst_descriptors, return_aux=return_aux
        )
        if return_aux:
            global_rst_descriptors, auxiliary = fusion_output
        else:
            global_rst_descriptors = fusion_output

        uav_patch_feature = sgm_outputs[4]['descriptor_flatten']
        known_coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=patches.dtype,
            device=patches.device,
        )
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        # The global descriptors influence only the PSG position prior.
        prior_output = self.sim_pos_prior(
            uav_patch_feature,
            global_rst_descriptors,
            temperature=attention_temperature,
            return_weights=return_aux,
        )
        if return_aux:
            pos_soft_prior, attention_weights = prior_output
            auxiliary['attention_weights'] = attention_weights
            auxiliary['position_prior'] = pos_soft_prior
            auxiliary['global_rst_descriptors'] = global_rst_descriptors
        else:
            pos_soft_prior = prior_output

        # Keep the official CA and heading path on the original RST descriptors.
        cross_attn_feats = original_rst_descriptors
        if self.add_patch_coord:
            cross_attn_feats = cross_attn_feats + coord_embs
        ctx_feat = self.neighbors_cross_attn(
            uav_patch_feature, cross_attn_feats
        )

        combined = torch.cat((uav_patch_feature, ctx_feat), dim=1)
        pos_pred = self.pos_regressor(
            torch.cat((combined, pos_soft_prior), dim=1)
        )
        dir_pred = self.dir_regressor(combined)
        if return_aux:
            return pos_pred, dir_pred, auxiliary
        return pos_pred, dir_pred


class PARCASGM_v5a_GlobalRST_PosPrior_LCPR1(
    PARCASGM_v5a_GlobalRST_PosPrior
):
    """Joint-100 experiment: Experiment F plus local position refinement."""

    model_name = 'phr5_globalrst_f_lcpr1_joint100'

    def __init__(
        self,
        lcpr_feature_dim=64,
        lcpr_radius=0.15,
        lcpr_temperature=0.1,
        lcpr_gate_bias=-2.0,
        **kwargs,
    ):
        feature_dim = kwargs.get('feature_dim', 256)
        super().__init__(**kwargs)
        self.model_name = 'phr5_globalrst_f_lcpr1_joint100'
        self.local_position_refiner = LocalCandidatePositionRefiner(
            input_dim=feature_dim,
            feature_dim=lcpr_feature_dim,
            radius=lcpr_radius,
            temperature=lcpr_temperature,
            gate_bias=lcpr_gate_bias,
        )

    def forward(
        self,
        patches,
        debug_dir='',
        return_aux=False,
        attention_temperature=1.0,
    ):
        if patches.dim() != 5 or patches.size(1) != 5:
            raise ValueError(
                f"Expected patches shaped [B, 5, C, H, W], got {tuple(patches.shape)}"
            )

        batch_size = patches.size(0)
        sgm_outputs = [self.sgm(patches[:, index]) for index in range(5)]
        original_rst_descriptors = torch.stack(
            [output['descriptor_flatten'] for output in sgm_outputs[:4]], dim=1
        )
        rst_maps = torch.stack(
            [output['nl_feat'] for output in sgm_outputs[:4]], dim=1
        )
        fusion_output = self.rst_global_fusion(
            rst_maps, original_rst_descriptors, return_aux=return_aux
        )
        if return_aux:
            global_rst_descriptors, auxiliary = fusion_output
        else:
            global_rst_descriptors = fusion_output

        uav_descriptor = sgm_outputs[4]['descriptor_flatten']
        uav_map = sgm_outputs[4]['nl_feat']
        known_coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=uav_descriptor.dtype,
            device=uav_descriptor.device,
        )
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).expand(
            batch_size, -1, -1
        )

        prior_output = self.sim_pos_prior(
            uav_descriptor,
            global_rst_descriptors,
            temperature=attention_temperature,
            return_weights=return_aux,
        )
        if return_aux:
            position_prior, attention_weights = prior_output
            auxiliary['attention_weights'] = attention_weights
            auxiliary['position_prior'] = position_prior
            auxiliary['global_rst_descriptors'] = global_rst_descriptors
        else:
            position_prior = prior_output

        cross_attention_features = original_rst_descriptors
        if self.add_patch_coord:
            cross_attention_features = cross_attention_features + coord_embs
        context = self.neighbors_cross_attn(
            uav_descriptor, cross_attention_features
        )
        combined = torch.cat((uav_descriptor, context), dim=1)
        coarse_position = self.pos_regressor(
            torch.cat((combined, position_prior), dim=1)
        )
        heading = self.dir_regressor(combined)
        position, lcpr_auxiliary = self.local_position_refiner(
            rst_maps, uav_map, coarse_position
        )

        if return_aux:
            auxiliary.update(lcpr_auxiliary)
            return position, heading, auxiliary
        return position, heading


class PARCASGM_v5a_MSRDCP_P5(PARCASGM_v5a_GlobalRST_PosPrior):
    """P5 stage 1: frozen experiment F plus dense probabilistic positioning."""

    model_name = 'phr5_msrdcp_p5_s1'
    requires_init_checkpoint = True
    initialization_missing_prefixes = ('dense_position_head.',)
    uses_position_distribution_loss = True
    default_learning_rate = 1e-4
    default_optimizer = 'AdamW'

    def __init__(
        self,
        dense_feature_dim=64,
        dense_num_rotations=16,
        dense_output_size=32,
        dense_scale_sizes=(16, 24, 32),
        dense_template_sizes=(6, 8, 10),
        dense_rotation_temperature=0.15,
        dense_heatmap_temperature=0.10,
        dense_gate_bias=-2.0,
        dense_refine_radius=0.125,
        dense_heatmap_sigma=0.08,
        dense_heatmap_loss_weight=0.30,
        dense_coarse_loss_weight=0.20,
        dense_refine_loss_weight=0.10,
        freeze_base=True,
        **kwargs,
    ):
        feature_dim = kwargs.get('feature_dim', 256)
        super().__init__(**kwargs)
        if dense_heatmap_sigma <= 0:
            raise ValueError("Dense heatmap sigma must be positive")
        if min(
            dense_heatmap_loss_weight,
            dense_coarse_loss_weight,
            dense_refine_loss_weight,
        ) < 0:
            raise ValueError("Dense position loss weights must be non-negative")
        self.model_name = 'phr5_msrdcp_p5_s1'
        self.freeze_base = bool(freeze_base)
        self.dense_heatmap_sigma = float(dense_heatmap_sigma)
        self.dense_heatmap_loss_weight = float(dense_heatmap_loss_weight)
        self.dense_coarse_loss_weight = float(dense_coarse_loss_weight)
        self.dense_refine_loss_weight = float(dense_refine_loss_weight)
        if self.freeze_base:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.dense_position_head = MultiScaleRotationMarginalizedPositionHead(
            input_dim=feature_dim,
            feature_dim=dense_feature_dim,
            num_rotations=dense_num_rotations,
            output_size=dense_output_size,
            scale_sizes=dense_scale_sizes,
            template_sizes=dense_template_sizes,
            rotation_temperature=dense_rotation_temperature,
            heatmap_temperature=dense_heatmap_temperature,
            gate_bias=dense_gate_bias,
            refine_radius=dense_refine_radius,
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_base:
            for name, module in self.named_children():
                if name != 'dense_position_head':
                    module.eval()
            self.dense_position_head.train(True)
        return self

    def _experiment_f_forward_with_maps(self, patches):
        batch_size = patches.size(0)
        sgm_outputs = [self.sgm(patches[:, index]) for index in range(5)]
        rst_descriptors = torch.stack(
            [output['descriptor_flatten'] for output in sgm_outputs[:4]], dim=1
        )
        rst_maps = torch.stack(
            [output['nl_feat'] for output in sgm_outputs[:4]], dim=1
        )
        global_descriptors = self.rst_global_fusion(rst_maps, rst_descriptors)
        uav_descriptor = sgm_outputs[4]['descriptor_flatten']
        uav_map = sgm_outputs[4]['nl_feat']
        known_coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=uav_descriptor.dtype,
            device=uav_descriptor.device,
        )
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        prior = self.sim_pos_prior(uav_descriptor, global_descriptors)
        cross_attention_features = rst_descriptors
        if self.add_patch_coord:
            cross_attention_features = cross_attention_features + coord_embs
        context = self.neighbors_cross_attn(
            uav_descriptor, cross_attention_features
        )
        combined = torch.cat((uav_descriptor, context), dim=1)
        base_position = self.pos_regressor(torch.cat((combined, prior), dim=1))
        base_heading = self.dir_regressor(combined)
        return base_position, base_heading, rst_maps, uav_map

    def forward(self, patches, debug_dir='', return_aux=False):
        if patches.dim() != 5 or patches.size(1) != 5:
            raise ValueError(
                f"Expected patches shaped [B, 5, C, H, W], got {tuple(patches.shape)}"
            )
        if self.freeze_base:
            with torch.no_grad():
                base_position, heading, rst_maps, uav_map = (
                    self._experiment_f_forward_with_maps(patches)
                )
        else:
            base_position, heading, rst_maps, uav_map = (
                self._experiment_f_forward_with_maps(patches)
            )
        position, auxiliary = self.dense_position_head(
            rst_maps, uav_map, base_position
        )
        auxiliary['base_position'] = base_position
        if return_aux:
            return position, heading, auxiliary
        return position, heading

    def compute_position_losses(self, auxiliary, target_position):
        target_position = target_position.float()
        probability = auxiliary['position_probability'].float()
        grid = self.dense_position_head.position_grid.to(
            device=target_position.device, dtype=target_position.dtype
        )
        squared_distance = (
            grid[None] - target_position[:, None, None, :]
        ).square().sum(dim=-1)
        target_heatmap = torch.softmax(
            -squared_distance.flatten(1) / (2.0 * self.dense_heatmap_sigma ** 2),
            dim=-1,
        ).reshape_as(probability)
        heatmap = F.kl_div(
            probability.clamp_min(1e-8).log(), target_heatmap,
            reduction='batchmean',
        )
        coarse = F.smooth_l1_loss(
            auxiliary['coarse_position'].float(), target_position
        )
        refine = F.smooth_l1_loss(
            auxiliary['dense_position'].float(), target_position
        )
        total = (
            self.dense_heatmap_loss_weight * heatmap
            + self.dense_coarse_loss_weight * coarse
            + self.dense_refine_loss_weight * refine
        )
        return {
            'heatmap': heatmap,
            'coarse': coarse,
            'refine': refine,
            'total': total,
        }


class PARCASGM_v5a_GlobalRST_Aux(PARCASGM_v5a_GlobalRST):
    """Experiment E: global RST fusion with quadrant and geometry constraints."""

    uses_auxiliary_losses = True

    def __init__(
        self,
        backbone_name='vgg16',
        feature_dim=256,
        coord_enc_dims=[16, 64, 256],
        regressor_dims=[1024, 256, 64],
        reduction_ratio=1,
        num_clusters=4,
        freeze_backbone=True,
        partial_unfreeze=False,
        add_patch_coord=True,
        global_token_grid_size=4,
        global_num_heads=8,
        global_feedforward_dim=512,
        global_dropout=0.1,
        quad_loss_weight=0.05,
        attention_loss_weight=0.10,
        auxiliary_warmup_epochs=5,
        attention_temperature=1.0,
    ):
        super().__init__(
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord,
            global_token_grid_size=global_token_grid_size,
            global_num_heads=global_num_heads,
            global_feedforward_dim=global_feedforward_dim,
            global_dropout=global_dropout,
        )
        if quad_loss_weight < 0 or attention_loss_weight < 0:
            raise ValueError("Auxiliary loss weights must be non-negative")
        if auxiliary_warmup_epochs < 0:
            raise ValueError("auxiliary_warmup_epochs must be non-negative")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")

        self.model_name = 'phr5_globalrst_e'
        self.quad_loss_weight = quad_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.auxiliary_warmup_epochs = auxiliary_warmup_epochs
        self.attention_temperature = attention_temperature

    def forward(self, patches, debug_dir='', return_aux=False):
        return super().forward(
            patches,
            debug_dir=debug_dir,
            return_aux=return_aux,
            attention_temperature=self.attention_temperature,
        )

    def auxiliary_weight_scale(self, epoch):
        if self.auxiliary_warmup_epochs == 0:
            return 1.0
        return min(1.0, float(epoch + 1) / self.auxiliary_warmup_epochs)

    def compute_auxiliary_losses(self, auxiliary, coords, epoch):
        original = auxiliary['original_descriptors'].detach()
        context = auxiliary['context_descriptors']
        loss_quad = (1.0 - F.cosine_similarity(context, original, dim=-1)).mean()

        rel_coords = self.sim_pos_prior.rel_coords.to(
            device=coords.device, dtype=coords.dtype
        )
        geometry_weights = (
            (1.0 + coords[:, None, 0] * rel_coords[None, :, 0])
            * (1.0 + coords[:, None, 1] * rel_coords[None, :, 1])
            / 4.0
        )
        geometry_weights = geometry_weights.clamp_min(0.0)
        geometry_weights = geometry_weights / geometry_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)

        attention_weights = auxiliary['attention_weights']
        loss_attention = F.kl_div(
            attention_weights.float().clamp_min(1e-6).log(),
            geometry_weights.float(),
            reduction='batchmean',
        )

        warmup_scale = self.auxiliary_weight_scale(epoch)
        weighted_quad = warmup_scale * self.quad_loss_weight * loss_quad
        weighted_attention = (
            warmup_scale * self.attention_loss_weight * loss_attention
        )
        return {
            'quad': loss_quad,
            'attention': loss_attention,
            'weighted_quad': weighted_quad,
            'weighted_attention': weighted_attention,
            'total': weighted_quad + weighted_attention,
            'warmup_scale': warmup_scale,
        }


class PARCASGM_v5a_GlobalRST_Quad(PARCASGM_v5a_GlobalRST_Aux):
    """Experiment C: global RST fusion constrained only by L_quad."""

    def __init__(self, *args, **kwargs):
        attention_loss_weight = kwargs.pop('attention_loss_weight', 0.0)
        if attention_loss_weight != 0:
            raise ValueError(
                "Experiment C requires attention_loss_weight=0"
            )
        kwargs.setdefault('quad_loss_weight', 0.05)
        super().__init__(
            *args,
            attention_loss_weight=0.0,
            **kwargs,
        )
        self.model_name = 'phr5_globalrst_c'


class PARCASGM_v5a_GlobalRST_Attn(PARCASGM_v5a_GlobalRST_Aux):
    """Experiment D: global RST fusion constrained only by L_attn."""

    def __init__(self, *args, **kwargs):
        quad_loss_weight = kwargs.pop('quad_loss_weight', 0.0)
        if quad_loss_weight != 0:
            raise ValueError("Experiment D requires quad_loss_weight=0")
        kwargs.setdefault('attention_loss_weight', 0.10)
        super().__init__(
            *args,
            quad_loss_weight=0.0,
            **kwargs,
        )
        self.model_name = 'phr5_globalrst_d'


class PARCASGM_v5a_H1(PARCASGM_v5a):
    """Experiment H1: unit-circle output and circular heading objective."""

    default_loss_type = 'circular'

    def __init__(self,
                 backbone_name='vgg16',
                 feature_dim=256,
                 coord_enc_dims=[16, 64, 256],
                 regressor_dims=[1024, 256, 64],
                 reduction_ratio=1,
                 num_clusters=4,
                 freeze_backbone=True,
                 partial_unfreeze=False,
                 add_patch_coord=True,
                 heading_norm_eps=1e-6):
        super().__init__(
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord,
        )
        self.model_name = 'phr5_h1_circle'
        self.heading_norm_eps = heading_norm_eps
        self.dir_regressor = nn.Sequential(
            *list(self.dir_regressor.children()),
            UnitL2Normalize(eps=heading_norm_eps),
        )


class PARCASGM_v5a_MSPCOC(PARCASGM_v5a):
    """Experiment H2: learned cross-view orientation correlation."""

    default_loss_type = 'mspcoc'
    default_optimizer = 'AdamW'
    default_learning_rate = 1e-4
    requires_init_checkpoint = True
    uses_heading_distribution_loss = True

    def __init__(
        self,
        backbone_name='vgg16',
        feature_dim=256,
        coord_enc_dims=[16, 64, 256],
        regressor_dims=[1024, 256, 64],
        reduction_ratio=1,
        num_clusters=4,
        freeze_backbone=True,
        partial_unfreeze=False,
        add_patch_coord=True,
        heading_feature_dim=64,
        heading_radial_bins=8,
        heading_angle_bins=72,
        heading_crop_scales=(0.75, 1.0, 1.25),
        heading_base_radius=0.5,
        heading_num_heads=4,
        heading_dropout=0.1,
        heading_temperature=0.1,
        heading_scale_temperature=0.2,
        heading_gate_bias=-3.0,
        heading_kappa=32.0,
        heading_dist_weight=1.0,
        heading_corr_weight=0.5,
        heading_final_weight=0.5,
        freeze_base=True,
    ):
        super().__init__(
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord,
        )
        if heading_kappa <= 0:
            raise ValueError("heading_kappa must be positive")
        if min(
            heading_dist_weight,
            heading_corr_weight,
            heading_final_weight,
        ) < 0:
            raise ValueError("MS-PCOC loss weights must be non-negative")

        self.model_name = 'phr5_h2_mspcoc'
        self.freeze_base = bool(freeze_base)
        self.heading_kappa = float(heading_kappa)
        self.heading_dist_weight = float(heading_dist_weight)
        self.heading_corr_weight = float(heading_corr_weight)
        self.heading_final_weight = float(heading_final_weight)
        self.ms_pcoc_head = MSPCOCHeadingHead(
            input_dim=feature_dim,
            feature_dim=heading_feature_dim,
            num_radial_bins=heading_radial_bins,
            num_angle_bins=heading_angle_bins,
            crop_scales=heading_crop_scales,
            base_radius=heading_base_radius,
            num_heads=heading_num_heads,
            dropout=heading_dropout,
            orientation_temperature=heading_temperature,
            scale_temperature=heading_scale_temperature,
            gate_bias=heading_gate_bias,
        )
        if self.freeze_base:
            for name, parameter in self.named_parameters():
                if not name.startswith('ms_pcoc_head.'):
                    parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_base:
            for name, module in self.named_children():
                if name != 'ms_pcoc_head':
                    module.eval()
            self.ms_pcoc_head.train(True)
        return self

    def _official_forward_with_maps(self, patches):
        batch_size = patches.size(0)
        sgm_outputs = [self.sgm(patches[:, i]) for i in range(5)]
        rst_descriptors = torch.stack(
            [output['descriptor_flatten'] for output in sgm_outputs[:4]], dim=1
        )
        rst_maps = torch.stack(
            [output['nl_feat'] for output in sgm_outputs[:4]], dim=1
        )
        uav_descriptor = sgm_outputs[4]['descriptor_flatten']
        uav_map = sgm_outputs[4]['nl_feat']

        known_coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=uav_descriptor.dtype,
            device=uav_descriptor.device,
        )
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        pos_soft_prior = self.sim_pos_prior(uav_descriptor, rst_descriptors)
        cross_attention_features = rst_descriptors
        if self.add_patch_coord:
            cross_attention_features = cross_attention_features + coord_embs
        context = self.neighbors_cross_attn(
            uav_descriptor, cross_attention_features
        )
        combined = torch.cat((uav_descriptor, context), dim=1)
        position = self.pos_regressor(
            torch.cat((combined, pos_soft_prior), dim=1)
        )
        base_heading = self.dir_regressor(combined)
        return position, base_heading, rst_maps, uav_map

    def forward(
        self,
        patches,
        debug_dir='',
        return_aux=False,
        heading_position=None,
    ):
        if patches.dim() != 5 or patches.size(1) != 5:
            raise ValueError(
                f"Expected patches shaped [B, 5, C, H, W], got {tuple(patches.shape)}"
            )
        if self.freeze_base:
            with torch.no_grad():
                position, base_heading, rst_maps, uav_map = (
                    self._official_forward_with_maps(patches)
                )
        else:
            position, base_heading, rst_maps, uav_map = (
                self._official_forward_with_maps(patches)
            )
        correlation_center = position if heading_position is None else heading_position
        final_heading, auxiliary = self.ms_pcoc_head(
            rst_maps,
            uav_map,
            correlation_center,
            base_heading,
        )
        auxiliary['heading_position'] = correlation_center.detach()
        if return_aux:
            return position, final_heading, auxiliary
        return position, final_heading

    @staticmethod
    def _wrapped_heading_loss(prediction, target):
        prediction = F.normalize(prediction, p=2, dim=-1, eps=1e-6)
        target = F.normalize(target, p=2, dim=-1, eps=1e-6)
        cross = prediction[:, 0] * target[:, 1] - prediction[:, 1] * target[:, 0]
        dot = (prediction * target).sum(dim=-1).clamp(-1.0, 1.0)
        delta = torch.atan2(cross, dot)
        return F.smooth_l1_loss(delta, torch.zeros_like(delta))

    def compute_heading_losses(
        self, auxiliary, target_heading, target_position=None
    ):
        probability = auxiliary['orientation_probability'].float()
        target_heading = F.normalize(
            target_heading.float(), p=2, dim=-1, eps=1e-6
        )
        target_angle = torch.atan2(
            target_heading[:, 1], target_heading[:, 0]
        )
        basis = self.ms_pcoc_head.heading_basis.float()
        bin_angles = torch.atan2(basis[:, 1], basis[:, 0])
        circular_offset = bin_angles.unsqueeze(0) - target_angle.unsqueeze(1)
        soft_target = torch.softmax(
            self.heading_kappa * torch.cos(circular_offset), dim=-1
        )
        distribution = -(
            soft_target * probability.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        correlation = self._wrapped_heading_loss(
            auxiliary['correlation_heading'].float(), target_heading
        )
        final = self._wrapped_heading_loss(
            auxiliary['final_heading'].float(), target_heading
        )
        total = (
            self.heading_dist_weight * distribution
            + self.heading_corr_weight * correlation
            + self.heading_final_weight * final
        )
        return {
            'distribution': distribution,
            'correlation': correlation,
            'final': final,
            'total': total,
        }


class PARCASGM_v5a_GPRVH(PARCASGM_v5a):
    """Experiment H3-D: geometry-preserved rotation pose-volume heading."""

    model_name = 'phr5_h3d_gprvh'
    default_optimizer = 'AdamW'
    default_learning_rate = 1e-4
    default_loss_type = 'gprvh'
    requires_init_checkpoint = True
    uses_heading_distribution_loss = True
    initialization_missing_prefixes = ('gprv_head.',)

    def __init__(
        self,
        backbone_name='vgg16',
        feature_dim=256,
        coord_enc_dims=[16, 64, 256],
        regressor_dims=[1024, 256, 64],
        reduction_ratio=1,
        num_clusters=4,
        freeze_backbone=True,
        partial_unfreeze=False,
        add_patch_coord=True,
        heading_feature_dim=64,
        heading_spatial_size=8,
        heading_candidate_grid_size=9,
        heading_candidate_radius=0.5,
        heading_angle_bins=36,
        heading_crop_half_extent=0.5,
        heading_orientation_temperature=0.1,
        heading_translation_temperature=0.2,
        heading_residual_limit_degrees=5.0,
        heading_position_sigma=0.15,
        heading_kappa=20.0,
        heading_volume_weight=1.0,
        heading_circle_weight=0.5,
        heading_phase2_weight=0.15,
        heading_opposite_weight=0.1,
        heading_opposite_margin=0.2,
        freeze_base=True,
    ):
        super().__init__(
            backbone_name=backbone_name,
            feature_dim=feature_dim,
            coord_enc_dims=coord_enc_dims,
            regressor_dims=regressor_dims,
            reduction_ratio=reduction_ratio,
            num_clusters=num_clusters,
            freeze_backbone=freeze_backbone,
            partial_unfreeze=partial_unfreeze,
            add_patch_coord=add_patch_coord,
        )
        if heading_position_sigma <= 0 or heading_kappa <= 0:
            raise ValueError("H3-D position sigma and heading kappa must be positive")
        if min(
            heading_volume_weight,
            heading_circle_weight,
            heading_phase2_weight,
            heading_opposite_weight,
            heading_opposite_margin,
        ) < 0:
            raise ValueError("H3-D loss weights and margin must be non-negative")

        self.model_name = type(self).model_name
        self.freeze_base = bool(freeze_base)
        self.heading_position_sigma = float(heading_position_sigma)
        self.heading_kappa = float(heading_kappa)
        self.heading_volume_weight = float(heading_volume_weight)
        self.heading_circle_weight = float(heading_circle_weight)
        self.heading_phase2_weight = float(heading_phase2_weight)
        self.heading_opposite_weight = float(heading_opposite_weight)
        self.heading_opposite_margin = float(heading_opposite_margin)
        if self.freeze_base:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.gprv_head = GeometryPoseVolumeHeadingHead(
            input_dim=feature_dim,
            feature_dim=heading_feature_dim,
            spatial_size=heading_spatial_size,
            candidate_grid_size=heading_candidate_grid_size,
            candidate_radius=heading_candidate_radius,
            num_angle_bins=heading_angle_bins,
            crop_half_extent=heading_crop_half_extent,
            orientation_temperature=heading_orientation_temperature,
            translation_temperature=heading_translation_temperature,
            residual_limit_degrees=heading_residual_limit_degrees,
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_base:
            for name, module in self.named_children():
                if name != 'gprv_head':
                    module.eval()
            self.gprv_head.train(True)
        return self

    def _official_forward_with_maps(self, patches):
        batch_size = patches.size(0)
        sgm_outputs = [self.sgm(patches[:, index]) for index in range(5)]
        rst_descriptors = torch.stack(
            [output['descriptor_flatten'] for output in sgm_outputs[:4]], dim=1
        )
        rst_maps = torch.stack(
            [output['nl_feat'] for output in sgm_outputs[:4]], dim=1
        )
        uav_descriptor = sgm_outputs[4]['descriptor_flatten']
        uav_map = sgm_outputs[4]['nl_feat']
        known_coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=uav_descriptor.dtype,
            device=uav_descriptor.device,
        )
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        pos_soft_prior = self.sim_pos_prior(uav_descriptor, rst_descriptors)
        cross_attention_features = rst_descriptors
        if self.add_patch_coord:
            cross_attention_features = cross_attention_features + coord_embs
        context = self.neighbors_cross_attn(
            uav_descriptor, cross_attention_features
        )
        combined = torch.cat((uav_descriptor, context), dim=1)
        position = self.pos_regressor(
            torch.cat((combined, pos_soft_prior), dim=1)
        )
        base_heading = self.dir_regressor(combined)
        return position, base_heading, rst_maps, uav_map

    def forward(
        self,
        patches,
        debug_dir='',
        return_aux=False,
        heading_position=None,
    ):
        if patches.dim() != 5 or patches.size(1) != 5:
            raise ValueError(
                f"Expected patches shaped [B, 5, C, H, W], got {tuple(patches.shape)}"
            )
        if self.freeze_base:
            with torch.no_grad():
                position, base_heading, rst_maps, uav_map = (
                    self._official_forward_with_maps(patches)
                )
        else:
            position, base_heading, rst_maps, uav_map = (
                self._official_forward_with_maps(patches)
            )
        correlation_center = position if heading_position is None else heading_position
        final_heading, auxiliary = self.gprv_head(
            rst_maps,
            uav_map,
            correlation_center,
            base_heading,
        )
        auxiliary['heading_position'] = correlation_center.detach()
        auxiliary['base_heading'] = F.normalize(
            base_heading, p=2, dim=-1, eps=1e-6
        )
        if return_aux:
            return position, final_heading, auxiliary
        return position, final_heading

    def compute_heading_losses(
        self, auxiliary, target_heading, target_position=None
    ):
        if target_position is None:
            raise ValueError("H3-D pose-volume loss requires target_position")
        target_heading = F.normalize(
            target_heading.float(), p=2, dim=-1, eps=1e-6
        )
        target_position = target_position.float()
        target_angle = torch.atan2(target_heading[:, 1], target_heading[:, 0])
        candidate_offsets = auxiliary['candidate_offsets'].float()
        displacement = target_position - auxiliary['heading_position'].float()
        position_delta = displacement[:, None] - candidate_offsets[None]
        position_log_target = -position_delta.square().sum(dim=-1) / (
            2.0 * self.heading_position_sigma ** 2
        )

        basis = self.gprv_head.heading_basis.float()
        bin_angles = torch.atan2(basis[:, 1], basis[:, 0])
        circular_offset = bin_angles[None] - target_angle[:, None]
        angle_log_target = self.heading_kappa * torch.cos(circular_offset)
        joint_log_target = (
            position_log_target[:, :, None] + angle_log_target[:, None, :]
        )
        joint_target = torch.softmax(joint_log_target.flatten(1), dim=-1)
        volume_log_probability = torch.log_softmax(
            auxiliary['pose_volume_logits'].float().flatten(1)
            / self.gprv_head.orientation_temperature,
            dim=-1,
        )
        volume = -(joint_target * volume_log_probability).sum(dim=-1).mean()

        final_heading = F.normalize(
            auxiliary['final_heading'].float(), p=2, dim=-1, eps=1e-6
        )
        circle = (1.0 - (final_heading * target_heading).sum(dim=-1)).mean()
        target_phase2 = torch.stack(
            (torch.cos(2.0 * target_angle), torch.sin(2.0 * target_angle)),
            dim=-1,
        )
        phase2_heading = F.normalize(
            auxiliary['phase2_heading'].float(), p=2, dim=-1, eps=1e-6
        )
        phase2 = (
            1.0 - (phase2_heading * target_phase2).sum(dim=-1)
        ).mean()

        wrapped_angle = torch.remainder(target_angle, 2.0 * math.pi)
        target_index = torch.round(
            wrapped_angle * self.gprv_head.num_angle_bins / (2.0 * math.pi)
        ).long() % self.gprv_head.num_angle_bins
        opposite_index = (
            target_index + self.gprv_head.num_angle_bins // 2
        ) % self.gprv_head.num_angle_bins
        orientation_logits = auxiliary['orientation_logits'].float()
        true_score = orientation_logits.gather(1, target_index[:, None]).squeeze(1)
        opposite_score = orientation_logits.gather(
            1, opposite_index[:, None]
        ).squeeze(1)
        opposite = F.relu(
            self.heading_opposite_margin + opposite_score - true_score
        ).mean()

        total = (
            self.heading_volume_weight * volume
            + self.heading_circle_weight * circle
            + self.heading_phase2_weight * phase2
            + self.heading_opposite_weight * opposite
        )
        return {
            'distribution': volume,
            'correlation': phase2,
            'final': circle,
            'opposite': opposite,
            'total': total,
        }


class PARCASGM_v5a_GlobalRST_PosPrior_GPRVH(PARCASGM_v5a_GPRVH):
    """Stage 1: frozen experiment F localization with a trainable GPRV-H head."""

    model_name = 'phr5_globalrst_f_gprvh_s1'
    initialization_missing_prefixes = ('gprv_head.',)

    def __init__(
        self,
        global_token_grid_size=4,
        global_num_heads=8,
        global_feedforward_dim=512,
        global_dropout=0.1,
        **kwargs,
    ):
        feature_dim = kwargs.get('feature_dim', 256)
        num_clusters = kwargs.get('num_clusters', 4)
        super().__init__(**kwargs)
        self.rst_global_fusion = RSTGlobalContextFusion(
            feature_dim=feature_dim,
            descriptor_dim=feature_dim * num_clusters,
            token_grid_size=global_token_grid_size,
            num_heads=global_num_heads,
            feedforward_dim=global_feedforward_dim,
            dropout=global_dropout,
        )
        if self.freeze_base:
            for parameter in self.rst_global_fusion.parameters():
                parameter.requires_grad = False

    def _official_forward_with_maps(self, patches):
        batch_size = patches.size(0)
        sgm_outputs = [self.sgm(patches[:, index]) for index in range(5)]
        rst_descriptors = torch.stack(
            [output['descriptor_flatten'] for output in sgm_outputs[:4]], dim=1
        )
        rst_maps = torch.stack(
            [output['nl_feat'] for output in sgm_outputs[:4]], dim=1
        )
        global_rst_descriptors = self.rst_global_fusion(
            rst_maps, rst_descriptors
        )
        uav_descriptor = sgm_outputs[4]['descriptor_flatten']
        uav_map = sgm_outputs[4]['nl_feat']

        known_coords = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=uav_descriptor.dtype,
            device=uav_descriptor.device,
        )
        coord_embs = self.coord_encoder(known_coords).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        pos_soft_prior = self.sim_pos_prior(
            uav_descriptor, global_rst_descriptors
        )
        # Preserve experiment F: global context changes only the PSG prior.
        cross_attention_features = rst_descriptors
        if self.add_patch_coord:
            cross_attention_features = cross_attention_features + coord_embs
        context = self.neighbors_cross_attn(
            uav_descriptor, cross_attention_features
        )
        combined = torch.cat((uav_descriptor, context), dim=1)
        position = self.pos_regressor(
            torch.cat((combined, pos_soft_prior), dim=1)
        )
        base_heading = self.dir_regressor(combined)
        return position, base_heading, rst_maps, uav_map


class PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100(
    PARCASGM_v5a_GlobalRST_PosPrior_GPRVH
):
    """Joint-100: train experiment F and H3-D together from initialization."""

    model_name = 'phr5_globalrst_f_gprvh_joint100'
    requires_init_checkpoint = False
    default_optimizer = 'Adam'
    default_learning_rate = 5e-5

    def __init__(self, heading_loss_warmup_epochs=10, **kwargs):
        if heading_loss_warmup_epochs < 0:
            raise ValueError("Heading loss warmup epochs must be non-negative")
        kwargs['freeze_base'] = False
        super().__init__(**kwargs)
        self.model_name = type(self).model_name
        self.heading_loss_warmup_epochs = int(heading_loss_warmup_epochs)

    def heading_loss_scale(self, epoch):
        if self.heading_loss_warmup_epochs == 0:
            return 1.0
        return min(
            1.0,
            float(epoch + 1) / float(self.heading_loss_warmup_epochs),
        )


class RSBlockDatasetPA_v3q(Dataset):
    """
    Remote sensing data processing class and its processing pipeline design
    # Prerequisites:
        # - PATCH_SIZE
        # - transform_pipeline1_gentle()
        # - transform_pipeline3()
        TPP1 = (Random color jitter + Random Gaussian noise + Random Gaussian blur + Random cutout)
        TPP1_gentle = Gentle version of TPP1, consistent but with compressed params and reduced probability
        TPP2 = (Random affine transform + Perspective angle)
        TPP3 = (ToTensor+Normalize)
    """
    def __init__(self, metadata_csv: str, is_train: bool = True):
        self.df = pd.read_csv(metadata_csv)
        self.is_train = is_train  #True by default except training, False for other scenarios

        # Separate tile and uav for uav weather augmentation
        self.tile_transform = transforms.Compose([
            # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
            transform_pipeline1_gentle(),
            transform_pipeline3()
        ])
        # Training/validation: uav also uses gentle (TPP2 abandoned in v3q)
        self.uav_transform = transforms.Compose([
            # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
            transform_pipeline1_gentle(),
            # transform_pipeline_weather(),  #Note: add weather augmentation pipeline later
            transform_pipeline3()
        ])

        # Non-training phase: Minimal transform (ToTensor+Normalize)
        self.test_transform = transform_pipeline3()

        # Column name config (corresponding to column names in metadata.csv)
        self.tile_patch_cols = ['p1_path', 'p2_path', 'p3_path', 'p4_path']
        self.uav_col = 'target_path'

    def __len__(self):
        return len(self.df)

    @staticmethod
    def load_cvimg_to_rgb_pil(path: str) -> Image.Image:
        """Read image with cv2 and convert to RGB PIL.Image with fault tolerance."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Fail to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def prepare_patch(self, path: str, is_uav: bool) -> torch.Tensor:
        """
        Read -> RGB(PIL) -> Select appropriate pipeline -> Return tensor(C,H,W)
        Training:    base = base_transform；uav = uav_transform
        Non-training:  base/uav = test_transform
        """
        img = self.load_cvimg_to_rgb_pil(path)
        if self.is_train:
            tfm = self.uav_transform if is_uav else self.tile_transform
        else:
            tfm = self.test_transform
        return tfm(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 4 base patches
        patches = [self.prepare_patch(row[col], is_uav=False) for col in self.tile_patch_cols]
        # 1 uav/target patch
        uav_patch = self.prepare_patch(row[self.uav_col], is_uav=True)
        patches.append(uav_patch)

        patches_tensor = torch.stack(patches, dim=0)  # [5, C, H, W]

        sample = {
            'patches': patches_tensor,
            'coords': torch.tensor([row['x_norm'], row['y_norm']], dtype=torch.float32),
            'ccs_coords': torch.tensor([row['x_uccs'], row['y_uccs']], dtype=torch.float32),
            'agl_coords': torch.tensor([row['x_cosa'], row['y_sina']], dtype=torch.float32),
            'theta': torch.tensor([row['theta']], dtype=torch.float32),
            'block_xy': torch.tensor([row['block_x'], row['block_y']], dtype=torch.int32),
            'target_path': row[self.uav_col],
            # Additional info (default value if CSV has no corresponding column)
            'pyramid':   row.get('pyr', 1),
            'alt_norm':  row.get('alt_norm', 1.0),
            'alt': row.get('alt', 200.0),
            'fov': row.get('fov', 64.0),
        }
        return sample

    # Convenience: External processes (navigation/feature table creation) quickly get paths of 5 images
    def get_patch_paths(self, idx: int) -> List[str]:
        row = self.df.iloc[idx]
        return [row[c] for c in self.base_patch_cols] + [row[self.uav_col]]

class RSBlockDatasetPA_v3q_weather(RSBlockDatasetPA_v3q):
    """Weather augment dataset."""
    def __init__(
        self,
        augname: str,
        metadata_csv: str,
        is_train: bool = True,
        apply_weather_eval: bool = True,
    ):
        super().__init__(metadata_csv, is_train)
        self.apply_weather_eval = apply_weather_eval

        self.eval_uav_transform = transforms.Compose(
            [
                # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transform_pipeline1_gentle(),
                transform_pipeline_weather(augname),
                transform_pipeline3(),
            ]
        )

        self.uav_transform = transforms.Compose(
            [
                # transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transform_pipeline1_gentle(),
                transform_pipeline_weather(augname),
                transform_pipeline3(),
            ]
        )

    def prepare_patch(self, path: str, is_uav: bool) -> torch.Tensor:

        img = self.load_cvimg_to_rgb_pil(path)
        if self.is_train:
            tfm = self.uav_transform if is_uav else self.tile_transform
        else:
            if is_uav and self.apply_weather_eval:
                tfm = self.eval_uav_transform
            else:
                tfm = self.test_transform
        return tfm(img)


def par_dataloader(metadata_csv, dataset_class, dataset_kwargs, BATCH_SIZE):
    """
    PAR dataset preprocessing: Load--Augment--Split--Output train/val/test subsets
    """
    # Load full dataset
    if dataset_class == RSBlockDatasetPA_v3q_weather:
        print(dataset_kwargs["augname"]) #Weather augmentation modes: Mixed, individual Rain,Snow,Cloud,Brightness,Noop, see transform_pipeline_weather
        augm_dataset = dataset_class(
            augname=dataset_kwargs["augname"], metadata_csv=metadata_csv, is_train=True
        )
        norm_dataset = dataset_class(
            augname=dataset_kwargs["augname"], metadata_csv=metadata_csv, is_train=False
        )
    else:
        augm_dataset = dataset_class(
            metadata_csv = metadata_csv,
            is_train=True
        )
        norm_dataset = dataset_class(
            metadata_csv = metadata_csv,
            is_train=False
        )

    # Split dataset
    train_size = int(0.85 * len(augm_dataset))
    val_size = int(0.05 * len(augm_dataset))
    test_size = len(augm_dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(
        range(len(augm_dataset)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  #If changed, test samples cannot be guaranteed to be unseen
        # generator=torch.Generator().manual_seed(11)
    )
    
    # Create training set (use full augmentation)
    train_dataset = torch.utils.data.Subset(augm_dataset, train_indices.indices)
    
    # Create validation and test sets (only use general_transform)
    val_dataset = torch.utils.data.Subset(norm_dataset, val_indices.indices)
    test_dataset = torch.utils.data.Subset(norm_dataset, test_indices.indices)
    
    # Create data loaders
    num_workers = 8  # Or 8, depending on machine CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,   # ★ Reuse workers to avoid restart per epoch
        prefetch_factor=2,         # Default 2 is fine, adjust if needed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, test_loader, test_dataset
    
def load_config_and_model(model_config_dir):
    """Before loading pth for navigation simulation, this program can automatically load
    PAR trained weight files according to the specified path,
    and automatically match the corresponding model keyword dictionary to prevent manual configuration errors
    Because model keywords are saved in training_configure.json during PAR model training
    """
    # Build full config file path
    config_path = os.path.join(model_config_dir, "training_configure.json")
    
    # Read JSON file to config, get model keyword and model_class name from the dictionary
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract model_kwargs
    model_kwargs = config.get("model_kwargs", {})
    
    # Dynamically import model_class
    model_class_name = config.get("model_class")
    if not model_class_name:
        raise ValueError("|ERR| 'model_class' not found in config")
    else:
        print(f"|TIPS| Now You are loading model {model_class_name}.")
    
    # Assume model class is in 'models' module (adjust according to actual situation)
    try:
        # Replace 'your_module' with actual module name containing model class, e.g., 'models'
        # model_module = import_module("posaglreg_models")  
        model_module = import_module("cvphr.models.posaglreg.models")  #dynamic import
        # Note: The above line will error if "posaglreg_models.py" module is renamed in the future!!!
        model_class = getattr(model_module, model_class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import {model_class_name}: {str(e)}")
    
    # Output results
    print("Model Class:", model_class)
    print("Model Kwargs:", model_kwargs)
    
    return model_class, model_kwargs


"""****************************************************************************
*                                                                             *
*                                 CVPHR-net input                               *
*                                                                             *
****************************************************************************"""

model_kwargs_par_ca_sgm_v5a={
    'backbone_name': 'vgg16',
    'feature_dim': 256,                 # d=256
    'coord_enc_dims': [16, 64, 256],    # 2 >> nc*[16, 64, d] 
    'regressor_dims': [1024, 256, 64],  #nc*2*d  >> [1024, 256, 64] >> 2
    'reduction_ratio': 1,
    'num_clusters': 4,
    'freeze_backbone': True,
    'partial_unfreeze':False,
    'add_patch_coord':True, 
}

model_kwargs_par_ca_sgm_v5a_globalrst = {
    **model_kwargs_par_ca_sgm_v5a,
    'global_token_grid_size': 4,
    'global_num_heads': 8,
    'global_feedforward_dim': 512,
    'global_dropout': 0.1,
}

model_kwargs_par_ca_sgm_v5a_globalrst_aux = {
    **model_kwargs_par_ca_sgm_v5a_globalrst,
    'quad_loss_weight': 0.05,
    'attention_loss_weight': 0.10,
    'auxiliary_warmup_epochs': 5,
    'attention_temperature': 1.0,
}

model_kwargs_par_ca_sgm_v5a_globalrst_quad = {
    **model_kwargs_par_ca_sgm_v5a_globalrst,
    'quad_loss_weight': 0.05,
    'attention_loss_weight': 0.0,
    'auxiliary_warmup_epochs': 5,
    'attention_temperature': 1.0,
}

model_kwargs_par_ca_sgm_v5a_globalrst_attn = {
    **model_kwargs_par_ca_sgm_v5a_globalrst,
    'quad_loss_weight': 0.0,
    'attention_loss_weight': 0.10,
    'auxiliary_warmup_epochs': 5,
    'attention_temperature': 1.0,
}

model_kwargs_par_ca_sgm_v5a_mspcoc = {
    **model_kwargs_par_ca_sgm_v5a,
    'heading_feature_dim': 64,
    'heading_radial_bins': 8,
    'heading_angle_bins': 72,
    'heading_crop_scales': (0.75, 1.0, 1.25),
    'heading_base_radius': 0.5,
    'heading_num_heads': 4,
    'heading_dropout': 0.1,
    'heading_temperature': 0.1,
    'heading_scale_temperature': 0.2,
    'heading_gate_bias': -3.0,
    'heading_kappa': 32.0,
    'heading_dist_weight': 1.0,
    'heading_corr_weight': 0.5,
    'heading_final_weight': 0.5,
    'freeze_base': True,
}

model_kwargs_par_ca_sgm_v5a_gprvh = {
    **model_kwargs_par_ca_sgm_v5a,
    'heading_feature_dim': 64,
    'heading_spatial_size': 8,
    'heading_candidate_grid_size': 9,
    'heading_candidate_radius': 0.5,
    'heading_angle_bins': 36,
    'heading_crop_half_extent': 0.5,
    'heading_orientation_temperature': 0.1,
    'heading_translation_temperature': 0.2,
    'heading_residual_limit_degrees': 5.0,
    'heading_position_sigma': 0.15,
    'heading_kappa': 20.0,
    'heading_volume_weight': 1.0,
    'heading_circle_weight': 0.5,
    'heading_phase2_weight': 0.15,
    'heading_opposite_weight': 0.1,
    'heading_opposite_margin': 0.2,
    'freeze_base': True,
}

model_kwargs_par_ca_sgm_v5a_globalrst_posprior_gprvh_s1 = {
    **model_kwargs_par_ca_sgm_v5a_globalrst,
    **{
        key: value
        for key, value in model_kwargs_par_ca_sgm_v5a_gprvh.items()
        if key not in model_kwargs_par_ca_sgm_v5a
    },
}

model_kwargs_par_ca_sgm_v5a_globalrst_posprior_gprvh_joint100 = {
    **model_kwargs_par_ca_sgm_v5a_globalrst_posprior_gprvh_s1,
    'freeze_base': False,
    'heading_loss_warmup_epochs': 10,
}

model_kwargs_par_ca_sgm_v5a_msrdcp_p5 = {
    **model_kwargs_par_ca_sgm_v5a_globalrst,
    'dense_feature_dim': 64,
    'dense_num_rotations': 16,
    'dense_output_size': 32,
    'dense_scale_sizes': (16, 24, 32),
    'dense_template_sizes': (6, 8, 10),
    'dense_rotation_temperature': 0.15,
    'dense_heatmap_temperature': 0.10,
    'dense_gate_bias': -2.0,
    'dense_refine_radius': 0.125,
    'dense_heatmap_sigma': 0.08,
    'dense_heatmap_loss_weight': 0.30,
    'dense_coarse_loss_weight': 0.20,
    'dense_refine_loss_weight': 0.10,
    'freeze_base': True,
}

model_kwargs_par_ca_sgm_v5a_globalrst_posprior_lcpr1 = {
    **model_kwargs_par_ca_sgm_v5a_globalrst,
    'lcpr_feature_dim': 64,
    'lcpr_radius': 0.15,
    'lcpr_temperature': 0.1,
    'lcpr_gate_bias': -2.0,
}


"""****************************************************************************
*                                                                             *
*                            CVPHR-model-dictionary                           *
*                                                                             *
****************************************************************************"""
# Class dictionary of model
MODEL_CLASS_DICT = {
    "PARCASGM_v5":          PARCASGM_v5,
    "PARCASGM_v5a":         PARCASGM_v5a,
    "PARCASGM_v5a_H1":      PARCASGM_v5a_H1,
    "PARCASGM_v5a_MSPCOC":  PARCASGM_v5a_MSPCOC,
    "PARCASGM_v5a_GPRVH":   PARCASGM_v5a_GPRVH,
    "PARCASGM_v5a_GlobalRST_PosPrior_GPRVH": PARCASGM_v5a_GlobalRST_PosPrior_GPRVH,
    "PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100": PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100,
    "PARCASGM_v5a_GlobalRST_PosPrior_LCPR1": PARCASGM_v5a_GlobalRST_PosPrior_LCPR1,
    "PARCASGM_v5a_MSRDCP_P5": PARCASGM_v5a_MSRDCP_P5,
    "PARCASGM_v5a_GlobalRST": PARCASGM_v5a_GlobalRST,
    "PARCASGM_v5a_GlobalRST_PosPrior": PARCASGM_v5a_GlobalRST_PosPrior,
    "PARCASGM_v5a_GlobalRST_Quad": PARCASGM_v5a_GlobalRST_Quad,
    "PARCASGM_v5a_GlobalRST_Attn": PARCASGM_v5a_GlobalRST_Attn,
    "PARCASGM_v5a_GlobalRST_Aux": PARCASGM_v5a_GlobalRST_Aux,
}

MODEL_KEYWARDS_DICT = {
    "PARCASGM_v5":          model_kwargs_par_ca_sgm_v5a,
    "PARCASGM_v5a":         model_kwargs_par_ca_sgm_v5a,
    "PARCASGM_v5a_H1":      model_kwargs_par_ca_sgm_v5a,
    "PARCASGM_v5a_MSPCOC":  model_kwargs_par_ca_sgm_v5a_mspcoc,
    "PARCASGM_v5a_GPRVH":   model_kwargs_par_ca_sgm_v5a_gprvh,
    "PARCASGM_v5a_GlobalRST_PosPrior_GPRVH": model_kwargs_par_ca_sgm_v5a_globalrst_posprior_gprvh_s1,
    "PARCASGM_v5a_GlobalRST_PosPrior_GPRVH_Joint100": model_kwargs_par_ca_sgm_v5a_globalrst_posprior_gprvh_joint100,
    "PARCASGM_v5a_GlobalRST_PosPrior_LCPR1": model_kwargs_par_ca_sgm_v5a_globalrst_posprior_lcpr1,
    "PARCASGM_v5a_MSRDCP_P5": model_kwargs_par_ca_sgm_v5a_msrdcp_p5,
    "PARCASGM_v5a_GlobalRST": model_kwargs_par_ca_sgm_v5a_globalrst,
    "PARCASGM_v5a_GlobalRST_PosPrior": model_kwargs_par_ca_sgm_v5a_globalrst,
    "PARCASGM_v5a_GlobalRST_Quad": model_kwargs_par_ca_sgm_v5a_globalrst_quad,
    "PARCASGM_v5a_GlobalRST_Attn": model_kwargs_par_ca_sgm_v5a_globalrst_attn,
    "PARCASGM_v5a_GlobalRST_Aux": model_kwargs_par_ca_sgm_v5a_globalrst_aux,
}

DATASET_CLASS_DICT = {
    "RSBlockDatasetPA_v3q":         RSBlockDatasetPA_v3q,
    "RSBlockDatasetPA_v3q_weather": RSBlockDatasetPA_v3q_weather,
}
