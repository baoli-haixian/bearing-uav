# Experiment F + CSME

## Scope

Independent model: `PARCASGM_v5a_GlobalRST_PosPrior_CSME`.
Run prefix: `phr5_f_csme`. No CDM or P-RMC is included.
The original F, GLUF, dataset, and training implementation are unchanged.

CSME adds content-aware spatial moments to the four RST descriptors used by CA.
F's global fusion and PSG still receive the original descriptors and feature maps.
RCE is retained. UAV descriptors and the official prediction heads are retained.
Both position and heading losses can train CSME through the shared CA.

## Method

For each RST, recompute the existing GLUF aggregation weights using its own
`conv_node` and normalized centroids: cluster soft assignment times nonnegative
cosine similarity. No second backbone or independent assignment network is used.
Normalize weights across spatial locations, then calculate mean x/y, variance x/y,
and covariance xy. Coordinates are feature-cell centers in [-1, 1], x right and
y down, local to each tile. They are not geographic positions or yaw labels.

A shared MLP (5 -> 32 -> 256) maps moments into each cluster descriptor.
Fuse with `normalize(descriptor + scale * MLP(moments))`, flatten the four
clusters and normalize again. Then add the original RCE and pass to original CA.
The module adds 8,641 parameters at feature_dim=256, including one scalar scale.
Scale starts at zero; MLP weights use standard random initialization. Initially
the scale receives task gradients; the MLP receives gradients once scale moves
away from zero. Empty clusters (mass <= 1e-6) are masked and unchanged.

Loss remains `0.8 * SmoothL1(position) + 0.2 * SmoothL1(heading)`.
No auxiliary labels or losses are introduced. Spatial moments do not guarantee
rotation equivariance or improved metrics, and can compress multimodal layouts.

## Environment and data

Use the same Python/PyTorch environment, requirements, ImageNet VGG-16 cache,
and Bearing-UAV-90K data as experiment F. The repository requirements and main
README remain the source for installation and author-provided download links.
Run from the repository root with `Bearing_UAV_90K` available there. Do not
initialize from a finished F checkpoint for a matched 100-epoch comparison.
VGG ImageNet initialization follows the repository default; it is distinct from
resuming a Bearing-UAV checkpoint.

## Train (commands only; not launched during implementation)

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_CSME \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 100 --factor_bslr 0.5 --device_id 0 \
  --gcth none --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1
```

Use `--is_3d 0` for an independent 2D run. In the current trainer, factor_bslr=0.5
gives batch_size=16 and learning_rate=5e-5. Check the saved training_configure.json
before interpreting results. Keep optimizer, seed, checkpoint selection and data
split identical to the F comparison. The current repository split is 85/5/10,
not the paper's stated 70/20/10. Do not label this an exact paper protocol.
`--flag_test 1` uses the existing post-training evaluation path. Config-based
model restoration is registered for existing testing/navigation loaders.

## Local synthetic tests

```bash
python -m unittest tests.test_experiment_f_csme
```

Tests cover known moments, zero clusters, initial equivalence with F, unchanged
PSG at nonzero scale, finite gradients, model output shape and config/checkpoint
restoration. These do not establish real-dataset accuracy or training speed.

## Evaluation and research boundaries

Compare F versus F+CSME on MLE, LSR@15, Recall@1, MHE and HSR@15 using identical
protocols. Only this five-moment variant is implemented; centroid-only and shuffled
coordinate controls are future ablations, not additional registered experiments.
No server training is needed to create or push this implementation.

References motivating aggregation and spatial encoding (not identical modules):

- SALAD, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.pdf
- Learnable Query Aggregation with KV Routing, 2025 preprint: https://arxiv.org/pdf/2512.23938
- RoPE-ViT, ECCV 2024: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01584.pdf

Spatial moments themselves are classical statistics, not a new mathematical
operation. The hypothesis tested here is their usefulness as cluster-aligned
tile-internal geometry alongside the original RCE and F.
