# Experiment F + CDM: content-direction matching with CA feedback

## Model

`PARCASGM_v5a_GlobalRST_PosPrior_CDM` (`phr5_f_cdm`) extends experiment F
without changing F, P-RMC, or the official model classes. Input is four RSTs
followed by one UAV patch, `[B, 5, 3, 256, 256]`. The output remains
`(position, heading)`, each `[B, 2]`.

The shared GLUF produces five `[B, 256, 16, 16]` local maps. Experiment F
still mosaics the four RST maps and sends global context only to the PSG
position prior. CDM projects each map into 32 content channels and 16 pairs of
direction channels. Cosine content matching forms a soft UAV-to-four-RST
correspondence matrix of `[B, 256, 1024]`. PSG's four tile weights add a soft
log-prior to the matching logits; they are detached for this use, so the CDM
heading loss does not directly update GlobalRST through that prior.

For each RST, soft correspondences pool the RST direction vectors for every
UAV token. Their dot and 2-D cross products form 16 pairs of directional
evidence. A shared projection yields one 128-D relation descriptor per RST.
A weighted sum of those descriptors predicts an auxiliary unit heading
vector. A separate projection returns four 1024-D residuals for the original
RST descriptors. Its learned scale starts at exactly zero. The modified RST
descriptors then receive the **original RCE coordinate embeddings** before
the official CA; RCE itself is unchanged. Official position and heading
regressors remain the final outputs.

This is a learned cross-view direction relation, **not** a guaranteed
rotation-equivariant descriptor or a geometric estimate of physical UAV yaw.
In particular, oblique 3-D views can have parallax and perspective effects.

Training uses the existing dataset position and heading labels:

```text
L = 0.8 * SmoothL1(position, p)
  + 0.2 * [SmoothL1(heading, h) + 0.1 * SmoothL1(aux_heading, h)]
```

The net auxiliary coefficient is 0.02. The training history's legacy
`heading_distribution` series is zero for this model because CDM has no
angle-bin distribution; `heading_final` contains only the main heading
SmoothL1 term. `train_loss_dir`/`val_loss_dir` include the auxiliary term.

## Local tests

```bash
python -m unittest tests.test_experiment_f_cdm -v
```

The tests use synthetic tensors and a fake VGG. They do not read
Bearing-UAV-90K or measure localization accuracy.

## Server prerequisites

All experiment-controlled files and caches must stay under
`/data/xuly/cmq/`. Log in to `ssh xuly@10.6.3.45`, then from that host use
`ssh 12.12.12.1` for the two V100s. Confirm the GPU is idle with
`nvidia-smi` before starting; do not interrupt another process. Use the
repository checkout at `/data/xuly/cmq/bearinguav`, or another checkout
inside `/data/xuly/cmq/` with the same dataset link.

Check the existing data and ImageNet VGG-16 weights:

```bash
cd /data/xuly/cmq/bearinguav
test -s Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
test -s /data/xuly/cmq/experiment_f/bearinguav/.cache/torch/hub/checkpoints/vgg16-397923af.pth
```

The VGG-16 file initializes the frozen official backbone. Do not initialize
from a completed F or P-RMC checkpoint for the fair 100-epoch comparison.
The training code keeps the current repository's 85/5/10 split and seed 42.
That is **not** the 70/20/10 split stated in the paper, so comparisons to
the paper table need that protocol difference reported explicitly.

## Train

Use an idle physical GPU number in place of `0`. Inside the tmux session,
set a task-owned runtime directory under `/data/xuly/cmq/` so caches and
temporary files stay in scope:

```bash
cd /data/xuly/cmq/bearinguav
mkdir -p /data/xuly/cmq/experiment_f_cdm_runtime/{home,tmp,cache,pycache,matplotlib,logs}
export HOME=/data/xuly/cmq/experiment_f_cdm_runtime/home
export TMPDIR=/data/xuly/cmq/experiment_f_cdm_runtime/tmp
export XDG_CACHE_HOME=/data/xuly/cmq/experiment_f_cdm_runtime/cache
export PYTHONPYCACHEPREFIX=/data/xuly/cmq/experiment_f_cdm_runtime/pycache
export MPLCONFIGDIR=/data/xuly/cmq/experiment_f_cdm_runtime/matplotlib
export TORCH_HOME=/data/xuly/cmq/experiment_f/bearinguav/.cache/torch
export PYTHONPATH=/data/xuly/cmq/bearinguav
export HISTFILE=/dev/null
conda activate bearing_env
tmux new -s bearing_f_cdm_3d
```

Once inside that tmux session, run:

```bash
cd /data/xuly/cmq/bearinguav
CUDA_VISIBLE_DEVICES=0 python -u -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_CDM \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 --rsi_id 96 --n_sample 100 \
  --num_epochs 100 --factor_bslr 0.5 \
  --learning_rate 5e-5 --optimizer Adam \
  --device_id 0 --gcth none \
  --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1 \
  2>&1 | tee /data/xuly/cmq/experiment_f_cdm_runtime/logs/train_3d.log
```

This explicitly sets batch size 16 via `factor_bslr=0.5`, learning rate
`5e-5`, 100 epochs, Adam, and five-epoch checkpoints. The existing loader
uses eight workers and prefetch factor two. Detach tmux with `Ctrl+B`, `D`.
Reconnect with `tmux attach -t bearing_f_cdm_3d`. Output checkpoints and
test metrics are created under this checkout's `results/c4ma/`.

For 2-D, use a separate result/tmux session, replace `--is_3d 1` with
`--is_3d 0`, and verify the 2-D metadata and free GPU first. Compare both
2-D and 3-D Recall@1, LSR@15, HSR@15, MLE and MHE with experiment F and
F+P-RMC. CDM's effectiveness is unverified until those runs complete.
