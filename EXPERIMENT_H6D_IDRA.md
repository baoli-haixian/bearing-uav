# Experiment H6-D: F + Implicit Directional Relation Adapter

## 1. Purpose

H6-D keeps Experiment F's position path unchanged and adds a lightweight,
parallel heading branch. IDRA learns directional relations from the spatial UAV
and RST feature maps without enumerating angle bins or rotating a feature bank.

The registered model is:

```text
PARCASGM_v5a_GlobalRST_PosPrior_IDRA
```

The result directory prefix is:

```text
phr5_f_h6d_idra
```

## 2. Architecture

The four north-aligned RST feature maps are arranged as a 2x2 mosaic. The RST
mosaic and UAV feature map pass through a shared direction-sensitive adapter:

```text
1x1 projection
  -> parallel depthwise 1x5 / 5x1 / 3x3 convolutions
  -> concatenation + 1x1 fusion + residual
  -> adaptive 8x8 pooling + learned 2-D position embedding
```

Four learned heading queries independently attend to UAV and RST tokens. Their
concatenation, difference and element-wise product form an implicit relation
latent. A small MLP maps the latent to a unit heading vector. A gate initialized
with bias `-2` fuses this prediction with the official heading prediction.

The IDRA output never enters the position regressor. The GlobalRST descriptor is
still used only by Experiment F's PSG position prior.

## 3. Loss

The task loss keeps the official weighting:

```text
L = 0.8 * L_position + 0.2 * L_heading
```

H6-D uses:

```text
L_heading = SmoothL1(normalize(h_final), normalize(h_gt))
            + 0.02 * L_rotation_consistency
```

`L_rotation_consistency` rotates only the UAV feature map by a random non-zero
multiple of 90 degrees and requires the implicit heading vector to rotate by the
same amount. It does not require an additional dataset label.

## 4. Environment

Use the same environment, dataset and VGG-16 initialization as the official
Bearing-UAV experiment. From the repository root:

```bash
conda create -n bearing_env python=3.9 -y
conda activate bearing_env
pip install -r requirements.txt
```

Place the dataset at:

```text
./Bearing_UAV_90K/
```

The expected structure and metadata are unchanged from Experiment F. No new
preprocessing or labels are required.

## 5. Test before training

```bash
python -m unittest tests.test_experiment_h6d_idra -v
```

The test checks model registration, tensor shapes, finite forward/backward,
rotation convention, gate initialization, position equivalence with Experiment
F and task-gradient separation.

## 6. Train from scratch for 100 epochs

For a fair comparison with Experiment A and Experiment F, train the combined
model jointly for 100 epochs rather than adding 100 epochs after an F run:

```bash
python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_IDRA \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 100 \
  --factor_bslr 0.5 \
  --device_id 0 \
  --gcth none \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1
```

This uses the repository defaults already used by Experiment F:

```text
batch_size = 16
learning_rate = 5e-5
optimizer = Adam
position_weight = 0.8
heading_weight = 0.2
epochs = 100
```

## 7. Evaluation

Use the checkpoint selected by validation loss and the repository's normal test
path. Compare H6-D with A and F under the same split and seed. The primary H6-D
metrics are `MHE` and `HSR@15`; also report `Recall@1`, `LSR@15` and `MLE` to
verify that the unchanged Experiment F position path did not regress.

For a strict ablation, keep every command-line argument fixed and change only
`--model_class`.
