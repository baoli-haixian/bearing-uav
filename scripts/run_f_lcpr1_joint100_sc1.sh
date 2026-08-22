#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/xuly/cmq/experiment_f_lcpr1_joint100/bearinguav"
GPU_ID="${1:-0}"

case "$PROJECT_DIR" in
  /data/xuly/cmq/*) ;;
  *) echo "Refusing to run outside /data/xuly/cmq" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
test -f Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv
mkdir -p log/c4ma

export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec /home/xuly/.conda/envs/bearing_env/bin/python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_GlobalRST_PosPrior_LCPR1 \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 100 \
  --factor_bslr 0.5 \
  --learning_rate 5e-5 \
  --optimizer Adam \
  --device_id 0 \
  --gcth none \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1
