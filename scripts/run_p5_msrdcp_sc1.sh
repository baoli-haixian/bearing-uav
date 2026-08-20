#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/xuly/cmq/experiment_p5_msrdcp/bearinguav"
F_CHECKPOINT="/data/xuly/cmq/experiment_f/bearinguav/results/c4ma/phr5_globalrst_f_d96100_3d_b16_l0.5_e100_gNone_20260815_120653/best_model.pth"
GPU_ID="${1:-0}"

case "$PROJECT_DIR" in
  /data/xuly/cmq/*) ;;
  *) echo "Refusing to run outside /data/xuly/cmq" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
test -f "$F_CHECKPOINT"
test -f Bearing_UAV_90K/metadata/metadata.csv
mkdir -p log/c4ma

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export BEARINGUAV_OFFLINE_WEIGHTS=1
exec /home/xuly/.conda/envs/bearing_env/bin/python -m cvphr.train.cvphr_train \
  --model_class PARCASGM_v5a_MSRDCP_P5 \
  --dataset_class RSBlockDatasetPA_v3q \
  --is_3d 1 \
  --rsi_id 96 \
  --n_sample 100 \
  --num_epochs 20 \
  --factor_bslr 0.5 \
  --learning_rate 1e-4 \
  --optimizer AdamW \
  --device_id 0 \
  --gcth none \
  --checkpoint_interval 5 \
  --flag_ckpt 1 \
  --flag_test 1 \
  --init_checkpoint "$F_CHECKPOINT"
