#!/usr/bin/env bash
set -euo pipefail

variant=${1:?Pass residual or auxiliary}
gpu=${2:?Pass physical GPU 0 or 1}
case "$variant:$gpu" in
  residual:0|residual:1|auxiliary:0|auxiliary:1) ;;
  *) echo "Invalid variant/GPU: $variant $gpu" >&2; exit 2 ;;
esac

root=/data/xuly/cmq/experiment_f_cdm_ablation
project=$root/bearinguav
torch_home=/data/xuly/cmq/experiment_f/bearinguav/.cache/torch
cd "$project"
test "$(pwd -P)" = "$project"
test "$(readlink -f Bearing_UAV_90K)" = /data/xuly/cmq/Bearing_UAV_90K
test -s "$torch_home/hub/checkpoints/vgg16-397923af.pth"

case "$variant" in
  residual) model_class=PARCASGM_v5a_GlobalRST_PosPrior_CDMResidualOnly ;;
  auxiliary) model_class=PARCASGM_v5a_GlobalRST_PosPrior_CDMAuxOnly ;;
esac

for view in 3d 2d; do
  case "$view" in
    3d) is_3d=1; dataset=c4m_254k_96bc_b15_s100_v3d ;;
    2d) is_3d=0; dataset=c4m_254k_96bc_b15_s100 ;;
  esac
  test -s "Bearing_UAV_90K/$dataset/metadata/metadata.csv"

  gpu_memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  gpu_util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  if (( gpu_memory >= 100 || gpu_util >= 5 )); then
    echo "GPU $gpu occupied before $view: ${gpu_memory}MiB, ${gpu_util}%" >&2
    exit 3
  fi

  runtime=$root/runtime/$variant/$view
  mkdir -p "$runtime"/{tmp,cache,pycache,matplotlib,logs}
  export TMPDIR=$runtime/tmp TMP=$runtime/tmp TEMP=$runtime/tmp
  export XDG_CACHE_HOME=$runtime/cache
  export PYTHONPYCACHEPREFIX=$runtime/pycache
  export MPLCONFIGDIR=$runtime/matplotlib
  export TORCH_HOME=$torch_home
  export PYTHONPATH=$project
  export PYTHONDONTWRITEBYTECODE=1
  export CUDA_VISIBLE_DEVICES=$gpu
  export PYTHONUNBUFFERED=1

  echo "START $(date -Is) host=$(hostname) variant=$variant view=$view gpu=$gpu commit=$(git rev-parse --short HEAD) cwd=$(pwd -P)"
  /home/xuly/.conda/envs/bearing_env/bin/python -u -m cvphr.train.cvphr_train \
    --model_class "$model_class" \
    --dataset_class RSBlockDatasetPA_v3q \
    --is_3d "$is_3d" --rsi_id 96 --n_sample 100 \
    --num_epochs 100 --factor_bslr 0.5 \
    --learning_rate 5e-5 --optimizer Adam \
    --device_id 0 --gcth none \
    --checkpoint_interval 5 --flag_ckpt 1 --flag_test 1 \
    2>&1 | tee "$runtime/logs/train.log"
done
