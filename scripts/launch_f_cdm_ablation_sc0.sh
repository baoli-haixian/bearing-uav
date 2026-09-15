#!/usr/bin/env bash
set -euo pipefail

variant=${1:?Pass residual or auxiliary}
gpu=${2:?Pass physical GPU 0 or 1}
case "$variant:$gpu" in
  residual:0|residual:1|auxiliary:0|auxiliary:1) ;;
  *) echo "Invalid variant/GPU: $variant $gpu" >&2; exit 2 ;;
esac

root=/data/xuly/cmq/experiment_f_cdm_ablation
socket=$root/runtime/tmux/ablation.sock
session=bearing_f_cdm_${variant}
mkdir -p "$root/runtime/tmux"
cd "$root"
test "$(pwd -P)" = "$root"
if /usr/bin/tmux -S "$socket" has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session" >&2
  exit 3
fi

/usr/bin/tmux -S "$socket" new-session -d -s "$session" \
  "ssh -o BatchMode=yes -o ConnectTimeout=8 xuly@12.12.12.1 'cd /data/xuly/cmq/experiment_f_cdm_ablation/bearinguav && /bin/bash scripts/run_f_cdm_ablation_sc1.sh $variant $gpu'"
/usr/bin/tmux -S "$socket" list-sessions
