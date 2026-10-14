#!/usr/bin/env bash
set -euo pipefail

project=/home/czr/code/G1Recovery_HoST
run=${A6_WATCH_RUN:-logs/host_a6_v17_episode15_long_from2399_4096_seed42_to3399}
train_log=${A6_WATCH_LOG:-$run/train_2400_3399.log}
first=${A6_WATCH_FIRST:-2449}
last=${A6_WATCH_LAST:-3399}
cuda=${A6_WATCH_CUDA:-4}
cd "$project"
eval_dir="$run/watch_eval_center15"
mkdir -p "$eval_dir"
summary="$eval_dir/summary.tsv"
if [[ ! -e "$summary" ]]; then
  printf 'iteration\tprone_stood_64\tsupine_stood_64\tprone_invalid_64\tsupine_invalid_64\tforce_trips\tlift_trips\tfree_stood_64\tflat_stood_64\n' > "$summary"
fi

for iteration in $(seq "$first" 50 "$last"); do
  checkpoint=$(printf '%s/model_%05d.pt' "$run" "$iteration")
  until [[ -s "$checkpoint" ]] && grep -Fqx "checkpoint=$checkpoint" "$train_log"; do
    sleep 20
  done
  for seed in 42 43; do
    output="$eval_dir/$(printf 'model_%05d_seed%d.json' "$iteration" "$seed")"
    if [[ ! -s "$output" ]]; then
      env CUDA_VISIBLE_DEVICES="$cuda" \
        MPLCONFIGDIR=/tmp/mpl-host-v17-watch \
        PYTHONPATH="$project/src" \
        /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/eval_a6_batch.py \
        --checkpoint "$checkpoint" --output "$output" \
        --device cuda:0 --seed "$seed" --steps 750 --per-scene 32 \
        --guided-offset-m 0 \
        > "${output%.json}.log" 2>&1
    fi
  done
  if ! grep -q "^${iteration}[[:space:]]" "$summary"; then
    python3 - "$eval_dir" "$iteration" >> "$summary" <<'PY'
import json
import sys
from pathlib import Path

root, iteration = Path(sys.argv[1]), int(sys.argv[2])
scenes = [
    json.loads((root / f"model_{iteration:05d}_seed{seed}.json").read_text())["scenes"]
    for seed in (42, 43)
]


def total(scene, key):
    return sum(item[scene][key] for item in scenes)


fields = (
    iteration,
    total("guided_prone", "escaped_then_stood_at_end"),
    total("guided_supine", "escaped_then_stood_at_end"),
    total("guided_prone", "invalid"),
    total("guided_supine", "invalid"),
    total("guided_prone", "invalid_by_force")
    + total("guided_supine", "invalid_by_force"),
    total("guided_prone", "invalid_by_lift")
    + total("guided_supine", "invalid_by_lift"),
    total("free_supine", "stood_at_end"),
    total("flat_supine", "stood_at_end"),
)
print("\t".join(map(str, fields)))
PY
  fi
  tail -n 1 "$summary"
done
