#!/usr/bin/env bash
set -euo pipefail

project=/home/czr/code/G1Recovery_HoST
run=${A6_WATCH_RUN:-logs/host_a6_v26_allside_from2999_to3999_4096_seed42}
cuda=${A6_WATCH_CUDA:-4}
cd "$project"
eval_dir="$run/watch_eval_all12_center15"
mkdir -p "$eval_dir"
summary="$eval_dir/summary.tsv"
if [[ ! -e "$summary" ]]; then
  printf 'iteration\tguided_prone_success_64\tguided_supine_success_64\tguided_left_success_64\tguided_right_success_64\tfree_prone_success_64\tfree_supine_success_64\tflat_prone_stood_64\tflat_supine_stood_64\tguided_invalid_256\n' > "$summary"
fi

for iteration in 3049 3199 3399 3599 3799 3999; do
  checkpoint=$(printf '%s/model_%05d.pt' "$run" "$iteration")
  until [[ -s "$checkpoint" ]] && grep -Fqx "checkpoint=$checkpoint" "$run/train.log"; do
    if ! pgrep -af 'scripts/train_host.py' | grep -F -- "--output $run" > /dev/null; then
      echo "Training stopped before $checkpoint became ready" >&2
      exit 1
    fi
    sleep 20
  done
  for seed in 42 43; do
    output="$eval_dir/$(printf 'model_%05d_seed%d.json' "$iteration" "$seed")"
    if [[ ! -s "$output" ]]; then
      env CUDA_VISIBLE_DEVICES="$cuda" \
        MPLCONFIGDIR=/tmp/mpl-host-v26-watch \
        PYTHONPATH="$project/src" \
        /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/eval_a6_batch.py \
        --checkpoint "$checkpoint" --output "$output" \
        --all-directions --device cuda:0 --seed "$seed" \
        --steps 750 --per-scene 32 --guided-offset-m 0 \
        > "${output%.json}.log" 2>&1
    fi
  done
  if ! grep -q "^${iteration}[[:space:]]" "$summary"; then
    /home/czr/code/G1Recovery_mjlab/.venv/bin/python - "$eval_dir" "$iteration" >> "$summary" <<'PY'
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
    total("guided_left_side_down", "escaped_then_stood_at_end"),
    total("guided_right_side_down", "escaped_then_stood_at_end"),
    total("free_prone", "escaped_then_stood_at_end"),
    total("free_supine", "escaped_then_stood_at_end"),
    total("flat_prone", "stood_at_end"),
    total("flat_supine", "stood_at_end"),
    sum(
        total(f"guided_{direction}", "invalid")
        for direction in ("prone", "supine", "left_side_down", "right_side_down")
    ),
)
print("\t".join(map(str, fields)))
PY
  fi
  tail -n 1 "$summary"
done
