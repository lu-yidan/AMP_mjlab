#!/usr/bin/env bash
set -euo pipefail

# Keep both policies on one physical GPU and exactly the same unseen resets.
export A6_GPU="${A6_GPU:-2}"
export A6_SEEDS="52 56"
export A6_MAIN_CHECKPOINT=logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt
export A6_PRONE_CHECKPOINT=logs/host_a6_matching_replay_v13_from374_seed42_450/model_00449.pt
export A6_EVAL_NAME=holdout_v17_499_v13_449
bash "$(dirname "${BASH_SOURCE[0]}")/eval_a6_composite_12_server.sh"

export A6_PRONE_CHECKPOINT=logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt
export A6_EVAL_NAME=holdout_v17_499_v18_474
bash "$(dirname "${BASH_SOURCE[0]}")/eval_a6_composite_12_server.sh"
