#!/usr/bin/env bash
set -euo pipefail

# Proprio-only v49 routing with a validated 1.10 lateral action gain on
# side-gated worlds. All checkpoints and other action paths remain v49.
exec bash "$(dirname "${BASH_SOURCE[0]}")/play_a6_proprio_composite_v49.sh" \
  --side-lateral-gain 1.10 "$@"
