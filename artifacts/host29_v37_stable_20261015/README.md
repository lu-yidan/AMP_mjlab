# HoST 29-DoF stable standing model — 2026-10-15

This directory contains the single selected 29-DoF standing checkpoint. The
intermediate checkpoints, training logs, evaluation videos, and temporary
artifacts are intentionally excluded from Git.

- Model: `model_100.pt`
- Source run: `logs/host29_posture_v37_4096_u300_20261014/model_100.pt`
- SHA256: `6317a1990393bcb603cec449042f30302b71a70a25e50a894e7af57e41945e09`
- Size: `6,970,878` bytes
- Selection: update 100, preferred over the final checkpoint

The 256-environment, 1000-step, seed-42 evaluation achieved 100% held and
final standing for prone, supine, left-side, and right-side starts, with zero
terminations. Final joint-speed RMS was `0.264 / 0.199 / 0.173 / 0.215`,
standing yaw-rate RMS was `0.436 / 0.378 / 0.360 / 0.385`, and standing
horizontal-speed RMS was `0.117 / 0.103 / 0.094 / 0.093`.

The checkpoint is the current stable source for V38 standing refinement and
the V15 A6 adapter run. It improves post-settle motion while retaining the
inherited leaning, split stance; it is not claimed to be a straight-standing
solution.
