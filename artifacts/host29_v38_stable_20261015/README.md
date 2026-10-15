# HoST 29-DoF selected standing model — 2026-10-15

This directory contains the repository's single selected 29-DoF standing
checkpoint. Intermediate checkpoints, optimizer snapshots, logs, videos, and
temporary evaluation outputs are intentionally excluded from Git.

- Model: `model_175.pt`
- Source run: `logs/host29_posture_v38_4096_u200_20261015/model_175.pt`
- SHA256: `68089b0dd13c13e01f0bd54cab6a4d41d1a5d470e474c6a7d932b4f2e201e03b`
- Size: `6,970,878` bytes
- Selection: update 175, preferred over the final checkpoint

Two independent 256-environment, 1,000-step evaluations (seeds 42 and 43)
both achieved 100% held and final standing from prone, supine, left-side, and
right-side starts, with zero terminations. Their four-posture aggregate means
were `0.798/0.815` final joint-speed RMS, `0.624/0.634` standing yaw-rate RMS,
`0.1023/0.1020` standing horizontal-speed RMS, and `0.0490/0.0487 m/s` final
root linear speed.

Repeated evaluations of the exact V37 source checkpoint produced aggregate
means around `0.813-0.820` joint-speed RMS, `0.628-0.638` yaw-rate RMS,
`0.102-0.104` horizontal-speed RMS, and `0.049-0.051 m/s` final root speed.
V38 therefore provides only a small but repeatable damping improvement. The
much lower one-off V37 result previously reported could not be reproduced and
is not used for model selection.

The V38 diagnostic and four videos show no knee walking, sustained turn,
periodic direction change, or repeated corrective stepping. The inherited
leaning, split stance and raised right arm remain; this is a stability model,
not a straight-standing solution.
