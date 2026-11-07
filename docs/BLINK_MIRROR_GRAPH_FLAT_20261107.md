# BlinkCore flat task: Mirror50 + custom CUDA Graph

Experiment: `BlinkCore-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM1024-Structured-Muon-EMA0995-Mirror50-CUDAGraph`.

Control: `BlinkCore-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM1024-Structured-Muon-EMA0995`.

Both use flat ground, blind 384-D actor, shared conditional AMP discriminator,
stratified resets, speed-command tracking and recovery, 4096 environments,
2.5 real steps per iteration, 1024×3 structured WM, 5 WM batches,
32-step imagination, Muon hidden matrices + Adam, actor EMA 0.995.
Fresh seed 42, target 40,000 iterations, save every 1,000.
Forward command caps 1/2/3 m/s at iterations 0/5k/15k.

## Changes and provenance

Reference: `fuyichen0725/blinkmimic`, commit
`d264658d6c9ce61fb52f9f0f8985a369f3ccc409`, `mirror.py`, `train.Graphed`, `optim.py`.
Muon implementation is vendored with its MIT license in `third_party/blinkmimic_core`.

- Mirror half the sampled WM windows, maintaining batch size. Reflect world y,
  swap left/right joints and negate roll/yaw, reflect angular velocity as an
  axial vector, mirror actions, encoder bias and lateral torso COM; preserve
  scalar shared friction, validity masks and the three symmetric aggregate
  auxiliary costs. No generated samples are inserted in real replay.
- Explicit CUDA graphs cover separate WM and actor+critic updates, including
  backward and optimizer steps. Three ordinary warmup calls count as training;
  capture then replays exactly once. Graph outputs are cloned before reuse.
- Replay sampling and expert-condition selection happen outside capture;
  every invocation copies fresh values into fixed input buffers. Simulator
  collection and discriminator updates remain outside these graphs.
- Use device-tensor learning rates and upstream capturable MuonAdam. Preserve
  normalization buffer addresses with in-place updates; otherwise capture
  would freeze old variances. Finite checks are performed outside replay;
  discriminator input guards use asynchronous checks in this opt-in mode.
- Capture status is logged as `graph/wm_captured` and `graph/policy_captured`.
  `time/policy_update_wall` measures the full synchronous policy update; old
  component timings from Python during capture are deliberately omitted.

This is a task adaptation, not a full original BlinkMimic reproduction: actor
architecture, command/recovery reward, resets and expert discriminator remain
our task's. No reference-action skip, native motion tracking/probe reset or
native simulator-collection graph is claimed.

## Validation and execution

- Mirror involution for state/action/DR, invariant scalar labels and masks.
- GPU graph regression against ordinary execution with changing input,
  normalization statistics and tensor learning rate.
- Upstream Muon checks on the target server (including capture and finite gates):
  all pass; reported numerical difference within its ordinary-run noise floor.
- Startup verified default pose/action-scale symmetry. FK mirror maximum position
  error was 9.91e-6 m; rotation-matrix error was zero.
- 256-env 12-iteration smoke passed with both graphs captured and replayed.
- Five existing controlled-discriminator regression tests pass.
- 4096-env 12-iteration smoke also passed, with both captured flags equal to 1.
  The launcher then started a separate fresh 40k run (normal 300-iteration WM warmup).

Server: `ssh -p 17535 root@192.168.11.91`, GPU 5.
Snapshot: `/root/workplace/lyd-blink-mirror-graph-flat-20261107`.
Launcher: `scripts/experiments/launch_blink_mirror_graph_flat.py`.
Records: `<snapshot>/records/<full-experiment-name>`.
Server W&B remains offline; local relay uploads the same JSONL metrics.
W&B: https://wandb.ai/tabletennis/lyd-multi-amp/runs/eosxln9r
Code branch: `lyd-supermimic-flat-speed-controls`; implementation commit `7de049a`.

This combined arm measures the package's learning and throughput. It cannot
separately attribute a quality change to mirror versus graph; that would require
a mirror-only control. CUDA Graph is intended to reduce launch overhead, not to
change the task objective.
