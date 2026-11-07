# BlinkCore graph-only control, 2026-11-08

Experiment: `BlinkCore-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM1024-Structured-Muon-EMA0995-NoMirror-CUDAGraph`.

Purpose: separate the custom update-graph acceleration from Mirror50 augmentation.
Compare with the completed BlinkCore base and BlinkCore Mirror50+CUDAGraph runs.
The new `--blink-graph-only` option is mutually exclusive with `--blink-mirror-graph`.
It uses the same capturable optimizers and graph implementation but feeds the WM
unmodified sampled windows (`blink_core.mirror=0.0`). Symmetry startup assertions
remain diagnostic; they do not modify replay, normalization or training inputs.

Same flat command tracking + recovery task, shared conditional discriminator,
blind 384-D actor, stratified reset/expert banks, 4096 envs, seed42, 2.5 real steps,
WM1024x3 with 5 updates, structured drift, Muon/Adam, EMA0.995, horizon32,
300-iteration WM warmup, speed caps 1/2/3m/s at iterations 0/5000/15000.
From scratch; target40k; save1000. User permits extension to50k if40k insufficient.
40k evaluation should use the existing 1268-state recovery bank and12-command
protocol, compare recovery time, acceleration, speed MAE and runtime, not just reward.
No automatic extension or scheduled evaluator has been installed by this launcher.

Server: `ssh -p 17535 root@192.168.11.91`, GPU0.
Snapshot: `/root/workplace/lyd-blink-graph-only-flat-20261108`.
Launcher: `scripts/experiments/launch_blink_graph_only_flat.py`.
The launcher requires a successful12-iteration4096-env preflight with both graph
capture flags before starting the separate fresh40k run.

Validation: six existing mirror/structured-WM/graph tests passed, including GPU
replay with changing normalization, inputs and learning rate. Runtime validation
and W&B link are recorded below after startup.

Runtime preflight passed: 4096 environments,12 iterations,exit0,both graph flags1.
Formal training config confirms mirror0.0 and has begun WM updates.
W&B (local JSONL relay): https://wandb.ai/tabletennis/lyd-multi-amp/runs/whausnhs
Branch: `lyd-supermimic-flat-speed-controls`.
