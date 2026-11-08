# BlinkCore graph-only 40k evaluation — 2026-11-08

Experiment: `BlinkCore-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM1024-Structured-Muon-EMA0995-NoMirror-CUDAGraph`.

40k completed with exit0 in4.3122 training-loop hours; custom update graphs active. Compared with original BlinkCore7.0667h:39.0% less time; Mirror50+Graph4.5450h:5.1% less. Timings were not measured under identical concurrent GPU load, so treat small differences cautiously.

## Matched evaluation

Same1268 recovery starts, initial-state hash and bank SHA256 as the prior five-run comparison.20s without resets; success=root height>0.65m and gz<−0.9 continuously1s. Times include that1s.12 command cases×8 matched perturbations;2s stand then10s command; final5s means. Flat nominal dynamics, EMA0.995 inference.

| BlinkCore variant (all single discriminator,2.5steps,WM1024,structured WM+Muon+EMA) | Hours | Recovery success | Mean success(s) | First3s joint acceleration RMS(rad/s²) |
|---|---:|---:|---:|---:|
| Original | 7.07 | 100% | 1.689 | 81.29 |
| Mirror50 + CUDA Graph | 4.55 | 100% | 1.719 | 98.62 |
| No mirror + CUDA Graph | 4.31 | 100% | 1.780 | 81.56 |

All recovery trials end upright. Four supine initial states take over6s to achieve success (IDs612,1096,1108,1196;11.96,6.78,8.40,6.24s). The recorded continuous-bad-state≥5s incidence is4/1268=0.315%. These are eventually successful trials, not20s failures. The original and mirror comparison had0 such continuous-bad-state cases. Do not report100% alone as equivalent recovery robustness.

## Command tracking

| Case | Target (vx,vy,wz) | Mean actual | MAE | Upright% |
|---|---|---|---|---:|
| stand | [0, 0, 0] | [-0.001, 0.001, 0.04] | [0.002, 0.001, 0.054] | 100.0 |
| forward05 | [0.5, 0, 0] | [0.436, 0.06, -0.003] | [0.066, 0.06, 0.055] | 100.0 |
| forward1 | [1, 0, 0] | [0.947, -0.007, -0.025] | [0.067, 0.065, 0.059] | 100.0 |
| forward2 | [2, 0, 0] | [1.934, -0.025, -0.001] | [0.069, 0.051, 0.045] | 100.0 |
| forward3 | [3, 0, 0] | [2.956, 0.025, -0.005] | [0.054, 0.035, 0.062] | 100.0 |
| backward05 | [-0.5, 0, 0] | [-0.48, 0.014, -0.037] | [0.031, 0.048, 0.06] | 100.0 |
| backward1 | [-1, 0, 0] | [-0.915, 0.017, 0.003] | [0.109, 0.046, 0.073] | 100.0 |
| left05 | [0, 0.5, 0] | [-0.034, 0.498, 0.021] | [0.056, 0.05, 0.058] | 100.0 |
| right05 | [0, -0.5, 0] | [-0.033, -0.493, 0.009] | [0.049, 0.073, 0.061] | 100.0 |
| yaw_left | [0, 0, 1] | [0.035, -0.015, 0.971] | [0.038, 0.025, 0.068] | 100.0 |
| yaw_right | [0, 0, -1] | [0.001, 0.045, -0.969] | [0.016, 0.046, 0.062] | 100.0 |
| forward_turn | [1, 0, 1] | [0.975, -0.119, 0.986] | [0.083, 0.119, 0.076] | 100.0 |

## Interpretation

Graph-only retains the early-recovery acceleration level of the original while offering substantially shorter training time. Reverse and lateral commands are accurate; forward1/2m/s lag slightly. It is an attractive efficiency/quality candidate, not an unconditional winner: recovery has a small slow supine tail, unlike the two previous BlinkCore runs.

This single-seed comparison cannot attribute trajectory differences causally to CUDA Graph; random-number consumption, execution/rounding changes and nonlinear learning can produce different policies. The acceleration proxy is not a visual-naturalness measurement. Current flat success is saturated; no50k continuation was launched as part of this evaluation. A50k extension remains authorized if needed, but improvement of the four slow starts must be measured rather than assumed.

Local data: `logs/eval_graph_only_40k_20261108/` (JSON,NPZ,commands,checkpoint SHA256 and aligned six-run summary).
Server data: `/root/workplace/lyd-blink-mirror-graph-flat-20261107/eval-graph-only-40k-20261108`.
