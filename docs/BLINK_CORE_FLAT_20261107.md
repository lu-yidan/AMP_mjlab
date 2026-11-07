# BlinkMimic core adaptation: velocity commands and recovery

Experiment: BlinkCore-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM1024-Structured-Muon-EMA0995.
Server 192.168.11.91:17535, GPU4. Root /root/workplace/lyd-blink-core-flat-20261107.

This is an explicit partial adaptation of BlinkMimic d264658, not a full port or
native reference-tracking reproduction. It trains our blind 384-input Actor on
flat velocity commands and stratified fall recovery with the shared conditional
AMP discriminator. Same rewards/reset bank/expert bank/speed exposure schedule
as the three flat controls. Fresh initialization, seed42, 4096 environments,
40k iterations, saves every1000. No reference-pose input or action skip.

Adapted components:
- Structured velocity-driven pose drift plus learned residuals, drift weight .625,
  dt .02 seconds. Position, orientation, joints and velocity targets follow the
  upstream formulas. Physical multi-step error uses a separate increment normalizer;
  residual-output normalization must not be used to scale physical error.
- Upstream MuonAdam (vendored MIT LICENSE): hidden weight matrices of Actor and WM
  use Muon, first/last layers and vectors use Adam. Critic stays Adam. No critic
  parameters are accidentally passed to the Actor optimizer. LR4e-4 for Actor/WM.
- WM1024x3, 5 WM updates/iteration, real collection2.5 steps, evaluation EMA.995.

Retained from our task: 32-step imagination, 300-iteration WM warmup, 128-step
replay, 512/256/128 Actor/Critic, sensor-cost auxiliary predictions, shared AMP,
compile/bf16, speed stages capped at1/2/3m/s after0/5000/15000 iterations.
Not ported: mirror augmentation, reference action skip, native custom CUDA graphs,
reference-clip failure resets or adaptive probe-based learning-rate schedules.
Thus this is a multi-component comparison, not a structured-WM-only ablation.

Validation: CPU tests verify target/integrate round trip, finite nonzero gradients,
physical state-error semantics, and zero-residual constant-velocity drift.
A full 4096-env 8-iteration smoke runs before full training. Check records/*/exit.json.
Native BlinkMimic benchmark remains separately located in /root/workplace/blinkmimic-d264658.
