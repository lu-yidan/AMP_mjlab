# 29-DoF HoST versus A6 root-cause audit

## Conclusion

The persistent `~1.09 m` standing equilibrium is a **HoST base-policy target
mismatch**, not an A6 head-height implementation error and not a flat-scene
side effect of the obstacle adapter.

The A6 adapter reconstructs the authoritative SMP head site as a massless site
at `[0, 0, 0.43] m` on `torso_link`, rotates that offset with the torso, and
measures its world height above the environment origin. The frozen SMP source
uses the same site definition and the same `1.15 m` near-standing boundary.
The two implementations therefore agree geometrically.

The inherited HoST objective is different. It calls `torso_link` the head,
measures torso height relative to the feet, and uses a `0.8 m` soft target with
a broad tolerance. V38 satisfies that HoST predicate and remains dynamically
stable, but its actual SMP/A6 virtual head settles below the A6 threshold.

## Paired V38 audit

V38 update 175 was evaluated from prone, supine, left-side, and right-side
resets for 1,150 control steps with no A6 obstacle and no residual controller.
The added audit metric is exactly the SMP/A6 virtual head-site height.

| Reset posture | Final mean head | Final minimum head | Mean maximum head | Formal contract at final |
|---|---:|---:|---:|---:|
| prone | 1.0921 m | 1.0782 m | 1.1060 m | 0/128 |
| supine | 1.0887 m | 1.0798 m | 1.1074 m | 0/128 |
| left side | 1.0924 m | 1.0785 m | 1.1086 m | 0/128 |
| right side | 1.0908 m | 1.0778 m | 1.1111 m | 0/128 |

This reproduces the low equilibrium on plain flat recovery, without any plate,
A6 reward, residual actor, or action-scale remapping. The failure therefore
precedes the obstacle task.

V17 independently showed that the same 29-DoF plant can reach `1.157 m` while
remaining upright after A6 flat-near adaptation. The `1.15 m` boundary is
physically reachable; the V38 optimization target simply does not select it.

## Diagnostic correction

`scripts/diagnose_host29_posture.py` now reports the same virtual-head metric
and formal contract. Its historical sagittal-target option remains backward
compatible, while a new explicit control-mode flag distinguishes the existing
incremental error controller from an absolute-position diagnostic. This
change affects diagnostics only and does not alter any trained policy or A6
runtime.

## V39 fix

V39 keeps the A6 definition unchanged and fixes the upstream standing source:

- source actor and observation normalizer are exactly V38 update 175 at launch;
- recovery behavior remains unconstrained by the new objective until the robot
  has already reached the inherited standing gate;
- post-standing rewards use the rotated `0.43 m` virtual head site, a `1.17 m`
  training target for margin, uprightness target `0.93`, and the formal
  `1.15 m / 0.90` contract;
- only the final actor layer is trainable, with row-specific hard trust regions;
- two trust-region sizes run in parallel so the selected checkpoint can require
  both four-posture recovery retention and A6-height improvement.

Both 64-environment, 450-step preflights reproduced the unchanged V38 source
at `1.0935 m` mean final head height, activated finite height-progress and
shortfall rewards, produced finite gradients, and passed without NaN/Inf.
Formal training uses new versioned logs and does not overwrite V38 or any A6
checkpoint.

## V39 result

Both 1,200-update V39 runs completed without numerical faults or velocity
terminations. The smaller trust region preserved four-posture recovery at
`128/128` held and final standing, but increased mean final virtual-head height
by only about `3 mm`; its best/final posture means remained
`1.0904--1.0953 m`. The larger trust region did not improve height. Every V39
checkpoint retained zero formal held/final environments.

Checkpoint-delta audit showed that V39 did not hit its output-layer bounds.
The failure was therefore not an overly tight clamp: changing only the final
linear layer did not provide enough state representation or control authority
to move the inherited V38 equilibrium by the required `~60 mm`.

## V40 result

V40 delayed all A6-height credit until 100 consecutive stable standing steps,
extended the episode to 1,150 control steps, and compared bounded adaptation
of the final two linear layers against bounded full-actor adaptation. Both
1,600-update runs completed without NaN/Inf, abnormal termination, or recovery
loss.

The final-two-layer run was effectively flat: all four postures remained at
`1.0900--1.0943 m`, with zero formal held/final environments. Full-actor V40
did move the standing equilibrium. Update 800 was the best stable point:

| Reset posture | V38 final mean head | V40 full u800 | Formal held/final |
|---|---:|---:|---:|
| prone | 1.0921 m | 1.1022 m | 0/128, 0/128 |
| supine | 1.0887 m | 1.0992 m | 0/128, 0/128 |
| left side | 1.0924 m | 1.1020 m | 0/128, 0/128 |
| right side | 1.0908 m | 1.1004 m | 0/128, 0/128 |

Recovery remained `128/128` held and final with zero terminations, but the
approximately `10 mm` gain was still far below the formal boundary. Later
full-actor checkpoints did not add reliable height and increased joint-speed
RMS, so the final checkpoint is rejected. V40 does not replace the selected
V38 artifact.

As a reachability cross-check, the A6-homotopy V17 actor produced nonzero
formal holds (`18.75--24.22%`) when evaluated directly from all four flat
postures, proving again that the plant and metric are feasible. It did not
preserve the V38 recovery contract: final standing fell to `73.44--81.25%`
and joint-speed RMS rose to roughly `2.8`. A hard V38-to-V17 switch after 250
steps likewise lost standing. The next repair must therefore learn or distill
a smooth post-standing transition instead of applying either actor globally.

No high-standing replacement model is selected. V38 remains the sole packaged
artifact, and A6 thresholds remain unchanged. All results remain explicitly
**29-DoF HoST A6 adapter** results.
