# 29-DoF HoST A6 V21 final report

## Scope

V21 is a native-scale 29-DoF HoST A6 adapter diagnostic, not the complete
paper method and not a formal-scale `0.25` result. It continues the selected
V20 scene-isolated residuals while adding post-escape objectives intended to
move the robot from the frozen V38 standing equilibrium near `1.09 m` to the
formal `1.15 m` head-height threshold.

The frozen base is V38 update 175 at action scale `0.9323874`. The residual
scale is `0.25`; the base actor and normalizer remain frozen, obstacle residuals
remain active after escape, and the applied residual on flat is exactly zero.

## Training

G- and G+ each completed 1,200 additional updates with 4,096 environments,
1,150-step episodes, std `0.035`, learning rate `7.5e-6`, and two PPO epochs.
The residual actor, critic, and normalizer were restored from the corresponding
V20 update-200 checkpoint with a fresh optimizer. V21 added two bounded terms
only after escape and motor activation:

- head-height progress over `1.04--1.17 m`, pre-dt weight `2.0`;
- joint height/upright/bilateral-support stance, pre-dt weight `3.0`.

Both 64-environment preflights and full runs completed without OOM, NaN/Inf,
velocity termination, or unexpected exit. Final checkpoint hashes are:

```text
G- model_1200_final.pt
   6ffed1ca0d6ca7cb6e5e190beaf625dc514611ac92da127c220eea8b3bfc4655
G+ model_1200_final.pt
   f07067de4caf61cb867ce9e2fc660cb9e1c36555edac4d30702f1e37fd154ec0
```

## Flat contract

Every evaluated V21 checkpoint (`u400/u800/u1200/final`) retained `128/128`
flat-near environments, had 100% final retained, and kept the maximum applied
flat residual exactly zero. Mean retained-step fraction remained between
`98.20%` and `98.23%`. Scene isolation therefore remained correct.

## Native-scale all-low diagnostic

All evaluations used 256 environments, 1,150 steps, seed 42, native action
scale `0.9323874`, residual scale `0.25`, and post-escape residual application.
No checkpoint produced a formal primary success or a nonzero formal-height
hold. The standard sweep had no abnormal termination.

| group | checkpoint | primary | escaped | vertical | free | final head | max head |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G- | u400 | 0/256 | 38/256 | 4/64 | 34/64 | 1.090 m | 1.102 m |
| G- | u800 | 0/256 | 42/256 | 9/64 | 33/64 | 1.094 m | 1.103 m |
| G- | u1200 | 0/256 | 39/256 | 6/64 | 33/64 | 1.092 m | 1.103 m |
| G- | final | 0/256 | 43/256 | 8/64 | 35/64 | 1.092 m | 1.103 m |
| G+ | u400 | 0/256 | 37/256 | 4/64 | 33/64 | 1.091 m | 1.101 m |
| G+ | u800 | 0/256 | 44/256 | 7/64 | 37/64 | 1.091 m | 1.101 m |
| G+ | u1200 | 0/256 | 37/256 | 5/64 | 32/64 | 1.093 m | 1.103 m |
| G+ | final | 0/256 | 35/256 | 5/64 | 30/64 | 1.094 m | 1.105 m |

V21 did not improve the V20 escape best of `42/256` for G- and `48/256` for
G+, and it did not materially change escaped posture. Escaped uprightness
remained about `0.970--0.972`, bilateral support remained ample, and the
maximum height-plus-upright hold above `1.15 m / 0.90` remained exactly zero.

## Residual-response audit

The evaluator was extended to measure post-escape residual magnitude, action
saturation, combined-action clipping, and per-joint residual response. Repeat
comparisons of V20 update 200 and V21 final show that V21 did produce a much
larger residual, but the added effort did not translate into height:

| group | policy | raw residual abs mean | residual saturation | combined clip | max head |
| --- | --- | ---: | ---: | ---: | ---: |
| G- | V20 u200 | 0.0117 | 0.00% | 3.45% | 1.101 m |
| G- | V21 final | 0.0625 | 0.00% | 3.72% | 1.104 m |
| G+ | V20 u200 | 0.0099 | 0.00% | 3.45% | 1.101 m |
| G+ | V21 final | 0.0855 | 0.00% | 3.77% | 1.104 m |

The largest V21 correction was unilateral right-knee extension
(`-0.0529/-0.0781` scaled residual for G-/G+), followed by asymmetric shoulder,
waist, and left-leg corrections. The residual was not saturated and combined
action clipping changed little. The failure is therefore not insufficient
residual range: unconstrained 29-dimensional PPO spent substantially more
action on an asymmetric local response while the frozen base continued to
restore its `1.09 m` posture equilibrium.

## Conclusion

V21 preserves the flat contract but fails its post-escape formal-height
hypothesis. Increasing the same reward weights, residual scale, or run length
is not justified. The next method should keep the V20 recovery residual frozen
and train a separate post-escape posture adapter from escaped-state data, with
an explicitly symmetric low-dimensional sagittal leg action space. That
separates obstacle escape from bilateral leg extension and prevents the
posture objective from rewriting recovery behavior or spending capacity on
unrelated upper-body rows.

No repository model artifact is replaced by V21.
