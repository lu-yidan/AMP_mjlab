# 29-DoF HoST A6 V20 final report

## Scope

V20 is a 29-DoF HoST A6 adapter diagnostic, not the complete paper method and
not a formal-scale `0.25` result. It tests whether a scene-isolated residual can
improve obstacle escape while preserving the frozen V38 flat controller and
remaining active after escape to optimize the formal standing predicate.

Frozen base checkpoint:

```text
logs/host29_posture_v38_4096_u200_20261015/model_175.pt
SHA256 68089b0dd13c13e01f0bd54cab6a4d41d1a5d470e474c6a7d932b4f2e201e03b
```

The base action scale is `0.9323874`. The residual scale is `0.25`; the base
actor and its observation normalizer are frozen. The residual is exactly zero
on flat and is applied in obstacle scenes from motor-on through the remainder
of the episode, including after confirmed escape.

## Training

G- and G+ each completed 1,600 updates with 4,096 environments, 1,150-step
episodes, std `0.04`, learning rate `1e-5`, and two PPO epochs. Both runs
finished without OOM, NaN/Inf, abnormal termination, or unexpected exit.
Final mean rewards were `13.25` for G- and `14.02` for G+.

The 64-environment preflights verified finite rewards and gradients, exact
zero applied residual on flat, and the gate truth table
`[flat=false, active=true, escaped=true, motor-off=false]`.

## Flat preservation

Every evaluated checkpoint (`u200/u400/u800/u1200/u1600`) in both groups kept
`128/128` flat-near environments, 100% final retained, and exactly zero flat
applied residual. Mean retained-step fraction stayed between `98.21%` and
`98.23%`. The scene-isolation contract therefore passed.

## Native-scale all-low diagnostic

The table reports escaped environments out of 256; the scene split is
128 flat, 64 vertical plate, and 64 free plate. Every checkpoint had zero
abnormal terminations.

| group | update | primary | escaped total | vertical | free |
| --- | ---: | ---: | ---: | ---: | ---: |
| G- | 200 | 0/256 | 42/256 | 7/64 | 35/64 |
| G- | 400 | 0/256 | 41/256 | 7/64 | 34/64 |
| G- | 800 | 0/256 | 36/256 | 4/64 | 32/64 |
| G- | 1200 | 0/256 | 36/256 | 4/64 | 32/64 |
| G- | 1600 | 0/256 | 42/256 | 8/64 | 34/64 |
| G+ | 200 | 0/256 | 48/256 | 13/64 | 35/64 |
| G+ | 400 | 0/256 | 43/256 | 5/64 | 38/64 |
| G+ | 800 | 0/256 | 36/256 | 3/64 | 33/64 |
| G+ | 1200 | 0/256 | 38/256 | 3/64 | 35/64 |
| G+ | 1600 | 0/256 | 35/256 | 6/64 | 29/64 |

The frozen native-scale baseline previously produced `33/256` escapes in each
group. Because MuJoCo GPU contact evaluation is not bitwise deterministic, the
selected checkpoints and baselines were rerun with added post-escape metrics:

| policy | escaped | vertical | free | final head | max head | final upright | min foot load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G- base | 34/256 | 7/64 | 27/64 | 1.090 m | 1.099 m | 0.972 | 139.0 N |
| G- u200 | 40/256 | 6/64 | 34/64 | 1.091 m | 1.100 m | 0.972 | 144.6 N |
| G- u1600 | 38/256 | 5/64 | 33/64 | 1.093 m | 1.103 m | 0.971 | 146.0 N |
| G+ base | 33/256 | 4/64 | 29/64 | 1.090 m | 1.101 m | 0.972 | 134.9 N |
| G+ u200 | 42/256 | 6/64 | 36/64 | 1.090 m | 1.102 m | 0.972 | 146.8 N |

The repeated diagnostics preserve the direction of the escape improvement,
especially on the free plate, but show no post-escape standing improvement.
For all escaped trials, mean maximum head height stayed near `1.10 m`; the
height-plus-upright hold above the formal `1.15 m / 0.90` thresholds was
exactly `0 s`. Uprightness and bilateral support load were already sufficient.

## Selection and conclusion

The retained diagnostic checkpoints are:

```text
G- logs/a6_29_v38_v20r1_native_postescape_gminus_4096_u1600_20261017/model_200.pt
   SHA256 c5222fe931f2e9a6a6f4479acb18ef414de68c66279206e86e7c520015cd6924
G+ logs/a6_29_v38_v20r1_native_postescape_gplus_4096_u1600_20261017/model_200.pt
   SHA256 330ba09fe1ce0bdb0ef8cb6a57c5e9850b6188bb4f48f3d6f9133924da4cb968
```

V20 succeeds at preserving flat behavior and gives a modest escape gain, but
it does not produce any formal primary success. The limiting term is now
isolated: the inherited height rewards are nearly saturated around the V38
`1.09 m` equilibrium, so continuing the same PPO run is not justified. The
next version should preserve scene isolation and the native-scale base while
adding an explicitly post-escape, non-saturated height/formal-stance objective
initialized from the selected early residuals.

No repository model artifact is replaced by V20.
