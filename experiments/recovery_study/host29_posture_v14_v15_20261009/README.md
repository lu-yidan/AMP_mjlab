# 29 DoF posture V14/V15 experiment

## Reward change

V14/V15 replace the saturated exponential hand-position and support-centering
rewards with dense distance costs. V14 continues from V13 with moderate strong
weights; V15 trains from random initialization with intentionally extreme
weights so inherited posture cannot dominate the objective.

## V14 result

- checkpoint: `logs/host29_posture_v14_4096_u2000_20261009/model_2000_final.pt`;
- SHA256: `582d8ea7d7173404b3e5c56e620be906f655f2004e9b6ea1db40367e9def33b3`;
- source: V13 actor, fresh critic and optimizer;
- training: 4096 environments, 2000 updates, learning rate `1e-5`, two PPO
  epochs, fixed action standard deviation `0.015`.

| posture | held | final standing | final upright | termination | joint-speed RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| prone | 100% | 84.38% | 91.80% | 0% | 3.169 |
| supine | 100% | 86.33% | 94.14% | 0% | 3.190 |
| left side | 100% | 86.72% | 93.36% | 0% | 3.140 |
| right side | 99.61% | 86.33% | 92.58% | 0% | 3.196 |

V14 is a clear regression. The stronger dense costs increase motion and reduce
final standing retention without solving the target geometry. Videos show
persistent torso lean, arms still folded or swept behind the torso, repeated
large corrective steps, and occasional late falls. V14 must not be used as an
A6 source.

Artifacts are under
`logs/evaluations/host29_posture_v14_final_256_seed42/`, including JSON, four
MP4 files, manifest, and 0/4/8/12/16-second contact sheets.

## V15 result

- checkpoint: `logs/host29_posture_v15_scratch_4096_u12000_20261009/model_12000_final.pt`;
- SHA256: `1d469b97195c21fd4faea288fb39f02c4a6b8b63b5476b9f49b996f6de2923ff`;
- source: random actor, critic, normalizers, and optimizer;
- training: 4096 environments, 12000 updates, standard PPO curricula and
  intentionally extreme post-standing posture weights.

| posture | held | final standing | final upright | termination | joint-speed RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| prone | 100% | 100% | 100% | 0% | 1.587 |
| supine | 100% | 100% | 100% | 0% | 0.915 |
| left side | 100% | 100% | 100% | 0% | 1.579 |
| right side | 100% | 100% | 100% | 0% | 1.485 |

V15 passes every numerical acceptance threshold, including the 2.20
joint-speed-RMS ceiling. It nevertheless fails the visual acceptance gate.
Across all four videos, the torso remains visibly tilted rather than centered
vertically over the feet, while both elbows stay strongly flexed and the hands
remain folded together around the abdomen or pelvis instead of resting beside
the hips. The extreme scalar rewards therefore found a low-motion standing
solution without enforcing the intended geometry. V15 must not be used as an
A6 source without a redesigned posture representation or explicit geometric
constraint.

Artifacts are under
`logs/evaluations/host29_posture_v15_final_256_seed42/`, including the formal
JSON, four MP4 files, manifest, and 0/4/8/12/16-second contact sheets.
