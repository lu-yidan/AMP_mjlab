# 29 DoF posture V11 result and V13 launch

## V11 from-scratch result

- checkpoint: `logs/host29_posture_v11_scratch_4096_u12000_20261008/model_12000_final.pt`;
- SHA256: `c24f7da91fcca780006577b5c45156fda16158a67ddc8b363506951d5c2c00a7`;
- initialization: random actor, critic, normalizers, and optimizer;
- training: 4096 environments, 12000 updates, standard action-scale and
  pull-force curricula.

The formal evaluation uses 256 environments per posture, 1000 control steps,
and seed 42.

| posture | held | final standing | termination | first hold (s) | root speed | joint-speed RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prone | 100% | 100% | 0% | 2.38 | 0.148 | 2.603 |
| supine | 100% | 100% | 0% | 3.74 | 0.140 | 2.604 |
| left side | 100% | 100% | 0% | 2.48 | 0.149 | 2.569 |
| right side | 99.61% | 99.61% | 0% | 2.02 | 0.144 | 2.593 |

V11 confirms that training from scratch removes the inherited folded-arm local
optimum: the arms are substantially more open than V10/V12. It still fails the
acceptance gate because all four joint-speed RMS values exceed 2.20. Video
frames also show persistent torso lean and left/right arm asymmetry; supine and
side starts often settle into a tilted stance rather than centering the torso
over both ankles. Root translation is relatively quiet, so the remaining issue
is primarily posture geometry and joint-level motion rather than continuous
large stepping.

Artifacts are under
`logs/evaluations/host29_posture_v11_scratch_final_256_seed42/` and include the
evaluation JSON, four videos, manifest, and 0/4/8/12/16-second contact sheets.

## V13 design

V13 continues from the V11 actor because V11 has the desired recovery skill and
escaped the folded-arm optimum. It does not reuse V11's critic or optimizer.
The actor normalizer and action scale are frozen, action standard deviation is
fixed at 0.01, and the complete actor is trained for 1000 updates with one PPO
epoch, learning rate `3e-6`, and zero entropy.

Targeted post-stand changes are:

- hands beside hips `+0.55`;
- torso vertical `+0.35` and dense torso tilt `-0.55`;
- torso position over the ankle midpoint `+0.30`;
- linear upper-pose error `-0.30`;
- horizontal foot speed `-0.08`;
- leg and upper-body joint-speed penalties `-0.0015/-0.0008`.

The 64-environment, 100-step preflight passed with finite observations,
actions, and rewards. The formal run is active on GPU4:

```text
scripts/train_host29_v13.py
  --checkpoint logs/host29_posture_v11_scratch_4096_u12000_20261008/model_12000_final.pt
  --log-dir logs/host29_posture_v13_4096_u1000_20261009
  --num-envs 4096 --updates 1000
```

## V13 formal result

- final checkpoint: `logs/host29_posture_v13_4096_u1000_20261009/model_1000_final.pt`;
- SHA256: `af37ec8ece40452e0214cc0682bad1cda69f60f30afa1e08e501d0396a6d1233`.

| posture | held | final standing | termination | first hold (s) | root speed | joint-speed RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prone | 100% | 100% | 0% | 2.42 | 0.142 | 2.589 |
| supine | 100% | 100% | 0% | 3.72 | 0.147 | 2.619 |
| left side | 100% | 100% | 0% | 2.44 | 0.149 | 2.601 |
| right side | 99.22% | 99.22% | 0% | 2.00 | 0.147 | 2.603 |

V13 does not pass the acceptance gate. All four joint-speed RMS values remain
above 2.20 and are effectively unchanged from V11. The videos show a modestly
more repeatable posture, but the torso still leans away from the ankle
midpoint, the arms remain bent and asymmetric rather than resting beside the
hips, and side-start rollouts retain a visibly offset stance. The added reward
terms did not materially change the V11 solution within 1000 conservative
updates.

No V14 or A6 continuation is started. Further work needs a user decision on
whether to change the posture representation/reference trajectory, strengthen
the optimization regime, or relax the visual and joint-speed acceptance gate.
