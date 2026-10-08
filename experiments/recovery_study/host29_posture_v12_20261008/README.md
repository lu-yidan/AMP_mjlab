# 29 DoF posture V12 formal evaluation

## Checkpoints

- Source V10: `logs/host29_posture_v10_4096_u2000_20261008/model_2000_final.pt`
- V12 final: `logs/host29_posture_v12_4096_u1000_20261008/model_1000_final.pt`
  (`0a11cb7d3097f76cb3b108407efb519a4ba5c7a127d7d524b4fa3eeb99e62254`)

V12 transfers the V10 actor exactly, initializes a fresh critic and optimizer,
and trains the complete actor for 1000 updates. The inherited observation
normalizer and action scale remain frozen. Action standard deviation is fixed
at `0.015`; PPO uses one epoch, learning rate `5e-6`, and zero entropy.

The reward revision removes the ambiguous hand-distance and upper-chain-center
terms and adds:

- hands beside hips: `+0.40`;
- torso tilt: `-0.30`;
- linear upper-pose error: `-0.20`;
- horizontal foot speed after standing: `-0.05`.

## Formal result

Evaluation uses 256 environments per posture, 1000 control steps, and seed 42.

| posture | ever | held | final standing | termination | first hold (s) | final joint-speed RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prone | 100% | 100% | 100% | 0% | 2.10 | 2.106 |
| supine | 100% | 100% | 100% | 0% | 1.84 | 2.201 |
| left side | 100% | 100% | 100% | 0% | 1.96 | 2.152 |
| right side | 100% | 100% | 100% | 0% | 1.92 | 2.129 |

V12 preserves reliable stand-up behavior, but it does not pass the complete
acceptance gate. Supine joint-speed RMS is marginally above the `2.20` limit.
More importantly, the four videos still show hands and elbows held near the
chest, persistent torso pitch or lateral offset, and repeated corrective
steps. The denser target terms did not move the V10 policy out of its visual
local optimum.

The full local artifacts are under
`logs/evaluations/host29_posture_v12_final_256_seed42/`, including
`evaluation.json`, four MP4 files, the manifest, and 0/4/8/12/16-second contact
sheets.

## Next experiment

V11 is a from-scratch control experiment with random actor, critic,
normalizers, and optimizer. It determines whether the folded-arm/tilted-torso
solution is primarily inherited-policy bias. V11 must be formally evaluated
before choosing the V13 initialization and reward changes.
