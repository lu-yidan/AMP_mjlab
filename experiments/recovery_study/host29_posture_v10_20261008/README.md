# 29 DoF posture V10 formal evaluation

## Checkpoints

- Source V4: `logs/host29_elbowflex_v4_20261007/model_15999.pt`
- V10 final: `logs/host29_posture_v10_4096_u2000_20261008/model_2000_final.pt`
  (`7486372405b3316324de33d93601915dd3fecfea960d8d4fb9c83cfd906b645c`)

V10 restarts directly from the original V4 actor instead of inheriting V8 or
V9. It uses a fresh critic and optimizer, trains the complete actor MLP for
2000 updates at `1e-6`, and freezes the inherited normalizer, action scale,
and action standard deviation. Its reward configuration matches V9.

## Formal result

Evaluation uses 256 environments per posture, 1000 control steps, seed 42,
and the checkpoint action scale.

| posture | ever | held | final standing | termination | first hold (s) | final joint-speed RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prone | 100% | 100% | 100% | 0% | 2.10 | 2.070 |
| supine | 100% | 100% | 100% | 0% | 1.84 | 2.091 |
| left side | 100% | 100% | 100% | 0% | 1.96 | 2.129 |
| right side | 100% | 100% | 100% | 0% | 1.92 | 2.022 |

V10 passes every numeric threshold and improves the aggregate speed metrics
over V8/V9. It still fails the visual requirements:

- the hands and elbows remain held in front of the chest;
- the torso remains visibly pitched or laterally tilted;
- the upper body does not settle over the support midpoint;
- repeated stepping remains the primary balance compensation.

The full metrics are in `evaluation.json`; `video_manifest.json` records the
four source videos. Contact sheets under `frames/` sample each video at 0, 4,
8, 12, and 16 seconds.
