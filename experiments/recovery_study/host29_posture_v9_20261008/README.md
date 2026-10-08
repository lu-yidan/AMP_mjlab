# 29 DoF posture V9 formal evaluation

## Checkpoints

- Source V8: `logs/host29_posture_v8_4096_20261008/model_500_final.pt`
  (`0bf9e3bd5d260436a8d1bc0bee169ab0c2c7895ddb361d00dfdadc42973c6c54`)
- V9 final: `logs/host29_posture_v9_4096_20261008/model_500_final.pt`
  (`204239ede2e50198e4878598ff50094ef327a8f0e26f06742e760e8dad268dc3`)

V9 preserves all V8 post-stand rewards and adds hand clearance (`+0.20`) and
torso-link verticality (`+0.25`). It keeps the inherited actor normalizer,
action scale `0.6089529991`, fixed action standard deviation `0.005`, and
trains only the final `Linear(128, 29)` actor layer for 500 updates at
`2e-6` learning rate.

## Formal result

Evaluation uses 256 environments per posture, 1000 control steps, seed 42,
and the checkpoint action scale.

| posture | ever | held | final standing | termination | first hold (s) | final joint-speed RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prone | 100% | 100% | 100% | 0% | 2.10 | 2.152 |
| supine | 100% | 100% | 100% | 0% | 1.82 | 2.131 |
| left side | 100% | 100% | 100% | 0% | 1.96 | 2.102 |
| right side | 100% | 100% | 100% | 0% | 1.90 | 2.051 |

All numeric acceptance thresholds pass. The visual acceptance does not:

- both hands remain close to or crossed in front of the torso;
- the torso remains visibly tilted rather than vertical;
- the upper body stays offset from the support midpoint;
- repeated stepping remains the dominant balance compensation.

Therefore V9 is not accepted as a posture-improved replacement for V8. The
full structured metrics are in `evaluation.json`; `video_manifest.json`
records the four source videos. The contact sheets in `frames/` sample each
video at 0, 4, 8, 12, and 16 seconds.
