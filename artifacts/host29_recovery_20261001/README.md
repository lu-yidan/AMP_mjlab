# HoST 29-DoF recovery model — 2026-10-01

`model_10000.pt` is the selected deployment checkpoint from the corrected
mixed-assistance run. `model_11999.pt` is retained on the training server as the
final checkpoint, but is not selected because its mean final root speed is
higher (0.801 m/s versus 0.446 m/s).

Evaluation used 256 environments for each of prone, supine, left-side and
right-side starts, 500 control steps (10 s), no pull force, and a 50-step (1 s)
continuous standing hold criterion. The selected checkpoint achieved 100% held
and 100% final-standing rates in all four postures, with no abnormal
terminations.

The checkpoint action scale is 0.9401794672. The success condition is torso
height above mean foot height greater than 0.6 m and projected-gravity z less
than -0.8.

Files:

- `model_10000.pt`: selected training checkpoint.
- `eval_model10000_unassisted.json`: full four-posture evaluation.
- `eval_model11999_unassisted.json`: final-checkpoint evaluation for comparison.
- `env.yaml`, `agent.yaml`, `launch.json`: frozen run configuration and lineage.
- `manifest.json`: hashes, selection decision and aggregate metrics.

Training run on the server:

`logs/host29_mixedforce_fix_seed42_4096_20260930`

