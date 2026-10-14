# A6 prone postescape replay curriculum

The G1 29-DoF G+ escape policy can leave the 6 kg guided board while prone,
but often remains low and curled after the 25-step clearance latch. At the
0.15 m edge offset, `model_01699.pt` escaped in 13/16 sampled prone episodes
within 10 s but stood in only 1/16. With 15 s, 4/16 stood; with 20 s, 5/16
stood. More time helps but does not solve the pose-transition failure.

`scripts/collect_a6_postescape.py` recorded 512 genuine, no-assist guided-prone
states from `model_01699.pt` at the first 25-step full-body clearance. The
compressed source bank is `assets/a6/host29/escaped_prone_1699_512.npz`.
Each row has a 36-value 29-DoF free-root pose, a 35-value generalized
velocity, and the guided board centre relative to its simulation origin.
All values are finite and all 512 poses are unique. The captured torso height
has median 0.112 m; the board lies about 1.19 m from the torso in XY.

When `--a6-postescape-prone-replay-v1` is enabled, 320 of 4096 training worlds
(one quarter of the guided-prone quota) begin from a bank row with the board
at its recorded location. Those worlds start with `escaped=true`, a 25-step
clearance age, and motors active immediately; the remaining 3776 worlds keep
their original reset distribution. This exposes the *same PPO* to full-length
episodes starting from difficult postescape poses while preserving ordinary
escape, supine, free-board, and flat training. Its frozen flat teacher is still
the independent `model_00999.pt`.

The low-noise continuation improved deterministic prone escape but regressed
guided-supine escape. The earlier A6 `model_01699.pt` retained good supine
behavior: at the 0.15 m offset, the low-noise `model_01749.pt` escaped in only
3/16 guided-supine mean-action replays; replacing only those supine actions
with the fixed `model_01699.pt` restored 13/16 escapes and 7/16 final stands.
The training entry point therefore adds a second *fixed reference policy* for
guided supine actions. Only the main 29-DoF PPO actor and its four critics are
updated. The supine reference is an auxiliary imitation loss on guided-supine
samples, while the original flat teacher still preserves flat and postescape
behavior. Both teacher checkpoint hashes are stored in new checkpoints.

The source path is `scripts/train_a6_postescape_replay_server.sh`; the next
training parent is selected after evaluating the low-noise `model_01799.pt`.
The new output directory is separate from previous experiments. Restore the
previous reset code from the server snapshot taken before upload if a rollback
is needed. The replay bank is data, not a checkpoint or generated cache.

Local verification built a 32-world CPU environment with three special
postescape starts. Their board coverage was zero, minimum clearance exceeded
10 cm, actor observations had the expected 612 values, all 29 actions were
available, and a 25-step zero-action simulation produced finite rewards and
no special-world invalid resets. Independent post-training escape-plus-stand
evaluation remains required.

The first 4096-world preflight with the old 5 PPO epochs and 5e-5 learning rate
had KL 0.135, too large for the 0.01 target. Repeating the same one-update
transfer with learning rate 1e-5 and one epoch reduced KL to 0.00953 with 7
guided invalid events. These safer defaults are now in the v12 server script.
