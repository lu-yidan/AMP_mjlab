# A6 29-DoF V22 final report

## Scope

V22 is a **29-DoF HoST A6 adapter** evaluated at the V38 native action scale
`0.9323874`. It is not a complete paper-method reproduction and it is not a
formal-scale `0.25` result.

The experiment tests whether post-escape standing can be separated from
obstacle recovery:

- the V38 base actor and observation normalizer are frozen;
- each group's selected V20 update-200 residual is frozen and remains active
  in obstacle scenes from motor-on through the end of the episode;
- a fresh correction policy activates only after confirmed escape;
- its effective action is restricted to equal left/right hip-pitch, knee, and
  ankle-pitch pairs; all other 23 action rows are exactly zero;
- correction scale is `0.15`, fixed std is `0.03`, learning rate is `5e-6`,
  PPO uses two epochs, and each run trains for 1,200 updates with 4,096
  environments and 1,150-step episodes.

## Verification

Paired 64-environment, 1,150-step preflights verified:

- correction gate `[flat=false, pre-escape=false, post-escape=true,
  motor-off=false]`;
- frozen recovery output is bit-identical to the selected V20 policy;
- all three bilateral correction pairs are exactly equal;
- all inactive correction rows are exactly zero;
- the flat correction is exactly zero;
- post-escape height and stance rewards are nonzero and finite;
- policy gradients cannot flow through inactive actor output rows;
- no NaN/Inf occurs.

Both formal runs completed normally. Final checkpoint hashes are:

- G-: `80a6441ab3c1923b1a30dfbcf0d6d2bd5e007d8241ba3f15a20c07bd6a171086`;
- G+: `34edd03fc8f654718fd4298825b9c6e02b226dce0f029e5ebf16788765255752`.

## Flat contract

The update-200 and final corrections were checked on 128 near-standing flat
environments for 1,150 steps. Both groups retained `128/128`, ended at 100%
retained, and reported:

- `flat_applied_residual_max = 0.0`;
- `flat_action_contract_max_error = 0.0`.

V22 therefore preserves the frozen V38 flat action exactly. The all-low formal
flat cells remain `0/128` primary because V22 intentionally does not alter
flat low-reset get-up behavior.

## Native-scale all-low sweep

All entries use 256 environments, 1,150 steps, seed 42, action scale
`0.9323874`, recovery scale `0.25`, and correction scale `0.15`.

| Group | Point | Primary | Escape | Escaped final head | Escaped max head | Height/upright hold |
|---|---:|---:|---:|---:|---:|---:|
| G- | 200 | 0/256 | 42/256 | 1.0916 m | 1.1017 m | 0.0 s |
| G- | 400 | 0/256 | 41/256 | 1.0924 m | 1.1020 m | 0.0 s |
| G- | 600 | 0/256 | 45/256 | 1.0911 m | 1.1004 m | 0.0 s |
| G- | 800 | 0/256 | 43/256 | 1.0920 m | 1.1017 m | 0.0 s |
| G- | 1000 | 0/256 | 42/256 | 1.0948 m | 1.1030 m | 0.0 s |
| G- | final | 0/256 | 45/256 | **1.0966 m** | **1.1032 m** | 0.0 s |
| G+ | 200 | 0/256 | 41/256 | 1.0909 m | 1.1005 m | 0.0 s |
| G+ | 400 | 0/256 | 40/256 | 1.0904 m | 1.1017 m | 0.0 s |
| G+ | 600 | 0/256 | 45/256 | 1.0908 m | **1.1042 m** | 0.0 s |
| G+ | 800 | 0/256 | 39/256 | 1.0915 m | 1.1018 m | 0.0 s |
| G+ | 1000 | 0/256 | 44/256 | 1.0918 m | 1.1018 m | 0.0 s |
| G+ | final | 0/256 | 41/256 | 1.0922 m | 1.1018 m | 0.0 s |

The V20 selected points escaped `42/256` for G- and `48/256` for G+. V22
therefore gives no reliable escape improvement: G- reaches `45/256`, while G+
falls below its V20 reference. No checkpoint produces a primary success or a
nonzero formal-height hold.

## Correction response

The projection works as designed. At G- final, mean scaled post-escape
corrections are approximately:

- knees: `-0.0183` signed, `0.0658` absolute per side;
- hip pitch: `-0.0050` signed, `0.0219` absolute per side;
- ankle pitch: `+0.0038` signed, `0.0044` absolute per side.

At G+ final they are approximately:

- knees: `-0.0248` signed, `0.0275` absolute per side;
- hip pitch: `-0.0017` signed, `0.0088` absolute per side;
- ankle pitch: `+0.0053` signed, `0.0058` absolute per side.

Correction saturation is 0% at every checkpoint. Combined-action clipping is
about 3% through most of the sweep and falls near 2--3% at final. The actor
therefore learned a real, symmetric extension response and was not blocked by
the output limit, but even the strongest G- correction raises escaped final
head height by only about 5 mm and max height by about 2--3 mm. The remaining
gap to the `1.15 m` formal threshold is roughly 47 mm.

## Conclusion

V22 validates the architectural isolation and rejects the learning hypothesis.
Flat behavior is protected exactly, recovery remains intact, and asymmetric
upper-body leakage is removed. Nevertheless, a reward-trained three-pair
sagittal correction converges to a low-height local equilibrium and does not
create formal standing.

The next step must be diagnostic rather than a longer V22 run or a blind scale
increase. A bounded post-escape action sweep should first map hip/knee/ankle
corrections to head height, uprightness, bilateral load, and clipping while
holding the frozen V38+V20 controller fixed. Only if that map contains a
stable `>=1.15 m` region should a supervised or anchored correction policy be
trained toward it. Otherwise the formal height predicate or the frozen base
posture target must be changed; PPO reward tuning alone is not justified.
