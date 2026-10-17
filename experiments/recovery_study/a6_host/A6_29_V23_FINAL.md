# A6 29-DoF V23 final report

## Scope

V23 is a **29-DoF HoST A6 adapter** diagnostic at the V38 native action
scale `0.9323874`. It is not a complete paper-method reproduction and it is
not a formal-scale `0.25` result.

The experiment asks a narrower question than V22: while the V38 base actor and
the selected V20 update-200 recovery residual remain frozen, does any static
post-escape correction in the allowed symmetric sagittal action box produce a
stable formal-height posture? The scanned values are actual additive actions,
not policy logits:

```text
left/right hip pitch   in {-0.15, -0.075, 0, 0.075, 0.15}
left/right knee        in {-0.15, -0.075, 0, 0.075, 0.15}
left/right ankle pitch in {-0.15, -0.075, 0, 0.075, 0.15}
```

This gives 125 combinations. Each combination uses 32 environments balanced
over vertical/free plates, four directions, and a three-natural/one-procedural
all-low reset split in every scene/direction cell. G- and G+ each evaluate
4,000 environments for 1,150 control steps.

## Verification

The two-combination, 64-environment preflights and both full scans verify:

- paired initial robot state error is exactly `0.0` across combinations;
- paired initial actor-observation error is exactly `0.0`;
- the gate truth table is `[flat=false, pre-escape=false,
  post-escape=true, motor-off=false]`;
- the frozen V20 recovery output is bit-identical to its direct policy output;
- the static correction is exactly zero before escape and while motors are off;
- the three bilateral pairs receive the requested values exactly and all other
  23 action rows remain zero;
- all tracked states and actions remain finite, with no NaN/Inf.

The V23 implementation is commit `f9d0bde`. Full-scan result hashes are:

| Group | Summary SHA256 | Per-combination SHA256 |
|---|---|---|
| G- | `f71cf1db6df3cb452961f5dc2e9b7ceb3f1ebafd005e755ba1967e7753e23d5d` | `a9b09fcfb7f61d843f54f4adf294c7c08190665163bf690ac6f014b584cedb4f` |
| G+ | `42c115a1d4f3da9ef29a0dec7e4c6ad6a39b22b48349fbe5c7cbef234e662cf1` | `b9658d0d61584ddcf9a11ec99bb52c675a93fa73c6c09206a07989abb0fb28ff` |

## Full-grid result

No scanned combination produces a formal primary success, a single control
step satisfying the `1.15 m` height plus `0.90` upright predicate, or even one
escaped trial whose maximum head height reaches `1.15 m`.

| Group | Primary | Any height/upright hold | Any max head >= 1.15 m | Escape diagnostics | Abnormal |
|---|---:|---:|---:|---:|---:|
| G- | 0/4000 | 0/4000 | 0/4000 | 1182/4000 | 3/4000 |
| G+ | 0/4000 | 0/4000 | 0/4000 | 1166/4000 | 5/4000 |

The escape totals aggregate paired trials across all 125 post-escape action
choices; they are diagnostics, not independent reset samples or a success
rate claim.

The highest observed post-escape head heights are still below the formal
threshold:

- G-: `1.14144 m` at `(hip=0.0, knee=-0.15, ankle=-0.15)`;
- G+: `1.13659 m` at `(hip=0.15, knee=-0.15, ankle=-0.15)`.

These are isolated maxima, not held states. Their formal height/upright hold
remains `0.0 s`. The combinations with the highest escaped mean maximum head
height are also lower:

- G-: `(0.15, -0.15, 0.0)`, mean max `1.11304 m`, mean final
  `1.10149 m` over 13 escaped trials;
- G+: `(0.075, -0.15, -0.075)`, mean max `1.11378 m`, mean final
  `1.10219 m` over 11 escaped trials.

The zero-correction references remain near the known frozen equilibrium:

- G-: escaped mean max/final `1.09719/1.08910 m`;
- G+: escaped mean max/final `1.10201/1.09061 m`.

No combination has more than 5% mean combined-action clipping among escaped
steps. The negative result is therefore not explained by broad saturation of
the allowed action box.

## Conclusion

V23 is a clear negative reachability result for the V22 action subspace. With
V38 and V20 recovery frozen, the complete symmetric hip-pitch/knee/
ankle-pitch cube `[-0.15, 0.15]^3` contains no stable formal-height region for
either G- or G+. A finer local scan or an anchored correction policy is not
justified because the coarse scan never crosses `1.15 m`, never activates the
height/upright predicate for even one step, and remains at least `8.6 mm`
below the threshold even at the best isolated G- transient.

The next method must change the posture authority rather than repeat PPO
reward tuning. The defensible options are to train a higher standing target in
the base actor while preserving its flat-stability contract, or to introduce
a broader full-body post-escape posture optimizer whose reachable pose is
verified before policy training. The three-pair frozen-base adapter line ends
here.

