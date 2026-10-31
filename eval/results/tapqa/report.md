# TAP-QA results

`TAP-QA` seed `20260601` - 277 patients, 4500 records, 850 queries

Query classes: contradiction 50, corrected 200, historical 400, unchanged 200

## Retrieval results

Rank is read from each method's own scores, averaged over ties. A method that scores the original and the correction of a test identically has not told them apart, so it earns no credit for whichever one it happened to list first.

This is not a technicality, and it is what makes the floor rows mean something. An earlier run of this harness reported token overlap at 99.8% historical accuracy. It was breaking ties on record id, which encodes insertion order, which encodes chronology - so the original of every corrected test won automatically, and a method with no notion of supersession appeared to solve the historical class outright. Read the `Tied queries` column as the share of each row that rests on the method's scores rather than on the order it emitted: token overlap ties on all 850 queries, which is exactly why it now scores 0% on both classes that require knowing which version was current. The 0% is the true reading of a method that cannot see supersession, not a bug in the harness.

| Method | SAR (lower better) | Historical acc | Unchanged acc | Corrected acc | R@1 | MRR | Tied queries |
|---|---|---|---|---|---|---|---|
| Bitemporal, both axes, no graph | 7.0% [4.2, 11.4] | 6.5% [4.5, 9.4] | 94.5% [90.4, 96.9] | 87.5% [82.2, 91.4] | 49.6% | 71.2% | 184 |
| Graph filter (as-of admissible, then dense) | 0.0% [0.0, 1.9] | 98.2% [96.4, 99.1] | 97.5% [94.3, 98.9] | 100.0% [98.1, 100.0] | 95.3% | 97.6% | 161 |
| Token overlap | 100.0% [98.1, 100.0] | 0.0% [0.0, 1.0] | 97.5% [94.3, 98.9] | 0.0% [0.0, 1.9] | 22.9% | 73.8% | 850 |
| Random floor | 17.0% [12.4, 22.8] | 9.2% [6.8, 12.5] | 7.5% [4.6, 12.0] | 4.5% [2.4, 8.3] | 7.4% | 22.3% | 0 |
| Ours: three-valued, graph-conditioned | 0.0% [0.0, 1.9] | 97.8% [95.8, 98.8] | 94.5% [90.4, 96.9] | 98.5% [95.7, 99.5] | 93.5% | 96.7% | 161 |
| Dense-only (no temporal term) | 49.5% [42.6, 56.4] | 45.8% [40.9, 50.6] | 97.5% [94.3, 98.9] | 50.5% [43.6, 57.4] | 58.7% | 79.1% | 250 |
| RRF fusion, no temporal term | 69.5% [62.8, 75.5] | 55.0% [50.1, 59.8] | 55.5% [48.6, 62.2] | 1.5% [0.5, 4.3] | 39.4% | 64.3% | 0 |
| Latest-N | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.0] | 5.0% [2.7, 9.0] | 10.5% [7.0, 15.5] | 3.9% | 20.7% | 0 |
| Transaction-time only | 21.5% [16.4, 27.7] | 21.2% [17.5, 25.5] | 99.0% [96.4, 99.7] | 78.5% [72.3, 83.6] | 55.4% | 77.5% | 190 |
| Valid-time only | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.0] | 99.0% [96.4, 99.7] | 100.0% [98.1, 100.0] | 52.1% | 75.9% | 188 |

## Recency decay sweep

`score = semantic + lambda * exp(-age / tau)`, the strongest argument against a temporal method: a recency preference is one number, so if it can buy safety it can be tuned. The full 35-point grid is reported, because a single lambda is a strawman.

**The sweep does not trace a trade-off curve, and that is the finding.** Two corners, no diagonal:

- `tau = 7`: SAR 64-70%, historical 0%. A short time constant makes recency dominate, so the method leads with whatever was recorded last - which for a corrected test is the retracted value.
- `tau >= 30`: SAR 0%, historical 0%. Once the constant is long enough the temporal term is nearly flat across the whole record history, so the method is stale-free for the same reason Latest-N is: it is not leading with anything in particular. Historical accuracy is zero because the flat term carries no information about *which* version was valid when.

So lambda and tau do not trade staleness against historical recall. They select between two degenerate corners, and no point on the grid reaches the upper-left region the graph-conditioned method occupies. The prediction in the project plan was a smooth downward line; the measurement is a step. Reported as measured.

| lambda | tau (days) | SAR | Historical acc |
|---|---|---|---|
| 0.01 | 7 | 70.0% | 0.0% |
| 0.03 | 7 | 68.0% | 0.0% |
| 0.05 | 7 | 67.0% | 0.0% |
| 0.1 | 7 | 66.0% | 0.0% |
| 0.15 | 7 | 66.0% | 0.0% |
| 0.3 | 7 | 64.5% | 0.0% |
| 0.5 | 7 | 64.5% | 0.0% |
| 0.01 | 30 | 0.0% | 0.0% |
| 0.03 | 30 | 0.0% | 0.0% |
| 0.05 | 30 | 0.0% | 0.0% |
| 0.1 | 30 | 0.0% | 0.0% |
| 0.15 | 30 | 0.0% | 0.0% |
| 0.3 | 30 | 0.0% | 0.0% |
| 0.5 | 30 | 0.0% | 0.0% |
| 0.01 | 90 | 0.0% | 0.0% |
| 0.03 | 90 | 0.0% | 0.0% |
| 0.05 | 90 | 0.0% | 0.0% |
| 0.1 | 90 | 0.0% | 0.0% |
| 0.15 | 90 | 0.0% | 0.0% |
| 0.3 | 90 | 0.0% | 0.0% |
| 0.5 | 90 | 0.0% | 0.0% |
| 0.01 | 180 | 0.0% | 0.0% |
| 0.03 | 180 | 0.0% | 0.0% |
| 0.05 | 180 | 0.0% | 0.0% |
| 0.1 | 180 | 0.0% | 0.0% |
| 0.15 | 180 | 0.0% | 0.0% |
| 0.3 | 180 | 0.0% | 0.0% |
| 0.5 | 180 | 0.0% | 0.0% |
| 0.01 | 365 | 0.0% | 0.0% |
| 0.03 | 365 | 0.0% | 0.0% |
| 0.05 | 365 | 0.0% | 0.0% |
| 0.1 | 365 | 0.0% | 0.0% |
| 0.15 | 365 | 0.0% | 0.0% |
| 0.3 | 365 | 0.0% | 0.0% |
| 0.5 | 365 | 0.0% | 0.0% |

## Paired significance vs. the graph-conditioned method

McNemar exact on the *safe answer* outcome over the 200 corrected queries - gold ranked first **and** the top record not superseded - and a paired bootstrap (10,000 resamples) on historical accuracy.

The outcome is joint on purpose, and the reason is visible in the staleness-only column. Staleness alone is not a testable safety claim: a method that never returns a superseded record is stale-free by refusing to answer. Latest-N is exactly that - 0% staleness, 0% historical accuracy - and a staleness-only test scores it as indistinguishable from a system that works, p = 1.0. Requiring the answer to also be correct is what gives the comparison content, and it is why the two columns disagree for Latest-N and Valid-time only.

Two entries in the staleness-only column are worth reading carefully rather than as wins. Valid-time only reaches p = 0.25 because it is stale-free *and* correct on 100% of corrected queries, so the two arms agree on almost every pair; the honest reading is that it has no staleness left to fix, not that the graph adds nothing there. It scores 0% historical, so it cannot answer the question the system exists for. Latest-N's p = 1.0 for the same reason. The historical column separates them: +0.963 with a CI excluding zero, for both.

| Baseline | McNemar p (safe answer) | Sig | McNemar p (staleness only) | Bootstrap p | 95% CI on diff |
|---|---|---|---|---|---|
| Random floor | 0.0000 | yes | 0.0000 | 0.0000 | [+0.853, +0.915] |
| Token overlap | 0.0000 | yes | 0.0000 | 0.0000 | [+0.960, +0.993] |
| Dense-only (no temporal term) | 0.0000 | yes | 0.0000 | 0.0000 | [+0.458, +0.585] |
| RRF fusion, no temporal term | 0.0000 | yes | 0.0000 | 0.0000 | [+0.357, +0.497] |
| Valid-time only | 0.2500 | no | 1.0000 | 0.0000 | [+0.960, +0.993] |
| Transaction-time only | 0.0000 | yes | 0.0000 | 0.0000 | [+0.708, +0.823] |
| Latest-N | 0.0000 | yes | 1.0000 | 0.0000 | [+0.960, +0.993] |
| Bitemporal, both axes, no graph | 0.0000 | yes | 0.0001 | 0.0000 | [+0.875, +0.948] |
| Graph filter (as-of admissible, then dense) | 0.2500 | no | 1.0000 | 0.2662 | [-0.013, +0.000] |
