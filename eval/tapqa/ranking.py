"""Rankings that carry the scores that produced them.

Why this module exists
----------------------
A ranker used to return a plain list of event ids in whatever order its
sort produced. The harness then read gold's position off that list. The
consequence was that the *order within a tied group* decided the result,
and every tiebreak that was nominally neutral was not.

The concrete damage: `rank_lexical` broke ties on `event_id`, which
encodes insertion order, which encodes chronology. The original of a
corrected test always has the lower index, so the original always won the
tie. That single line handed token overlap 99.8% historical accuracy and
0.0% corrected accuracy - not because lexical matching is temporally
aware, but because of how the ids sort. A trivial baseline appeared to
solve the historical class, and the corrected class was destroyed for the
same reason.

The fix is to stop treating a tie as a decision. A ranker now reports the
scores it actually computed, and gold's rank is the *average* of the
positions any tied group occupies. A method that cannot tell the original
from the correction therefore earns no credit for putting one first,
which is the truth about that method. A method that does distinguish them
reports different scores and keeps full credit.

With rank decided by scores, intra-tie ordering stops affecting any
metric, so the residual tiebreak is cosmetic and can stay deterministic.
"""

from __future__ import annotations

from typing import Iterable, List, NamedTuple, Sequence, Tuple


class Ranking(NamedTuple):
    """An ordered list of event ids plus the score behind each position.

    `ids` is sorted best-first. `scores` is parallel to it and is what
    `gold_rank` actually reads; the ordering of `ids` inside a tied group
    is presentational only.
    """

    ids: Tuple[str, ...]
    scores: Tuple[float, ...]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __getitem__(self, index):
        return self.ids[index]

    @property
    def top(self):
        return self.ids[0] if self.ids else None

    def contains_all(self, wanted: Iterable[str]) -> bool:
        return all(w in self.ids for w in wanted)


def from_scores(
    scored: Sequence[Tuple[float, str]],
    *,
    descending: bool = True,
) -> Ranking:
    """Build a Ranking from (score, event_id) pairs.

    Sorting is on score alone. Ties are left as ties on purpose: adding a
    secondary key here is exactly the mistake this module exists to
    prevent, because any secondary key drawn from time or id is
    correlated with the answer.
    """
    rows = sorted(scored, key=lambda t: (-t[0] if descending else t[0], t[1]))
    return Ranking(
        ids=tuple(event_id for _, event_id in rows),
        scores=tuple(float(score) for score, _ in rows),
    )


def from_order(ids: Sequence[str]) -> Ranking:
    """Wrap a ranker that only produces an order, with no scores.

    For methods that genuinely impose a total order - a random shuffle,
    newest-first - there is no tie to be neutral about, so strictly
    decreasing synthetic scores say exactly that and no more. This is
    only honest for a method whose stated behaviour *is* a total order;
    it must not be used to launder a genuine tie into a free win.
    """
    n = len(ids)
    return Ranking(
        ids=tuple(ids),
        scores=tuple(float(n - i) for i in range(n)),
    )


def gold_rank(ranking: Ranking, gold: Sequence[str]) -> float:
    """1-based rank of `gold`, averaged over tied scores. 0.0 if absent.

    Standard treatment of ties, and the reason it matters here: with two
    records tied at the top, gold's rank is 1.5, so `rank == 1` is false
    and the method gets no credit. Only a method that actually separates
    them puts one at 1.0.

    A fractional rank is deliberate. Forcing it to an integer would either
    invent a distinction the method did not make, or round 1.5 down to 1
    and hand back the credit this is meant to withhold.
    """
    if not gold or not ranking.ids:
        return 0.0
    wanted = set(gold)

    position = 0
    index = 0
    total = len(ranking.ids)
    while index < total:
        score = ranking.scores[index]
        end = index
        while end + 1 < total and ranking.scores[end + 1] == score:
            end += 1
        group = ranking.ids[index : end + 1]
        if wanted & set(group):
            # 1-based inclusive span of the tied group, averaged.
            first, last = position + 1, position + (end - index) + 1
            return (first + last) / 2.0
        position += end - index + 1
        index = end + 1
    return 0.0


def tie_count(ranking: Ranking) -> int:
    """How many positions share a score with a neighbour.

    Reported alongside the results. A method whose headline number rests
    on a large tie rate is not measuring what it appears to measure, and
    the reader is entitled to know the size of that caveat.
    """
    tied = 0
    for i in range(1, len(ranking.scores)):
        if ranking.scores[i] == ranking.scores[i - 1]:
            tied += 1
    return tied
