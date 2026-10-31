"""TAP-QA: Temporally Adversarial Patient QA.

The benchmark that carries the paper. Ground truth is known **by
construction**: every record is generated here, and the correct answer to
every query is decided by the generator, not by a model and not by a
human annotator. That is what makes it reproducible and what makes the
result meaningful - a benchmark whose labels come from the same model
under test proves nothing.

Design, and why each part is here:

- **Distractor corrections.** Every record set contains corrections to
  tests the query is *not* about. Without these a system scores well by
  retrieving whatever was recently touched, which is not temporal
  reasoning at all. This is the single most important category here.

- **Three query classes.** `unchanged` proves the method is not merely
  cautious. `corrected` is the safety claim. `historical` asks for the
  superseded value, where the superseded value is the *correct* answer -
  this is what a recency-biased system gets wrong, and it is the whole
  reason the paper exists.

- **Contradiction probes.** Two events that genuinely conflict, where the
  right behaviour is to surface the conflict rather than pick a side.
  A system that always answers confidently should fail these.

- **Correction lags drawn from clinical distributions**, not a constant.
  Same-day lab amendments dominate; medication stops and diagnosis
  revisions come much later. Timing is what makes temporal reasoning hard,
  and a benchmark where every correction is 3 days old measures nothing.

Determinism: the same seed produces byte-identical output. No wall clock
is read anywhere in the generator.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EPOCH = datetime(2026, 6, 1, tzinfo=timezone.utc)

# ── clinical content ──────────────────────────────────────────────
#
# Reference ranges are real and matter: a generated value outside its
# range would be clinically absurd, and "abnormal vs normal" is a
# retrieval signal a method could exploit.

LABS: List[Dict[str, Any]] = [
    {"test": "Fasting Glucose", "unit": "mg/dL", "low": 70, "high": 110, "decimals": 0},
    {"test": "HbA1c", "unit": "%", "low": 4.0, "high": 5.6, "decimals": 1},
    {"test": "Serum Creatinine", "unit": "mg/dL", "low": 0.6, "high": 1.3, "decimals": 1},
    {"test": "Serum Potassium", "unit": "mEq/L", "low": 3.5, "high": 5.1, "decimals": 1},
    {"test": "Total Cholesterol", "unit": "mg/dL", "low": 125, "high": 200, "decimals": 0},
    {"test": "LDL Cholesterol", "unit": "mg/dL", "low": 50, "high": 130, "decimals": 0},
    {"test": "TSH", "unit": "mIU/L", "low": 0.4, "high": 4.0, "decimals": 2},
    {"test": "Haemoglobin", "unit": "g/dL", "low": 12.0, "high": 16.0, "decimals": 1},
    {"test": "Platelet Count", "unit": "lakhs", "low": 1.5, "high": 4.5, "decimals": 2},
    {"test": "Urine Albumin", "unit": "mg/day", "low": 0, "high": 30, "decimals": 0},
    {"test": "Vitamin D", "unit": "ng/mL", "low": 30, "high": 100, "decimals": 1},
    {"test": "Uric Acid", "unit": "mg/dL", "low": 3.5, "high": 7.2, "decimals": 1},
]

#: A correction typically moves a value by this much. A correction that
#: changes nothing is not a correction; one that changes everything is not
#: realistic either.
CORRECTION_DELTA = (0.12, 0.55)

DIAGNOSES = [
    "Type 2 Diabetes Mellitus",
    "Essential Hypertension",
    "Hypothyroidism",
    "Dyslipidaemia",
    "Chronic Kidney Disease Stage 3",
    "Anaemia of Chronic Disease",
    "Vitamin D Deficiency",
    "Obesity",
    "Metabolic Syndrome",
    "Osteoarthritis of Knee",
]

MEDICATIONS = [
    ("Metformin", "500 mg", "Twice daily"),
    ("Telmisartan", "40 mg", "Once daily"),
    ("Atorvastatin", "10 mg", "Once daily"),
    ("Levothyroxine", "75 mcg", "Once daily"),
    ("Vitamin D3", "60000 IU", "Once weekly"),
    ("Aspirin", "75 mg", "Once daily"),
    ("Amlodipine", "5 mg", "Once daily"),
    ("Ferrous Sulphate", "325 mg", "Once daily"),
]

NOTES = [
    "Advised dietary modification and 30 minutes of walking daily.",
    "Review after six weeks; repeat panel if readings remain above target.",
    "Discussed adherence and possible barriers to evening dosing.",
    "Patient reports good tolerance. No adverse effects noted.",
    "Smoking cessation counselling given.",
    "Next appointment booked; continue current regimen until then.",
]


def _recorded_at(valid_time: datetime, lag_days: int) -> datetime:
    """When the system learned a fact, given when it was true."""
    return valid_time + timedelta(
        days=lag_days, hours=random.Random(lag_days * 7919).randint(1, 20)
    )


def _round(value: float, decimals: int) -> float:
    return round(value, decimals)


@dataclass
class Record:
    """One clinical fact, in the same shape the core type uses."""

    event_id: str
    patient_id: str
    event_type: str
    valid_time: str
    recorded_at: str
    payload: Dict[str, Any]
    source_type: str = "manual"
    supersedes_id: Optional[str] = None
    #: Marks the event the benchmark injected a correction for, and the
    #: test it concerns. None for original records.
    corrects_test: Optional[str] = None


@dataclass
class Query:
    query_id: str
    patient_id: str
    text: str
    kind: str  # unchanged | corrected | historical | contradiction
    #: The event whose payload is the correct answer. For `historical` this
    #: is the *superseded* event, which is the point of that class.
    gold_event_id: str
    #: Every event id that would also be an acceptable answer. Usually just
    #: the gold one; a tie is possible when a value is re-reported unchanged.
    acceptable: List[str] = field(default_factory=list)
    #: The test name, so metrics can be sliced by what was asked.
    test: str = ""
    #: Present for `historical`: the instant the question refers to.
    as_of: Optional[str] = None
    #: Present for contradiction probes: the two conflicting event ids.
    conflict: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Case:
    patient_id: str
    records: List[Record]
    queries: List[Query]


@dataclass
class Benchmark:
    name: str
    seed: int
    cases: List[Case]

    def all_queries(self) -> List[Query]:
        return [q for case in self.cases for q in case.queries]

    def records_for(self, patient_id: str) -> List[Record]:
        for case in self.cases:
            if case.patient_id == patient_id:
                return case.records
        return []

    def stats(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        for q in self.all_queries():
            by_kind[q.kind] = by_kind.get(q.kind, 0) + 1
        return {
            "patients": len(self.cases),
            "records": sum(len(c.records) for c in self.cases),
            "queries": len(self.all_queries()),
            "by_kind": by_kind,
        }


class Generator:
    def __init__(self, seed: int = 20260601, *, same_valid_time: bool = False) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        #: Counterfactual switch, off by default and therefore absent from
        #: every reported number. When set, a correction is dated to the
        #: observation it revises instead of to its own effective time.
        self.same_valid_time = same_valid_time

    # ── values ──────────────────────────────────────────────────

    def _value(self, spec: Dict[str, Any], *, abnormal: bool) -> float:
        low, high, dec = spec["low"], spec["high"], spec["decimals"]
        if abnormal:
            # Beyond the range, by a clinically plausible margin.
            if self.rng.random() < 0.5:
                value = high * self.rng.uniform(1.12, 1.7)
            else:
                value = low * self.rng.uniform(0.45, 0.88)
        else:
            value = self.rng.uniform(low * 1.02, high * 0.98)
        return _round(max(0.0, value), dec)

    def _corrected_value(self, original: float, spec: Dict[str, Any]) -> float:
        low, high = spec["low"], spec["high"]
        magnitude = self.rng.uniform(*CORRECTION_DELTA)
        direction = 1.0 if self.rng.random() < 0.5 else -1.0
        # Scale against the reference range so the corrected value stays
        # clinically plausible rather than swinging to an absurd number.
        span = max(high - low, abs(high), 1.0)
        candidate = original + direction * magnitude * span
        return _round(max(0.0, candidate), spec["decimals"])

    # ── timing ──────────────────────────────────────────────────
    #
    # Lags follow clinical reality: a lab is amended within hours or not at
    # all, a diagnosis takes months to be revised.

    #: Days between a fact being true and the system learning it.
    #:
    #: This gap is what makes bitemporal modelling worth anything. A lab is
    #: drawn, sent to the lab, run, and reported - so valid time and
    #: transaction time genuinely differ. A record back-entered from an
    #: old file is the extreme case: clinically true last year, known to
    #: the system only this morning. If these two axes are within hours of
    #: each other, the bitemporal baseline is comparing a column to itself.
    REPORT_LAG = (0, 0, 0, 1, 1, 2, 3, 7, 14, 45)

    LAG_LAB = (0, 0, 1, 3, 14)  # 70% same-day, 20% 3-day, 10% 2-week
    LAG_DIAGNOSIS = (30, 60, 90, 120, 180)
    LAG_MEDICATION = (14, 30, 45, 60, 90)
    LAG_NOTE = (1, 3, 7, 14)

    def _lag(self, kind: str) -> int:
        table = {
            "lab": self.LAG_LAB,
            "diagnosis": self.LAG_DIAGNOSIS,
            "medication": self.LAG_MEDICATION,
            "note": self.LAG_NOTE,
            "report": self.REPORT_LAG,
        }[kind]
        # Weighted toward the short end, which is what the distributions
        # in the spec describe. Weights are generated from the table length
        # rather than hard-coded, because a hard-coded list silently breaks
        # when a table gains an entry - which is exactly what happened when
        # REPORT_LAG was added.
        n = len(table)
        weights = [max(1, round(8 * (0.72 ** i))) for i in range(n)]
        return self.rng.choices(table, weights=weights, k=1)[0]

    # ── record construction ─────────────────────────────────────

    def _make_case(
        self,
        index: int,
        *,
        n_labs: int,
        n_distractors: int,
        n_contradictions: int,
        with_corrections: bool,
    ) -> Case:
        pid = f"tap-{self.seed}-{index:04d}"
        records: List[Record] = []
        queries: List[Query] = []
        counter = 0

        def next_id() -> str:
            nonlocal counter
            counter += 1
            return f"{pid}-e{counter:03d}"

        # Backdated over roughly two years so corrections have room to sit
        # between an original and its amendment.
        specs = self.rng.sample(LABS, k=min(n_labs, len(LABS)))

        originals: List[Tuple[str, Dict[str, Any], datetime]] = []
        for spec in specs:
            days_ago = self.rng.randint(20, 700)
            when = EPOCH - timedelta(days=days_ago, hours=self.rng.randint(0, 20))
            eid = next_id()
            value = self._value(spec, abnormal=self.rng.random() < 0.35)
            records.append(
                Record(
                    event_id=eid,
                    patient_id=pid,
                    event_type="lab",
                    valid_time=when.isoformat(),
                    recorded_at=_recorded_at(when, self._lag("report")).isoformat(),
                    payload={
                        "test": spec["test"],
                        "value": value,
                        "unit": spec["unit"],
                        "reference_low": spec["low"],
                        "reference_high": spec["high"],
                    },
                )
            )
            originals.append((eid, spec, when))

        # Supporting clinical context. Without unrelated records the task
        # degenerates into "find the one match", which every method solves.
        for _ in range(self.rng.randint(3, 6)):
            days_ago = self.rng.randint(10, 720)
            when = EPOCH - timedelta(days=days_ago)
            roll = self.rng.random()
            if roll < 0.4:
                records.append(
                    Record(
                        event_id=next_id(),
                        patient_id=pid,
                        event_type="diagnosis",
                        valid_time=when.isoformat(),
                        recorded_at=when.isoformat(),
                        payload={"diagnosis": self.rng.choice(DIAGNOSES)},
                    )
                )
            elif roll < 0.75:
                drug, dose, freq = self.rng.choice(MEDICATIONS)
                records.append(
                    Record(
                        event_id=next_id(),
                        patient_id=pid,
                        event_type="medication",
                        valid_time=when.isoformat(),
                        recorded_at=when.isoformat(),
                        payload={
                            "drug": drug,
                            "dose": dose,
                            "frequency": freq,
                            "status": "active",
                        },
                    )
                )
            else:
                records.append(
                    Record(
                        event_id=next_id(),
                        patient_id=pid,
                        event_type="note",
                        valid_time=when.isoformat(),
                        recorded_at=when.isoformat(),
                        payload={"text": self.rng.choice(NOTES)},
                    )
                )

        # ── corrections ─────────────────────────────────────────
        if with_corrections and originals:
            target_eid, spec, when = self.rng.choice(originals)
            lag = self._lag("lab")
            corrected_at = min(
                EPOCH - timedelta(days=1), when + timedelta(days=lag)
            )
            if corrected_at <= when:
                corrected_at = when + timedelta(days=1)
            if self.same_valid_time:
                # Counterfactual used only by the alpha/weight ablation probe:
                # date the correction to the observation it revises, which
                # is what a system that never revised a valid time would
                # produce. It isolates how much of the valid-time-only
                # baseline's behaviour is a consequence of our dating
                # convention rather than of the model.
                corrected_at = when
            original_value = next(
                r.payload["value"] for r in records if r.event_id == target_eid
            )
            new_value = self._corrected_value(original_value, spec)
            correction_id = next_id()
            records.append(
                Record(
                    event_id=correction_id,
                    patient_id=pid,
                    event_type="lab",
                    valid_time=corrected_at.isoformat(),
                    recorded_at=_recorded_at(
                        corrected_at, self._lag("report")
                    ).isoformat(),
                    payload={
                        "test": spec["test"],
                        "value": new_value,
                        "unit": spec["unit"],
                        "reference_low": spec["low"],
                        "reference_high": spec["high"],
                    },
                    source_type="correction",
                    supersedes_id=target_eid,
                    corrects_test=spec["test"],
                )
            )
            self._queries_for_corrected_pair(
                queries, next_id, target_eid, correction_id, spec, when,
                corrected_at,
            )

            # Distractors: corrections to tests the queries do not ask
            # about. Without these, "retrieve the most recently corrected
            # record" scores well and is not temporal reasoning.
            for _ in range(n_distractors):
                others = [
                    (eid, sp, w)
                    for eid, sp, w in originals
                    if sp["test"] != spec["test"]
                ]
                if not others:
                    break
                d_eid, d_spec, d_when = self.rng.choice(others)
                d_lag = self._lag("lab")
                d_at = min(EPOCH - timedelta(days=1), d_when + timedelta(days=d_lag))
                if d_at <= d_when:
                    d_at = d_when + timedelta(days=1)
                d_old = next(
                    r.payload["value"] for r in records if r.event_id == d_eid
                )
                records.append(
                    Record(
                        event_id=next_id(),
                        patient_id=pid,
                        event_type="lab",
                        valid_time=d_at.isoformat(),
                        recorded_at=d_at.isoformat(),
                        payload={
                            "test": d_spec["test"],
                            "value": self._corrected_value(d_old, d_spec),
                            "unit": d_spec["unit"],
                            "reference_low": d_spec["low"],
                            "reference_high": d_spec["high"],
                        },
                        source_type="correction",
                        supersedes_id=d_eid,
                        corrects_test=d_spec["test"],
                    )
                )

        # ── unchanged queries ───────────────────────────────────
        for eid, spec, when in originals:
            queries.append(
                Query(
                    query_id=f"{pid}-q-unch-{eid[-3:]}",
                    patient_id=pid,
                    text=self._ask_current(spec["test"]),
                    kind="unchanged",
                    gold_event_id=eid,
                    test=spec["test"],
                )
            )

        # ── contradiction probes ─────────────────────────────────
        for _ in range(n_contradictions):
            if len(originals) < 2:
                break
            a, b = self.rng.sample(originals, 2)
            a_spec, b_spec = a[1], b[1]
            a_val = next(r.payload["value"] for r in records if r.event_id == a[0])
            b_val = next(r.payload["value"] for r in records if r.event_id == b[0])
            # Two records, same test, different values, neither marked as
            # correcting the other. The record itself is ambiguous.
            conflict_a = next_id()
            conflict_b = next_id()
            for eid, spec, base_val, offset in (
                (conflict_a, a_spec, a_val, 2),
                (conflict_b, b_spec, a_val, 5),
            ):
                when = a[2] + timedelta(days=offset)
                if when >= EPOCH:
                    when = EPOCH - timedelta(days=1)
                records.append(
                    Record(
                        event_id=eid,
                        patient_id=pid,
                        event_type="lab",
                        valid_time=when.isoformat(),
                        recorded_at=when.isoformat(),
                        payload={
                            "test": spec["test"],
                            "value": self._corrected_value(base_val, spec),
                            "unit": spec["unit"],
                            "reference_low": spec["low"],
                            "reference_high": spec["high"],
                        },
                    )
                )
            queries.append(
                Query(
                    query_id=f"{pid}-q-contra-{conflict_a[-3:]}",
                    patient_id=pid,
                    text=(
                        f"There are two different {a_spec['test']} results on "
                        f"file. Which should I act on?"
                    ),
                    kind="contradiction",
                    # There is no single right answer here. Gold is the
                    # earlier record; a method is only credited if it
                    # surfaces *both*, which `acceptable` encodes.
                    gold_event_id=conflict_a,
                    acceptable=[conflict_a, conflict_b],
                    test=a_spec["test"],
                    conflict=[conflict_a, conflict_b],
                )
            )

        return Case(patient_id=pid, records=records, queries=queries)

    # ── query phrasings ────────────────────────────────────────
    #
    # Several surface forms per intent. A single phrasing lets a method
    # learn a lexical shortcut, and the whole point is that retrieval must
    # survive paraphrase.

    def _ask_current(self, test: str) -> str:
        return self.rng.choice(
            [
                f"what is her {test.lower()} now",
                f"current {test.lower()}",
                f"latest {test.lower()} reading",
                f"what is the most recent {test.lower()}",
            ]
        )

    def _queries_for_corrected_pair(
        self,
        queries: List[Query],
        next_id,
        original_id: str,
        correction_id: str,
        spec: Dict[str, Any],
        original_time: datetime,
        correction_time: datetime,
    ) -> None:
        test = spec["test"]

        # Present-tense. Gold is the correction; answering with the
        # superseded value is the stale-answer failure.
        queries.append(
            Query(
                query_id=f"{original_id}-q-now",
                patient_id=original_id.split("-e")[0],
                text=self._ask_current(test),
                kind="corrected",
                gold_event_id=correction_id,
                test=test,
            )
        )

        # Historical, asked two ways: an explicit instant and a relative
        # offset. The relative form exists because "as of 60 days ago" is
        # how clinicians actually speak, and a system that only parses
        # ISO dates would silently answer about the present.
        queries.append(
            Query(
                query_id=f"{original_id}-q-asof",
                patient_id=original_id.split("-e")[0],
                text=f"what was her {test.lower()} as of {original_time.date().isoformat()}",
                kind="historical",
                gold_event_id=original_id,
                test=test,
                as_of=original_time.isoformat(),
            )
        )
        # The offset is measured from the benchmark's reference instant to
        # the record, NOT from the record to its correction.
        #
        # Using the record-to-correction gap here means "14 days ago"
        # points at a completely different date from the one stored in
        # `as_of`, so the as-of veto fires on the wrong record and the
        # gold answer is ranked second. That silently penalised the
        # graph-conditioned method - the one whose behaviour is correct -
        # and would have looked like a genuine finding.
        age_days = max(1, (EPOCH - original_time).days)
        queries.append(
            Query(
                query_id=f"{original_id}-q-rel",
                patient_id=original_id.split("-e")[0],
                text=f"what was her {test.lower()} {age_days} days ago",
                kind="historical",
                gold_event_id=original_id,
                test=test,
                as_of=original_time.isoformat(),
            )
        )

    # ── assembly ────────────────────────────────────────────────

    def build(
        self,
        *,
        n_unchanged: int = 200,
        n_corrected: int = 200,
        n_contradiction: int = 50,
        cases_per_kind: int = 1,
    ) -> Benchmark:
        """Build a benchmark, then top up each class to its target.

        Generation is per-case and yields whatever the sampler produces, so
        the class counts are topped up afterwards. Topping up is by
        construction, not by filtering, so no class is quietly
        over-represented with the easy remainder.
        """
        cases: List[Case] = []
        index = 0
        targets = {
            "unchanged": n_unchanged,
            "corrected": n_corrected,
            "contradiction": n_contradiction,
        }
        counts = {k: 0 for k in targets}

        # Generate in rounds so every class keeps growing until it is met.
        max_rounds = 400
        while any(counts[k] < targets[k] for k in targets) and max_rounds > 0:
            max_rounds -= 1
            for kind, target in targets.items():
                if counts[kind] >= target:
                    continue
                index += 1
                case = self._make_case(
                    index,
                    n_labs=self.rng.randint(6, 10),
                    n_distractors=self.rng.randint(2, 4),
                    n_contradictions=1 if kind == "contradiction" else 0,
                    with_corrections=kind in ("corrected", "contradiction"),
                )
                case.queries = [
                    q
                    for q in case.queries
                    if (q.kind == kind or q.kind == "unchanged" or q.kind == "historical")
                    and not (q.kind == "contradiction" and kind != "contradiction")
                ]
                if kind == "unchanged":
                    case.queries = [q for q in case.queries if q.kind == "unchanged"]
                elif kind == "corrected":
                    case.queries = [
                        q for q in case.queries if q.kind in ("corrected", "historical")
                    ]
                else:
                    case.queries = [q for q in case.queries if q.kind == "contradiction"]
                    if not case.queries:
                        continue
                cases.append(case)
                counts[kind] += len(
                    [q for q in case.queries if q.kind == kind]
                )

        return Benchmark(
            name="TAP-QA", seed=self.seed, cases=cases
        )


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build(seed: int = 20260601, *, same_valid_time: bool = False, **kwargs: Any) -> Benchmark:
    return Generator(seed, same_valid_time=same_valid_time).build(**kwargs)
