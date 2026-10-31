"""The answer pipeline: retrieve, de-identify, generate, verify, assemble.

This is where the three research components meet. Order matters and is
not arbitrary:

    1. resolve temporal intent          (what does "now" mean here?)
    2. validity-aware retrieval         (weight by state, not by delta-t)
    3. as-of resolution                 (relative to the requested instant)
    4. de-identify                      (mandatory before any outbound call)
    5. generate                         (LLM proposes)
    6. provenance lock                  (linker decides, not the model)
    7. assemble                         (verified claims survive)

Step 6 is why the answer is trustworthy: the model does not get to decide
whether its own citations are valid.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from app.core.validity import Event, EventType, SourceType, SupersessionGraph, Validity
from app.services.collapse import collapse
from app.services.hybrid import semantic_for
from app.services.llm import LLMClient, LLMResult, parse_claims
from app.services.pii import Deidentifier
from app.services.provenance import (
    Claim,
    GroundedAnswer,
    ProvenanceLinker,
    assemble,
)
from app.services.retrieval import (
    ScoredEvent,
    TemporalIntent,
    ValidityAwareRetriever,
    parse_temporal_intent,
)

logger = logging.getLogger(__name__)

#: Where a supersession graph is absent (e.g. a pure-lexical smoke test),
#: this keeps the pipeline runnable rather than crashing.
_EMPTY_GRAPH = SupersessionGraph([])

#: Phrases by which an answer declares it could not answer. When the final
#: text matches, any attached citations are stripped: a citation is a claim
#: of support, and "I don't know" supported by three records is a
#: contradiction the UI would otherwise render as confidence. The text is
#: kept - the doctor should see the abstention - but it goes out alone.
_ABSTENTION_RE = re.compile(
    r"\b("
    r"i (do not|don't|can ?not|could not|am unable to) (know|answer|find|see)|"
    r"no (relevant|matching|such) (records?|information|context|data)|"
    r"no [a-z]+( [a-z]+){0,4} (are|is )?recorded\b|"
    r"could ?not (find|locate|answer)|"
    r"unable to (answer|find|locate)|"
    r"insufficient (context|information|data)|"
    r"not (found|present) in (the|this) records?|"
    r"outside (the|this) (record|context)|"
    r"nothing in (the|this) record"
    r")\b",
    re.IGNORECASE,
)


def strip_abstention_citations(answer: GroundedAnswer) -> GroundedAnswer:
    """Return the answer with citations removed if it is an abstention."""
    if answer.text and answer.citations and _ABSTENTION_RE.search(answer.text):
        logger.info(
            "answer abstains (%d chars); stripping %d citations",
            len(answer.text), len(answer.citations),
        )
        return GroundedAnswer(
            text=answer.text,
            citations=[],
            dropped=answer.dropped,
            suppressed=answer.suppressed,
        )
    return answer


@dataclass
class PatientContext:
    """Everything needed to answer one question about one patient."""

    patient_id: str
    events: List[Event]
    graph: SupersessionGraph
    deidentifier: Deidentifier
    display_name: str = "the patient"
    age: Optional[int] = None


@dataclass
class AnswerResult:
    answer: GroundedAnswer
    intent: TemporalIntent
    retrieved: List[ScoredEvent]
    llm: Optional[LLMResult]
    latency_ms: float

    @property
    def has_stale_assertion(self) -> bool:
        return self.answer.has_stale_assertion


def build_context(
    patient_id: str,
    events: Sequence[Event],
    deidentifier: Optional[Deidentifier] = None,
    display_name: str = "the patient",
    age: Optional[int] = None,
) -> PatientContext:
    graph = SupersessionGraph(events)
    return PatientContext(
        patient_id=patient_id,
        events=list(events),
        graph=graph,
        deidentifier=deidentifier or Deidentifier(patient_id=patient_id),
        display_name=display_name,
        age=age,
    )


_TYPE_CUES: Tuple[Tuple[EventType, Tuple[str, ...]], ...] = (
    (EventType.MEDICATION, ("medication", "drug", "prescription", "rx", "taking",
                            "tablets", "dose", "therapy")),
    (EventType.LAB, ("lab", "labs", "blood", "test", "results", "value", "level")),
    (EventType.DIAGNOSIS, ("diagnosis", "diagnoses", "condition", "problem")),
    (EventType.VITAL, ("vital", "vitals", "bp", "pressure", "weight", "height",
                       "pulse", "temperature", "spo2")),
    (EventType.NOTE, ("note", "notes", "history", "summary", "narrative")),
    (EventType.IMAGING, ("imaging", "xray", "x-ray", "scan", "radiograph", "mri",
                         "ct", "ultrasound")),
)


def _infer_type(query: str) -> Optional[EventType]:
    """Best-guess event type from the query, used only when nothing matched."""
    lowered = (query or "").lower()
    for event_type, cues in _TYPE_CUES:
        if any(cue in lowered for cue in cues):
            return event_type
    return None


def _render_event(
    event: Event,
    validity: Validity,
    graph: SupersessionGraph,
    ref: str,
) -> str:
    """One record, rendered for the model with its validity stated explicitly.

    Telling the model that a record is superseded is not sufficient on its
    own â€" the linker enforces it â€" but it lets the model choose to explain
    the correction rather than just going silent.
    """
    label = {
        "lab": "Lab result",
        "diagnosis": "Diagnosis",
        "medication": "Medication",
        "vital": "Vital sign",
        "note": "Clinical note",
        "imaging": "Imaging report",
    }.get(str(event.event_type), "Record")

    lines = [f"[{ref}] {label} ({event.valid_time.date().isoformat()})"]
    for key, value in (event.payload or {}).items():
        lines.append(f"  {key}: {value}")

    if graph.state(event) is Validity.SUPERSEDED:
        lines.append("  STATUS: SUPERSEDED - a later record corrects this one.")
        lines.append("  Do not state this as a current fact.")
    else:
        lines.append("  STATUS: current")
    return "\n".join(lines)


def render_records_unannotated(
    events: Sequence[Event],
) -> str:
    """Records rendered in the order given, with **no** validity annotation.

    This exists for exactly one reason: to measure what ranking alone is
    worth. `AnswerPipeline` annotates every record with its status, and that
    annotation is a defence *independent* of the ranker. If both arms of a
    comparison were annotated, a difference in what the model says could not
    be attributed to the order the records arrived in, because the
    annotation would already have told the model which record to distrust.

    So a benchmark that annotates both arms is measuring the linker, not the
    ranker, and will report the ranker as contributing nothing even when the
    ranker is the component under test.

    Same payload dump and same date format as `_render_event`, so the two
    renderings differ in the one variable under test and nothing else.
    """
    label = {
        "lab": "Lab result",
        "diagnosis": "Diagnosis",
        "medication": "Medication",
        "vital": "Vital sign",
        "note": "Clinical note",
        "imaging": "Imaging report",
    }
    blocks: List[str] = []
    for i, event in enumerate(events, start=1):
        kind = label.get(str(event.event_type), "Record")
        lines = [f"[R{i}] {kind} ({event.valid_time.date().isoformat()})"]
        for key, value in (event.payload or {}).items():
            lines.append(f"  {key}: {value}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_prompt(
    query: str, rendered: List[str], intent: TemporalIntent, deid=None
) -> str:
    """Assemble the outbound prompt.

    The as-of instruction states the requested instant. That date is
    scrubbed through the same de-identifier as everything else, so the
    model receives a relative offset rather than a calendar date. The
    exact instant has already been resolved locally against the event
    store, so precision is not lost.
    """
    if intent.kind == "as_of" and intent.instant is not None:
        stamp = intent.instant.date().isoformat()
        if deid is not None:
            stamp = deid.scrub(stamp)
        instruction = (
            f"The physician is asking about the state of records as of {stamp}. "
            f"A record marked SUPERSEDED may still be the correct answer IF it "
            f"was the version valid at that date."
        )
    elif intent.kind == "historical":
        instruction = (
            "The physician is asking about history. A SUPERSEDED record may be the "
            "correct answer here, but state it in past tense and say it was later changed."
        )
    else:
        instruction = (
            "The physician is asking about the present. Use only records marked current. "
            "If a record is SUPERSEDED, report the corrected value instead."
        )

    return f"{instruction}\n\nRECORDS:\n" + "\n\n".join(rendered) + f"\n\nQUESTION: {query}"


class _SurrogateIds:
    """Short ids the model can actually copy, mapped back to real ones.

    Asking a language model to reproduce a 36-character UUID verbatim is
    unreliable, and the failure is silent: it returns something like
    "ID_02", the provenance linker cannot resolve that, and a *correct*
    answer is discarded for citing a record that does not exist.

    That is the verifier working correctly on a model that misbehaved, so
    the fix belongs here rather than in the verifier - loosening the
    check would weaken the guarantee the paper is about. A two-token
    surrogate is within what these models copy reliably, and a reference
    that still fails to resolve is left unresolved so the linker rejects
    it as unknown.
    """

    def __init__(self) -> None:
        self._to_real: Dict[str, str] = {}
        self._to_surrogate: Dict[str, str] = {}

    def assign(self, event_id: str) -> str:
        existing = self._to_surrogate.get(event_id)
        if existing is not None:
            return existing
        ref = f"R{len(self._to_real) + 1}"
        self._to_real[ref] = event_id
        self._to_surrogate[event_id] = ref
        return ref

    def resolve(self, references: Sequence[str]) -> List[str]:
        """Map surrogate references to event ids, leaving unknowns alone.

        An unresolvable reference passes through unchanged so the linker
        sees it as unknown and rejects the claim, rather than being
        dropped here and turning a fabricated citation into an uncited
        one - which would be a different and worse failure.
        """
        return [self._to_real.get(ref, ref) for ref in references]

    def rewrite(self, claims: List[Claim]) -> List[Claim]:
        return [
            Claim(text=c.text, event_ids=tuple(self.resolve(c.event_ids)))
            for c in claims
        ]


#: A question asking for everything ("summarise it all", "what all
#: diseases", "full history") cannot be answered from 8 nearest neighbours:
#: the top-k fills with near-duplicates of the dominant topic and whole
#: record types never enter the context. Seen live: 11 diagnosis events in
#: the record, zero in the top-8, so the model truthfully reported "none
#: recorded". Summary intent widens recall; precision questions keep k=8.
_SUMMARY_RE = re.compile(
    r"\b(summariz|summaris|what all|list all|everything|full history|"
    r"till date|entire record|all (diseases|diagnoses|medications|records|reports)|"
    r"timeline|over time)\b",
    re.IGNORECASE,
)

#: Recall for a summary question. 30 records is ~3k prompt tokens, well
#: inside the smallest fallback model's window.
SUMMARY_K = 30


def summary_k(query: str, default_k: int = 8) -> int:
    """Retrieval depth for one question. Broad questions get broad recall."""
    if _SUMMARY_RE.search(query or ""):
        return max(default_k, SUMMARY_K)
    return default_k


class AnswerPipeline:
    """End-to-end grounded answering."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        semantic_fn=None,
        k: int = 8,
        alpha: float = 0.70,
    ):
        self.llm = llm
        #: When None, a per-request scorer is built that uses the vector
        #: index and falls back to lexical overlap. Injecting a fixed
        #: function is what the TAP-QA harness does, so the temporal
        #: weighting can be measured without the embedding layer in play.
        self.semantic_fn = semantic_fn
        self.k = k
        self.alpha = alpha

    def _semantic_for(
        self, context: PatientContext, intent: TemporalIntent
    ) -> Callable[[str, Event], float]:
        if self.semantic_fn is not None:
            return self.semantic_fn
        as_of = None
        if intent.kind == "as_of" and intent.instant is not None:
            as_of = int(intent.instant.timestamp())
        return semantic_for(
            context.patient_id,
            as_of=as_of,
            # Superseded records must be retrievable: the generator is told
            # their state so it can explain a correction, and a historical
            # query is answered largely from them.
            include_superseded=True,
        )

    async def answer(
        self,
        context: PatientContext,
        query: str,
        as_of: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        now = now or datetime.now(timezone.utc)

        # 1. intent
        intent = parse_temporal_intent(query, now)
        if as_of is not None:
            intent = TemporalIntent(
                kind="as_of", instant=as_of, confidence=1.0, cue="explicit"
            )

        # 2 + 3. validity-aware retrieval
        retriever = ValidityAwareRetriever(
            graph=context.graph, alpha=self.alpha, enforce_future_leak=True
        )
        scorer = self._semantic_for(context, intent)
        k = summary_k(query, self.k)
        retrieved = retriever.rank(
            query,
            context.patient_id,
            context.events,
            intent=intent,
            now=now,
            semantic_fn=scorer,
            k=k,
        )

        if not retrieved:
            return AnswerResult(
                answer=GroundedAnswer(text=""),
                intent=intent,
                retrieved=[],
                llm=None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 4. de-identify before anything outbound
        surrogates = _SurrogateIds()
        rendered = [
            _render_event(
                r.event, r.validity, context.graph, surrogates.assign(r.event_id)
            )
            for r in retrieved
        ]
        prompt = self._deidentify_prompt(context, query, rendered, intent)

        # 5. generate (the model proposes)
        #
        # A single-record question needs no model at all: the answer is the
        # record. Free tiers are slow and rate-limited (17s and 429s are
        # routine), and a physician asking "what is the potassium" should
        # not wait for that. Deterministic for a single match, LLM for
        # anything that needs reasoning across records.
        llm_result: Optional[LLMResult] = None
        claims: List[Claim] = []

        direct = self._direct_answer(context, retrieved, intent)
        if direct is not None:
            return AnswerResult(
                answer=direct,
                intent=intent,
                retrieved=retrieved,
                llm=None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.llm is not None:
            try:
                llm_result = await self.llm.complete(prompt)
                raw = parse_claims(llm_result.text)
                claims = surrogates.rewrite(
                    [
                        Claim(text=c["text"], event_ids=tuple(c["event_ids"]))
                        for c in raw
                    ]
                )
            except Exception:
                # A safety block or transport failure must not surface a
                # misleading answer. Degrade to a cited extract.
                logger.warning("generation failed; degrading to extract", exc_info=True)
                claims = []

        if not claims:
            # Deterministic fallback: cited extracts, no model involvement.
            # Slower to read than a summary, but it is never wrong.
            return AnswerResult(
                answer=strip_abstention_citations(
                    self._extract_answer(retrieved, intent, query)
                ),
                intent=intent,
                retrieved=retrieved,
                llm=llm_result,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 6. provenance lock
        linker = ProvenanceLinker(context.graph, context.patient_id)
        retrieved_ids: Set[str] = {r.event_id for r in retrieved}
        verdicts = linker.verify(claims, retrieved_ids)

        # 7. assemble, then re-inject placeholders locally
        grounded = assemble(verdicts)

        # The model produced claims and the verifier rejected every one of
        # them. Returning a blank page is a safe failure but a useless one:
        # the physician asked a legitimate question and there are records
        # that answer it. A cited extract is slower to read than prose and
        # is never wrong, so it is strictly better than nothing.
        #
        # This is a degradation, not a rescue. Nothing the model wrote is
        # used, so the grounding guarantee is untouched - and the reason
        # is logged, because a model rejecting itself this often is a
        # prompt problem worth knowing about.
        if not grounded.text and retrieved:
            reasons = [r for v in verdicts for r in v.reasons]
            logger.info(
                "every generated claim was rejected; falling back to a cited "
                "extract. reasons=%s",
                reasons[:4],
            )
            grounded = self._extract_answer(retrieved, intent, query)

        if context.deidentifier:
            grounded.text = self._rehydrate(grounded.text, context)

        return AnswerResult(
            answer=strip_abstention_citations(grounded),
            intent=intent,
            retrieved=retrieved,
            llm=llm_result,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # â"€â"€ helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def _deidentify_prompt(        self,
        context: PatientContext,
        query: str,
        rendered: List[str],
        intent: TemporalIntent,
    ) -> str:
        """Scrub the query and every rendered record before any outbound call.

        The query is scrubbed too, and that matters more than it looks: a
        physician asking "as of 2026-03-12" has typed a date, and that date
        is as identifying as a name. The as-of resolution has already
        happened locally against the event store, so the model does not
        need the calendar anchor.

        Clinical values survive; identifiers do not. The LLM only ever sees
        placeholders.
        """
        deid = context.deidentifier
        return _build_prompt(
            deid.scrub(query),
            [deid.scrub(r) for r in rendered],
            intent,
            deid,
        )

    def _rehydrate(self, text: str, context: PatientContext) -> str:
        """Restore the physician's own vocabulary after the round trip.

        The model only ever saw placeholders. Substituting them back here
        means exact clinical values survive without ever being sent.

        Tolerant of a model that drops the brackets when quoting a
        placeholder ("as of T_MINUS_14D"). Leaving a raw token in a
        clinical answer is worse than removing it, so bare tokens are
        collapsed out of the prose and the fact is stated plainly.
        """
        if not text:
            return text

        import re

        if context.display_name and context.display_name != "the patient":
            text = re.sub(r"\[?PT_0\d\]?", context.display_name, text)
        if context.age is not None:
            text = re.sub(r"\[?AGE_\d+\]?", str(context.age), text)

        # A relative-time token is meaningless to a reader. Rewrite the
        # common forms into plain language rather than leaving jargon.
        months = {"T_MINUS_365D": "about a year ago", "T_MINUS_180D": "about six months ago",
                  "T_MINUS_90D": "about three months ago", "T_MINUS_30D": "about a month ago",
                  "T_MINUS_7D": "about a week ago"}

        def offset(match: "re.Match[str]") -> str:
            token = match.group(1).upper()
            if token == "T_TODAY":
                return "today"
            if token == "T_YESTERDAY":
                return "yesterday"
            if token in months:
                return months[token]
            digits = re.search(r"(\d+)", token)
            if digits:
                return f"{digits.group(1)} days ago"
            return "earlier"

        text = re.sub(r"\[?(T_(?:MINUS|PLUS)_\d+D|T_TODAY|T_YESTERDAY)\]?", offset, text)

        # Tidy spacing left behind by the substitutions.
        text = re.sub(r"\s+([.,;:])", r"\1", text)
        text = re.sub(r"  +", " ", text)
        return text.strip()


    def _direct_answer(
        self,
        context: PatientContext,
        retrieved: List[ScoredEvent],
        intent: TemporalIntent,
    ) -> Optional[GroundedAnswer]:
        """Answer without the model when one record answers the question.

        When exactly one record matches and it is unambiguous, generating
        prose adds latency and risk but no information. This is the fast
        path, and it is the common case: "what is the potassium" against a
        record that holds one potassium.
        """
        if len(retrieved) < 2:
            return None

        best = retrieved[0]
        runner_up = retrieved[1]
        # A clear gap means only one record is really relevant.
        if best.semantic < 0.5 or (best.semantic - runner_up.semantic) < 0.15:
            return None
        if best.validity is Validity.SUPERSEDED and intent.kind != "as_of":
            return None

        rows, change_notes = collapse(
            [(i.event, i.validity) for i in retrieved if i.semantic >= best.semantic * 0.5]
        )
        if len(rows) != 1:
            return None

        return self._extract_answer(
            [ScoredEvent(
                event=rows[0][0], semantic=best.semantic, validity=rows[0][1],
                weight=1.0, final=1.0,
            )],
            intent,
            "",
        )

    @staticmethod
    def _extract_answer(
        retrieved: List[ScoredEvent],
        intent: Optional[TemporalIntent] = None,
        query: str = "",
    ) -> GroundedAnswer:
        """Model-free fallback: the records themselves, clearly labelled.

        This runs whenever the free tier is unavailable, so it has to be
        genuinely readable - a physician should be able to use it directly.

        Wording depends on intent. For an as-of query the record IS the
        historical answer and must not be described as superseded, or the
        user is told their correct historical answer is wrong.
        """
        historical = intent is not None and intent.kind in ("as_of", "historical")

        lines: List[str] = []
        corrections: List[str] = []
        citations: List[Dict] = []

        # Only records that actually matched. A potassium question should
        # not answer with the patient's blood pressure.
        #
        # Two gates are needed. A relative floor (half the best score) drops
        # the weak tail when something matched. An absolute floor drops
        # everything when nothing did, which is the common case for a query
        # like "current medications?" that shares no tokens with a lab row.
        best = max((i.semantic for i in retrieved), default=0.0)
        if best > 0:
            relevant = [i for i in retrieved if i.semantic >= best * 0.5]
        else:
            # Nothing matched lexically. Fall back to the event type, so a
            # medication question still returns medications.
            wanted = _infer_type(query)
            if wanted is not None:
                typed = [i for i in retrieved if i.event.event_type is wanted]
                relevant = typed or list(retrieved)
            else:
                relevant = list(retrieved)

        # One row per drug/test, not one per version. A longitudinal record
        # legitimately holds three Metformin rows; a physician wants one line.
        rows, change_notes = collapse([(i.event, i.validity) for i in relevant])

        for event, validity in rows:
            payload = event.payload or {}
            when = event.valid_time.date().isoformat()
            was_superseded_now = validity is Validity.SUPERSEDED

            if event.event_type == EventType.LAB:
                value = payload.get("value")
                unit = payload.get("unit") or ""
                flag = payload.get("flag")
                marker = {"low": " (low)", "high": " (high)"}.get(flag, "")
                head = f"**{payload.get('test', 'Lab')}**: {value} {unit}{marker}"
                if event.source_type == SourceType.CORRECTION and not historical:
                    reason = payload.get("amendment")
                    head += f" - corrected value, {when}"
                    if reason:
                        head += f" ({reason})"
                else:
                    head += f" - {when}"
            elif event.event_type == EventType.MEDICATION:
                status = str(payload.get("status", "active")).lower()
                dose = payload.get("dose", "")
                freq = payload.get("frequency", "")
                if status in ("stopped", "discontinued"):
                    reason = payload.get("reason")
                    head = f"**{payload.get('drug')}** {dose} {freq}".rstrip()
                    head += f" - STOPPED"
                    if reason:
                        head += f" ({reason})"
                else:
                    head = f"**{payload.get('drug')}** {dose} {freq}".rstrip()
            elif event.event_type == EventType.DIAGNOSIS:
                head = f"**{payload.get('diagnosis')}**"
                if payload.get("icd10"):
                    head += f" ({payload['icd10']})"
                if payload.get("revised_from") and not historical:
                    head += f" - revised from {payload['revised_from']}"
            elif event.event_type == EventType.VITAL:
                head = (
                    f"**{payload.get('measurement')}**: "
                    f"{payload.get('value')} {payload.get('unit', '')}"
                )
            elif event.event_type == EventType.NOTE:
                note = str(payload.get("note") or "").strip()
                author = str(payload.get("author") or "").strip()
                # An empty note renders as "- -", a row that says nothing
                # and teaches the doctor to ignore the list. It still
                # happened, so it still reads - with the date carrying it.
                head = (
                    f"{note} - {author}".strip(" -")
                    or f"Note recorded {when}"
                )
            elif event.event_type == EventType.IMAGING:
                head = (
                    f"**{payload.get('study')}**: "
                    f"{payload.get('impression', payload.get('findings', ''))}"
                )
            else:
                head = f"{event.event_type}: {payload}"

            if was_superseded_now and not historical:
                # Stale value: footnote only, never the main answer.
                corrections.append(
                    f"* A previous value for this record ({when}) was later "
                    f"corrected. The figure above is the current one."
                )
            else:
                lines.append(f"- {head}")

            citations.append({"event_id": event.event_id})

        text = "\n".join(lines)

        footnotes = list(dict.fromkeys(corrections + change_notes))
        if footnotes:
            text += "\n\n" + "\n".join(footnotes)

        return GroundedAnswer(
            text=text, citations=citations, dropped=[], suppressed=[]
        )
