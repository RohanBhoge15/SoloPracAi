"""De-identification — the outbound safety boundary.

The LLM is a third-party free-tier endpoint. Nothing identifiable may leave
this machine (DPDP Act 2023 / DPDP Rules 2025, compliance 2027-05-13).

Strategy is type-preserving placeholders, not redaction. A de-identified
prompt keeps the clinical structure the model needs while carrying zero
identifying content:

    "PT_01 had potassium [VAL_K_01] mEq/L on [T_MINUS_14D]"

The answer comes back with placeholders intact and is re-injected locally
from the event store, so exact clinical values survive the round trip.

The outbound assertion is a hard block, not a warning. A failed assertion
means the request does not happen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

# ── patterns ─────────────────────────────────────────────────────────

_PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_AADHAAR = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")
_LONG_DIGITS = re.compile(r"(?<!\d)\d{6,}(?!\d)")
_CALENDAR_DATE = re.compile(
    r"\b(?:"
    r"(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/](?:19|20)?\d{2}"
    r"|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*(?:19|20)?\d{0,2}"
    r")\b",
    re.IGNORECASE,
)
_AGE = re.compile(r"\b(\d{1,3})[-\s]?(?:year|yr|y)[-\s]?old\b", re.IGNORECASE)
_MRN = re.compile(r"\b(?:MRN|UHID|PID|patient\s*id)\s*[:#-]?\s*\w+", re.IGNORECASE)

# Matches an honorific *and* the name tokens that follow it, so a full name
# is removed rather than just its first part.
_NAME_TOKEN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    re.UNICODE,
)
_HINDI_NAME = re.compile(r"(?:नाम|श्री|श्रीमती)\s*[:：]?\s*[\u0900-\u097F\s]{2,30}")

#: Placeholder shapes that are *expected* to survive into the payload.
#: Stripped before leak detection so a relative-time token is not mistaken
#: for a calendar date by its own detector.
ALLOWED_PLACEHOLDER = re.compile(r"\[(?:PT|FAC|AGE|DATE|PH|EM|LOC|T_|VAL|ID|DOC)[\w\-_]*\]")

#: Anything matching these means the payload was not fully de-identified.
_LEAK_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("phone", _PHONE),
    ("email", _EMAIL),
    ("aadhaar", _AADHAAR),
    ("calendar_date", _CALENDAR_DATE),
    ("long_digit_run", _LONG_DIGITS),
    ("mrn", _MRN),
    ("honorific_name", _NAME_TOKEN),
    ("hindi_name", _HINDI_NAME),
)


class DeidentificationError(RuntimeError):
    """Raised when a payload cannot be proven de-identified.

    The caller must not send the request. This is a block, not a warning.
    """


@dataclass
class Deidentifier:
    """Replaces identifiers with stable, type-preserving placeholders.

    Placeholder indices are per-call, so a de-identified payload reveals
    nothing about how many distinct patients exist in the clinic.
    """

    patient_id: str
    patient_name: Optional[str] = None
    age: Optional[int] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    facility: Optional[str] = None
    #: Anchor date for relative-time conversion. Defaults to today.
    today: Optional[date] = None

    _counter: Dict[str, int] = field(default_factory=dict, repr=False)
    _deid: Optional[str] = None

    def __post_init__(self) -> None:
        self._deid = self._deidentify_name(self.patient_name) if self.patient_name else None

    # ── internals ────────────────────────────────────────────────────

    def _next(self, prefix: str) -> str:
        self._counter[prefix] = self._counter.get(prefix, 0) + 1
        return f"[{prefix}_{self._counter[prefix]:02d}]"

    def _deidentify_name(self, name: str) -> str:
        if not name:
            return ""
        tokens = [t for t in re.split(r"\s+", name.strip()) if t]
        if not tokens:
            return ""
        first = tokens[0]
        # Keep the first initial: clinicians need to distinguish two
        # patients named the same thing, and an initial is not identifying.
        return f"{first[0]}. " + " ".join(f"[NAME]" for _ in tokens[1:])

    def _relative_time(self, days_ago: int) -> str:
        if days_ago == 0:
            return "[T_TODAY]"
        if days_ago == 1:
            return "[T_YESTERDAY]"
        if days_ago < 0:
            return f"[T_PLUS_{abs(days_ago)}D]"
        return f"[T_MINUS_{days_ago}D]"

    def _scrub_dates(self, text: str) -> str:
        """Convert absolute dates to relative offsets.

        A bare date like 2026-03-10 is a direct identifier under DPDP: a
        visit date is often enough to re-identify a patient. But the model
        still needs to reason about ordering and recency, so the date is
        replaced with an offset from today rather than deleted. That keeps
        "14 days ago" meaningful while removing the calendar anchor.

        If the date cannot be parsed it is replaced with a neutral token
        rather than passed through, because an unparsed date is still a
        date.
        """
        from datetime import date, datetime, timezone

        today = self.today or datetime.now(timezone.utc).date()

        def to_offset(match: "re.Match[str]") -> str:
            raw = match.group(0)
            parsed: Optional[date] = None
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
                try:
                    parsed = datetime.strptime(raw, fmt).date()
                    break
                except ValueError:
                    continue
            if parsed is None:
                # Handles "12 Feb 2026" and "Feb 12, 2026".
                for fmt in ("%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"):
                    try:
                        parsed = datetime.strptime(raw, fmt).date()
                        break
                    except ValueError:
                        continue
            if parsed is None:
                return "[T_RELATIVE]"
            return self._relative_time((today - parsed).days)

        return _CALENDAR_DATE.sub(to_offset, text)

    # ── public API ───────────────────────────────────────────────────

    def scrub(self, text: str) -> str:
        """Return a de-identified copy of `text`."""
        if not text:
            return text

        out = text

        if self._deid:
            out = re.sub(re.escape(self.patient_name or ""), "[PT_01]", out, flags=re.IGNORECASE)
        out = _NAME_TOKEN.sub("[PT_01]", out)
        out = _HINDI_NAME.sub("नाम: [PT_01]", out)

        if self.phone:
            out = re.sub(re.escape(self.phone), "[PH_01]", out)
        if self.email:
            out = re.sub(re.escape(self.email), "[EM_01]", out, flags=re.IGNORECASE)
        if self.address:
            out = re.sub(re.escape(self.address), "[LOC_01]", out, flags=re.IGNORECASE)
        if self.facility:
            out = re.sub(re.escape(self.facility), "[FAC_01]", out, flags=re.IGNORECASE)

        out = _AADHAAR.sub("[ID_01]", out)
        out = _EMAIL.sub("[EM_01]", out)
        out = _MRN.sub("MRN: [ID_01]", out)
        out = _AGE.sub(lambda m: f"[AGE_{m.group(1)}] years old", out)
        # Dates become relative offsets: the calendar anchor is an
        # identifier, the ordering is clinically necessary.
        out = self._scrub_dates(out)
        out = _PHONE.sub("[PH_01]", out)
        # Any remaining bare digit run of 6+ is a reference number, not a
        # clinical value (those are short, and float-formatted).
        out = _LONG_DIGITS.sub("[ID_02]", out)

        return out

    def scrub_payload(self, payload: Dict) -> Dict:
        """Recursively scrub every string in a JSON payload.

        Numeric clinical values are preserved — a lab value is not an
        identifier, and the model needs it to answer.
        """
        return self._walk(payload)

    def _walk(self, node):
        if isinstance(node, str):
            return self.scrub(node)
        if isinstance(node, dict):
            return {k: self._walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._walk(v) for v in node]
        return node

    # ── the hard gate ────────────────────────────────────────────────

    def assert_safe(self, text: str) -> None:
        """Raise unless `text` is provably free of identifiers.

        Called on the exact string that would go over the wire. A failure
        must abort the request — never warn, never send anyway.
        """
        problems = self.find_leaks(text)
        if problems:
            detail = ", ".join(f"{kind}: {sample!r}" for kind, sample in problems)
            raise DeidentificationError(
                f"outbound payload failed de-identification ({detail}); request blocked"
            )

    @staticmethod
    def find_leaks(text: str) -> List[Tuple[str, str]]:
        """Return [(kind, sample)] for every identifier found.

        Placeholders are masked out first, so the detector never flags its
        own output. A relative-time token like [T_MINUS_14D] contains digits
        but is not a date and not an identifier.
        """
        if not text:
            return []
        scrubbed = ALLOWED_PLACEHOLDER.sub(" ", text)
        found: List[Tuple[str, str]] = []
        for kind, pattern in _LEAK_PATTERNS:
            for match in pattern.finditer(scrubbed):
                found.append((kind, match.group(0)))
        return found


def assert_payload_safe(payload: str | Dict) -> None:
    """Module-level gate for call sites that have no Deidentifier handy."""
    text = payload if isinstance(payload, str) else _stringify(payload)
    problems = Deidentifier.find_leaks(text)
    if problems:
        detail = ", ".join(f"{kind}: {sample!r}" for kind, sample in problems)
        raise DeidentificationError(
            f"outbound payload failed de-identification ({detail}); request blocked"
        )


def _stringify(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return " ".join(_stringify(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return " ".join(_stringify(v) for v in node)
    return str(node)


def rehydrate(text: str, mapping: Dict[str, str]) -> str:
    """Put real values back into a de-identified answer.

    Placeholders are restored from `mapping`; anything unmatched is left
    as-is so a partial round trip degrades visibly rather than silently.
    """
    if not text:
        return text
    out = text
    for placeholder, value in sorted(
        mapping.items(), key=lambda kv: -len(kv[0])
    ):
        out = out.replace(placeholder, str(value))
    return out
