"""The patient header on every issued document.

The bug guarded here printed `23 / —` on every prescription. The dash was not
a value that failed to load: there is no `sex` column on the patient row and
nothing anywhere that collects one, so the format string contained a literal
em-dash. An always-empty field on a clinical document reads as "the doctor
declined to record this", which is a claim the system cannot support and the
reader has no way to disprove.
"""

from __future__ import annotations

import re

from app.documents.render import patient_block


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


PATIENT = {
    "id": "3da7f6d2-fb16-462f-8d6f-baa673531863",
    "display_name": "Ranveer Thakur",
    "age": 23,
    "phone": "1234567890",
}


class TestPatientBlock:
    def test_no_sex_column_means_no_sex_label(self):
        """The exact case from the prescription: age present, sex absent.

        Before the fix this rendered "Age / Sex  23 / —".
        """
        html = patient_block(PATIENT)
        assert "Sex" not in html
        # And the value is just the age, with no trailing separator.
        assert "23 /" not in html
        assert ">23<" in html or "23" in _text(html)

    def test_age_is_still_shown(self):
        assert "Age" in patient_block(PATIENT)
        assert "23" in _text(patient_block(PATIENT))

    def test_sex_is_rendered_when_it_exists(self):
        """Not deleted -- just no longer printed empty.

        Sex is clinically meaningful on a prescription, so the field is kept
        and appears on its own once something can populate it.
        """
        html = patient_block({**PATIENT, "sex": "F"})
        assert "Age / Sex" in html
        assert "23 / F" in _text(html)

    def test_sex_present_but_age_absent(self):
        html = patient_block({**PATIENT, "age": None, "sex": "M"})
        assert "Age / Sex" in html
        # Age falls back to the dash rather than printing "— / M" oddly.
        assert "/ M" in _text(html)

    def test_both_absent_falls_back_cleanly(self):
        html = patient_block({**PATIENT, "age": None})
        assert "Age" in html
        assert "Sex" not in html

    def test_other_cells_are_unaffected(self):
        text = _text(patient_block(PATIENT))
        assert "Ranveer Thakur" in text
        assert "1234567890" in text
        assert "3DA7F6D2" in text

    def test_no_hardcoded_slash_dash_anywhere(self):
        """Guards the whole family of this bug, not just this instance.

        Any 'x / —' in the patient header is a field the system cannot fill,
        printed as though the clinician had declined to.
        """
        html = patient_block(PATIENT)
        assert "/ —" not in html
        assert "/ -" not in html

    def test_empty_string_sex_is_treated_as_absent(self):
        """A blank string is not a value. It must not produce "23 / "."""
        html = patient_block({**PATIENT, "sex": ""})
        assert "Sex" not in html
