"""Check MedGemma can actually read a clinical image.

The encoder-level checks prove the other models work. This one exists
because a language model returning *plausible* text is not the same as
reading correctly, and the failure mode is dangerous: a wrong number
confidently produced looks identical to a right one until someone checks.

So this prints the transcription verbatim and flags the two things that
matter clinically:

  1. did it decline to read (abstain), and
  2. if it did read, are the numbers *shaped* like lab values

It does not claim accuracy - it shows the output so a human can judge.
There is no ground truth for the images used here, and inventing one
would make this check worthless.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings

settings = get_settings()
settings.enable_vision_llm = True

from app.services.vision_llm import get_medgemma, transcribe  # noqa: E402

OUT = Path(__file__).resolve().parent / "_medgemma_probe.png"


def make_probe_image() -> bytes:
    """A synthetic lab report, so the test needs no patient data.

    Rendered with PIL rather than shipped as a fixture so the numbers
    are unambiguous and it is obvious what the model was looking at.
    """
    width, height = 900, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        title = ImageFont.truetype("arial.ttf", 34)
        body = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 19)
    except OSError:
        title = body = small = ImageFont.load_default()

    draw.text((48, 40), "SUNRISE DIAGNOSTICS", fill="black", font=title)
    draw.text((48, 90), "COMPLETE BLOOD COUNT", fill="black", font=body)
    draw.text(
        (48, 130),
        "Patient: Ramesh Iyer      Sample date: 2026-03-14",
        fill="black",
        font=small,
    )
    draw.line((48, 165, width - 48, 165), fill="black", width=2)

    rows = [
        ("TEST", "RESULT", "UNIT", "REFERENCE"),
        ("Haemoglobin", "13.4", "g/dL", "13.0 - 17.0"),
        ("Total WBC", "11200", "/cumm", "4000 - 11000"),
        ("Platelets", "1.92", "lakhs", "1.5 - 4.5"),
        (" Fasting Glucose", "186", "mg/dL", "70 - 110"),
        ("Serum Creatinine", "1.1", "mg/dL", "0.7 - 1.3"),
    ]
    y = 195
    for row in rows:
        for x, text in zip((48, 380, 570, 690), row):
            draw.text((x, y), text, fill="black", font=small)
        y += 46

    draw.text(
        (48, height - 60),
        "Verified by: Dr. A. Menon, MD Pathology",
        fill="black",
        font=small,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def shape_check(value: str) -> bool:
    """Is this string shaped like a measured quantity?"""
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def main() -> int:
    medgemma = get_medgemma()

    print("MedGemma status")
    print("-" * 58)
    print(f"  weights dir : {settings.medgemma_gguf}")
    print(f"  model file  : {settings.medgemma_file}")
    print(f"  projector   : {settings.medgemma_mmproj}")
    print(f"  context     : {settings.medgemma_context}")
    print(f"  threads     : {settings.medgemma_threads}")

    started = time.time()
    if not medgemma.available(settings):
        print("\n  FAILED: model did not load.")
        print("  Check that both .gguf files exist and that the container")
        print("  can see /models.")
        return 1
    print(f"  loaded in   : {medgemma.load_seconds}s")

    # A generated image rather than a real scan: no patient data is read
    # by a model that is not in the clinical path.
    raw = make_probe_image()
    print(f"\n  probe image : {len(raw):,} bytes, {OUT.name}")
    OUT.write_bytes(raw)

    print("\nWhat the model sees (rendered):")
    print("-" * 58)
    for line in [
        "SUNRISE DIAGNOSTICS",
        "COMPLETE BLOOD COUNT",
        "Patient: Ramesh Iyer   Sample date: 2026-03-14",
        "Haemoglobin      13.4    g/dL    13.0 - 17.0",
        "Total WBC        11200   /cumm   4000 - 11000",
        "Platelets        1.92    lakhs   1.5 - 4.5",
        "Fasting Glucose  186     mg/dL   70 - 110",
        "Serum Creatinine 1.1     mg/dL   0.7 - 1.3",
    ]:
        print(f"  {line}")

    print("\nWhat the model returns:")
    print("-" * 58)
    result = transcribe(raw, settings)

    if result.error:
        print(f"  error: {result.error}")
        print(f"  raw:   {result.raw[:400]!r}")
        return 1

    print(f"  document_type : {result.document_type or '(none)'}")
    print(f"  patient_name  : {result.patient_name or '(none)'}")
    print(f"  sample_date   : {result.sample_date or '(none)'}")
    print(f"  elapsed       : {result.seconds}s")
    print()

    if not result.readable:
        print("  The model declined to read the image.")
        print("  That is the SAFE direction and the system handles it:")
        print("  the document is held for review and nothing is recorded.")
        print()
        print("VERDICT: abstained. Correct behaviour for an uncertain read,")
        print("         but this probe should have been legible.")
        return 1

    numeric = 0
    print(f"  {'TEST':<22} {'VALUE':<12} {'UNIT':<10} read")
    print("  " + "-" * 54)
    for line in result.lines:
        test = line.get("test", "")
        value = line.get("value", "")
        unit = line.get("unit", "")
        ok = shape_check(value)
        numeric += int(ok)
        print(f"  {test[:21]:<22} {value[:11]:<12} {unit[:9]:<10} {'ok' if ok else 'UNCLEAR'}")

    print("  " + "-" * 54)
    print(f"  {numeric} of {len(result.lines)} rows returned a numeric value")
    print()
    print("  Reminder: this is a reading aid, not a source of record.")
    print("  Its output is stored with committed=false and needs a doctor")
    print("  to confirm before anything reaches the event log.")
    print()
    if numeric:
        print("VERDICT: produced a transcription. Compare it against the")
        print("         rendered report above before trusting any row.")
    else:
        print("VERDICT: readable but returned no usable numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
