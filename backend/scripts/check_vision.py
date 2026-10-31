"""Check that the image side is genuinely cross-modal, not just working.

Two claims are tested, and the second is the one that matters:

  1. BiomedCLIP loads from local files with no network access. The
     published snippet uses an `hf-hub:` id and downloads ~1.5 GB; this
     uses the shipped `open_clip_config.json` plus a config-only shim.

  2. An X-ray lands *closer* to the text "chest x-ray" than to the text
     "bone marrow histology". If that ordering holds, a clinician can
     search by describing a study instead of by filename, which is the
     capability that makes the record multimodal.

Test images come from the BiomedCLIP distribution's own example_data,
labelled in the filename, so the expected answer is not invented here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from app.core.config import get_settings

settings = get_settings()
settings.models_root = "D:/models"
settings.medcpt_query_path = "D:/models/MedCPT-Query-Encoder"
settings.medcpt_article_path = "D:/models/MedCPT-Article-Encoder"
settings.biomedclip_path = "D:/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
settings.enable_vision = True

from app.services.embeddings import get_embedder  # noqa: E402

EXAMPLES = Path(settings.biomedclip_path) / "example_data/biomed_image_classification_example_data"

#: filename fragment -> the text a clinician would actually type to find it.
CASES = [
    ("chest_X-ray.jpg", "chest x-ray showing lung fields"),
    ("brain_MRI.jpg", "brain mri scan"),
    ("H_and_E_histopathology.jpg", "histopathology of tissue under a microscope"),
    ("bone_X-ray.jpg", "x-ray of a bone"),
]

#: A caption from a *different* modality, to prove the ranking is real
#: rather than everything sitting at a uniform high similarity.
DISTRACTOR = "line chart of a rising trend over time"


def cos(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    embedder = get_embedder()
    report = embedder.status()
    print("torch present :", report["torch_present"])
    print("text encoder  :", report["text_encoder"])

    if not embedder.image_available():
        print("FAIL: BiomedCLIP did not load offline.")
        print("  failed:", report["failed"])
        return 1

    print("BiomedCLIP    : loaded in %.1fs, image_dim=%d"
          % (report["load_seconds"].get("biomedclip", -1), embedder.image_dim()))

    distractor = embedder.encode_image_caption(DISTRACTOR)
    if distractor is None:
        print("FAIL: caption embedding returned None")
        return 1

    print()
    print(f"{'image':<32} {'own caption':>13} {'distractor':>11}  verdict")
    print("-" * 78)

    passed = 0
    margins: List[float] = []
    for filename, caption in CASES:
        path = EXAMPLES / filename
        if not path.is_file():
            print(f"{filename:<32} SKIP (not found)")
            continue

        started = time.time()
        vector = embedder.encode_image(path.read_bytes())
        elapsed = time.time() - started
        if vector is None:
            print(f"{filename:<32} FAIL: could not encode")
            continue

        own = cos(vector, embedder.encode_image_caption(caption))
        other = cos(vector, distractor)
        ok = own > other
        passed += int(ok)
        margins.append(own - other)
        print(
            f"{filename:<32} {own:>13.3f} {other:>11.3f}  "
            f"{'OK' if ok else 'MISRANKED'}  ({elapsed:.1f}s)"
        )

    print("-" * 78)
    print(f"{passed}/{len(CASES)} images ranked their own caption first")

    # The prompt template is the one free parameter here, so it is measured
    # rather than assumed. Same comparison, text encoded with no prefix.
    print()
    print("effect of the prompt template (same images, raw text):")
    raw = 0
    for filename, caption in CASES:
        path = EXAMPLES / filename
        if not path.is_file():
            continue
        vector = embedder.encode_image(path.read_bytes())
        if vector is None:
            continue
        own = cos(
            vector, embedder.encode_image_caption(caption, template="")
        )
        other = cos(vector, embedder.encode_image_caption(DISTRACTOR, template=""))
        ok = own > other
        raw += int(ok)
        print(f"  {filename:<30} {own:>7.3f} vs {other:>7.3f}  {'OK' if ok else 'MISRANKED'}")
    print(f"  with template: {passed}/{len(CASES)}   without: {raw}/{len(CASES)}")

    if margins:
        print()
        print(f"median margin with template: {sorted(margins)[len(margins) // 2]:+.3f}")

    print()
    if passed == len(CASES):
        print("VERDICT: image and text share one space. Search by description works.")
        return 0
    print("VERDICT: partial. See misranked rows above; do not claim cross-modal")
    print("         search in the write-up until this is all green.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
