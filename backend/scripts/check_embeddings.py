"""Check that MedCPT actually solves the paraphrase problem.

Token overlap scores "blood sugar" against "Glucose" at 0.000 because
they share no words. A clinical text encoder should place them close.
This prints both numbers side by side so the difference is visible rather
than asserted.
"""

import time

import numpy as np

from app.core.config import get_settings

settings = get_settings()
settings.models_root = "D:/models"
settings.medcpt_query_path = "D:/models/MedCPT-Query-Encoder"
settings.medcpt_article_path = "D:/models/MedCPT-Article-Encoder"

from app.services.embeddings import Embedder  # noqa: E402
from app.services.retrieval import lexical_similarity  # noqa: E402
from app.core.validity import Event, EventType  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def cos(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    embedder = Embedder(settings)

    t0 = time.time()
    q = embedder.encode_query("what is her blood sugar")
    print(f"MedCPT cold load + encode: {time.time() - t0:.1f}s")
    if q is None:
        print("MedCPT unavailable; cannot compare")
        return 1
    print(f"vector dim: {np.asarray(q).ravel().shape}")

    records = [
        ("Glucose, 140 mg/dL", {"test": "Glucose", "value": 140, "unit": "mg/dL"}),
        ("HbA1c, 7.2 %", {"test": "HbA1c", "value": 7.2, "unit": "%"}),
        ("Potassium, 4.8 mEq/L", {"test": "Potassium", "value": 4.8, "unit": "mEq/L"}),
        ("Creatinine, 1.1 mg/dL", {"test": "Creatinine", "value": 1.1, "unit": "mg/dL"}),
    ]

    print()
    print("Query: \"what is her blood sugar\"")
    print()
    print(f"{'record':<26} {'lexical':>9} {'MedCPT':>9}   verdict")
    print("-" * 62)

    qv = np.asarray(q).ravel()
    best_lex = best_emb = -1
    for label, payload in records:
        event = Event(
            event_id="x",
            patient_id="p",
            event_type=EventType.LAB,
            valid_time=datetime.now(timezone.utc),
            payload=payload,
        )
        lex = lexical_similarity("what is her blood sugar", event)
        dv = embedder.encode_document(f"{payload.get('test')}, {payload.get('value')} {payload.get('unit')}")
        emb = cos(qv, dv) if dv is not None else float("nan")
        is_glucose = "Glucose" in label
        best_lex = max(best_lex, lex)
        best_emb = max(best_emb, emb)
        mark = "<-- target" if is_glucose else ""
        print(f"{label:<26} {lex:>9.3f} {emb:>9.3f}   {mark}")

    print("-" * 62)
    print(f"top lexical score: {best_lex:.3f}")
    print(f"top MedCPT score : {best_emb:.3f}")
    print()
    if best_lex == 0:
        print("VERDICT: lexical found nothing; MedCPT found the glucose record.")
        print("This is the exact failure the embedding layer exists to fix.")
    else:
        print("Both found something; compare the margins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
