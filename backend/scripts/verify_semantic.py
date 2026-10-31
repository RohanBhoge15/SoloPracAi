"""Verify the semantic layer against the *running* stack, over HTTP.

The model-level checks (`check_embeddings.py`, `check_vision.py`) prove the
encoders work. They do not prove the deployed API uses them. This script
answers the question that actually matters: if a doctor asks "what is her
blood sugar" in a browser right now, does the answer come from a vector
index or from token overlap?

Every step asserts on a value the server actually returned. If Qdrant is
down, or torch is missing from the image, or the collection was never
created, the `backend` field says `lexical` and this script fails rather
than quietly reporting a working system.

Run after:  docker compose up -d --build api
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

API = "http://localhost:8000"
EXAMPLES = r"D:\models\BiomedCLIP-PubMedBERT_256-vit_base_patch16_224\example_data/biomed_image_classification_example_data"

failures: List[str] = []


def req(
    method: str,
    path: str,
    payload: Any = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
) -> Tuple[int, Any]:
    """One HTTP call, with the query string encoded properly.

    Hand-built query strings break in two ways that look like server bugs:
    a space in a value is rejected outright by urllib, and a `+` in an
    ISO timestamp is read as a space, so `...T20:46:46+00:00` arrives as
    `...T20:46:46 00:00` and the server rightly refuses to parse it.
    """
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return response.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def step(number: int, title: str) -> None:
    # ASCII separators only: the Windows console codepage is cp1252 and
    # raises on box-drawing characters.
    print(f"\n{'-' * 64}\nSTEP {number}  {title}\n{'-' * 64}")


def check(label: str, passed: bool, detail: str = "") -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    if not passed:
        failures.append(label)


def main() -> int:  # noqa: C901 - a linear script is the clearest form here
    stamp = datetime.now(timezone.utc)

    # ── 1. what does this deployment claim to be able to do ──────────
    step(1, "What the deployment reports")
    status, body = req("GET", "/api/v1/semantic/status")
    if status != 200:
        print(f"  could not read status: HTTP {status} {body}")
        return 1

    embedder = body["embedder"]
    store = body["vector_store"]
    print(f"  text encoder     : {embedder['text_encoder']} ({embedder['text_dim']}-d)")
    print(f"  query tower      : loaded={embedder['query_tower_loaded']}")
    print(f"  article tower    : loaded={embedder['article_tower_loaded']} available={embedder['indexer_available']}")
    print(f"  vision enabled   : {embedder['vision_enabled']} / ready {embedder['vision_ready']}")
    print(f"  qdrant           : healthy={store['healthy']} points={store['points']}")
    print(f"  load times       : {embedder['load_seconds']}")
    if embedder.get("failed"):
        print(f"  FAILED models    : {embedder['failed']}")

    check("torch is installed in the image", embedder["torch_present"])
    check("MedCPT query tower loaded", embedder["query_tower_loaded"])
    check(
        "MedCPT article tower available (needed to index)",
        embedder["indexer_available"],
    )
    check("qdrant reachable", store["healthy"])

    if not embedder["query_tower_loaded"] or not store["healthy"]:
        print("\n  Stopping: without an encoder and a vector store there is")
        print("  nothing to verify. Check the api logs for the load failure.")
        return 1

    # ── 2. a patient with records that lexical cannot reach ──────────
    step(2, "Create a patient and add records")
    suffix = uuid.uuid4().hex[:8]
    status, patient = req(
        "POST",
        "/api/v1/patients",
        {
            "display_name": f"Semantic Check {suffix}",
            "age": 54,
            "gender": "female",
            "phone": f"+919{suffix[:7]}",
        },
    )
    check("patient created", status in (200, 201), f"HTTP {status}")
    if status not in (200, 201):
        print(f"  {patient}")
        return 1
    pid = patient["id"]

    # Deliberately chosen so that the query and the record share no tokens.
    # "blood sugar" against a record saying "HbA1c" scores exactly 0.000 on
    # token overlap, which is the whole reason the encoder exists.
    labs = [
        ("HbA1c", 7.2, "%", 400),
        ("Serum Creatinine", 1.1, "mg/dL", 300),
        ("Fasting Glucose", 186, "mg/dL", 20),
        ("Serum Potassium", 4.8, "mEq/L", 10),
    ]
    created = 0
    for test, value, unit, days_ago in labs:
        status, _ = req(
            "POST",
            f"/api/v1/patients/{pid}/events",
            {
                "event_type": "lab",
                "valid_time": (stamp - timedelta(days=days_ago)).isoformat(),
                "payload": {
                    "test": test,
                    "value": value,
                    "unit": unit,
                    "reference_low": 3.9 if test == "Serum Potassium" else None,
                    "reference_high": 5.1 if test == "Serum Potassium" else None,
                },
            },
        )
        if status in (200, 201):
            created += 1
    check("four lab records added", created == 4, f"{created}/4")

    # ── 3. index them ───────────────────────────────────────────────
    step(3, "Build the vector index from the event log")
    status, indexed = req(
        "POST", "/api/v1/semantic/reindex", params={"patient_id": pid}
    )
    check("reindex returned 200", status == 200, f"HTTP {status}")
    if status == 200:
        print(f"  events indexed : {indexed['vectors_written']} of {indexed['events']}")
        print(f"  backend        : {indexed['backend']}")
        check(
            "every record was vectorised",
            indexed["vectors_written"] == indexed["events"],
            f"{indexed['vectors_written']}/{indexed['events']}",
        )
        check("indexing ran through qdrant", indexed["backend"] == "qdrant")

    # ── 4. the paraphrase test ──────────────────────────────────────
    step(4, "Ask a question that shares no words with the record")
    query = "what is her blood sugar"
    status, found = req(
        "POST",
        "/api/v1/semantic/search",
        params={"patient_id": pid, "query": query},
    )
    check("search returned 200", status == 200, f"HTTP {status} {found if status != 200 else ''}")
    if status == 200:
        print(f"  query          : {query!r}")
        print(f"  backend        : {found['backend']}")
        print(f"  degraded       : {found['degraded']}")
        print()
        print(f"  {'score':>7}  {'validity':<12} {'record'}")
        for hit in found["hits"]:
            print(f"  {hit['score']:>7.3f}  {hit['validity']:<12} {hit['event_type']} {hit['event_id'][:8]}")

        check(
            "the vector index answered, not token overlap",
            found["backend"] == "qdrant",
            found["backend"],
        )
        check("results were returned at all", len(found["hits"]) > 0)

    # ── 5. an as-of query must not see the future ───────────────────
    step(5, "An as-of query is restricted to the past")
    past = (stamp - timedelta(days=100)).isoformat()
    status, asof = req(
        "POST",
        "/api/v1/semantic/search",
        params={"patient_id": pid, "query": "labs", "as_of": past},
    )
    check("as-of search returned 200", status == 200, f"HTTP {status}")
    if status == 200:
        cut = datetime.fromisoformat(past).timestamp()
        leaked = [
            h
            for h in asof["hits"]
            if datetime.fromisoformat(h["valid_time"]).timestamp() > cut
        ]
        check(
            "no record from after the requested instant came back",
            not leaked,
            f"{len(leaked)} leaked",
        )

    # ── 6. the correction chain still resolves ──────────────────────
    step(6, "A correction is superseded in the index, not just in the log")
    status, events = req("GET", f"/api/v1/patients/{pid}/events")
    glucose = next(
        (e for e in (events or []) if e.get("payload", {}).get("test") == "Fasting Glucose"),
        None,
    )
    if glucose is None:
        check("found the glucose record to correct", False)
    else:
        status, corrected = req(
            "POST",
            f"/api/v1/patients/{pid}/events",
            {
                "event_type": "lab",
                # A superseding record must also declare itself a
                # correction. The server enforces this pairing, which is
                # the right invariant: a record cannot revise another
                # without being marked as a revision.
                "source_type": "correction",
                "supersedes_id": glucose["id"],
                "valid_time": (stamp - timedelta(days=1)).isoformat(),
                "payload": {"test": "Fasting Glucose", "value": 112, "unit": "mg/dL"},
            },
        )
        check("correction recorded", status in (200, 201), f"HTTP {status} {corrected if status >= 400 else ''}")
        if status in (200, 201):
            req("POST", "/api/v1/semantic/reindex", params={"patient_id": pid})
            status, states = req(
                "POST",
                "/api/v1/semantic/search",
                params={
                    "patient_id": pid,
                    "query": "fasting glucose",
                    "include_superseded": "false",
                },
            )
            if status == 200:
                stale = [h for h in states["hits"] if h["validity"] == "superseded"]
                check(
                    "excluding superseded records hides the stale value",
                    not stale,
                    f"{len(stale)} superseded returned",
                )
            else:
                check("post-correction search returned 200", False, f"HTTP {status}")

    # ── 7. the image path ───────────────────────────────────────────
    step(7, "Image indexing (only if vision is enabled)")
    status, live = req("GET", "/api/v1/semantic/status")
    embedder_state = live["embedder"] if status == 200 else {}
    vision_enabled = bool(embedder_state.get("vision_enabled"))
    vision_ready = bool(embedder_state.get("vision_ready"))

    if not vision_enabled:
        print("  SKIPPED: ENABLE_VISION is off. That is the default, because")
        print("  BiomedCLIP adds ~750 MB of resident weights and ~1 s per")
        print("  image on CPU. The encoder itself is proven by")
        print("  scripts/check_vision.py, which runs it directly.")
    elif not vision_ready:
        # Distinct from 'off': this is a real failure and must not be
        # reported as an intentional skip.
        print("  FAILED: vision is enabled but BiomedCLIP did not load.")
        print(f"  models that failed: {embedder_state.get('failed')}")
        print("  This is not a skip. Check the api logs for the reason — a")
        print("  read-only /models mount is the usual cause.")
        failures.append("BiomedCLIP enabled but not loaded")
    else:
        image_path = f"{EXAMPLES}\\chest_X-ray.jpg"
        try:
            with open(image_path, "rb") as handle:
                image_bytes = handle.read()
        except OSError as exc:
            print(f"  could not read the test image: {exc}")
            image_bytes = None
        if image_bytes:
            boundary = "----solumverify"
            fields = {
                "caption": "chest radiograph, PA view",
                "findings": "no acute cardiopulmonary abnormality",
                "valid_time": (stamp - timedelta(days=30)).isoformat(),
            }
            body_bytes = b""
            for key, value in fields.items():
                body_bytes += (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode()
            body_bytes += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; '
                f'filename="chest.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
            ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

            request = urllib.request.Request(
                f"{API}/api/v1/semantic/image?patient_id={pid}",
                data=body_bytes,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    stored = json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                stored = {"error": exc.read().decode("utf-8", "replace")}

            if "error" in stored:
                print(f"  image upload failed: {stored['error']}")
                check("image indexed", False)
            else:
                print(f"  document        : {stored['document_id'][:8]}")
                print(f"  event           : {stored['event_id'][:8]}")
                print(f"  vector dim      : {stored['vector_dim']}")
                print(f"  study date      : {stored['valid_time']}")
                print(f"  vector indexed  : {stored['indexed']}")
                check("image was embedded", stored["vector_dim"] == 512)
                check("image became a real event", bool(stored.get("event_id")))
                check("image was vector-indexed", stored["indexed"])

                status, hits = req(
                    "POST",
                    "/api/v1/semantic/image-search",
                    params={
                        "patient_id": pid,
                        "description": "chest x-ray of the lungs",
                    },
                )
                check(
                    "described the study and found it",
                    status == 200 and len(hits.get("hits", [])) > 0,
                    f"HTTP {status}, "
                    f"{len(hits.get('hits', [])) if status == 200 else '-'} hits",
                )

    # ── summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("All checks passed. The running API answers from the vector index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
