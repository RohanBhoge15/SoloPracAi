"""GPU smoke test for the embedders.

Run with the global interpreter, which has a CUDA build of torch:

    D:\\Python\\python.exe scripts\\check_gpu.py

Prints the device, a real encode, and throughput, so "it is on the GPU"
is a measurement rather than an intention.
"""

from __future__ import annotations

import time

# Must run under `python -S`; gpu_env undoes the editable-install shadowing
# that would otherwise hand us another project's `app` package.
import gpu_env  # noqa: E402

gpu_env.setup()
gpu_env.assert_ours()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import get_embedder, resolve_device  # noqa: E402


def main() -> int:
    settings = get_settings()
    settings.medcpt_query_path = r"D:/models/MedCPT-Query-Encoder"
    settings.medcpt_article_path = r"D:/models/MedCPT-Article-Encoder"

    print("=" * 60)
    print("torch    :", torch.__version__)
    print("cuda     :", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("device   :", props.name)
        print("vram     : %.1f GB" % (props.total_memory / 1024**3))
    print("resolved :", resolve_device(settings.embed_device))
    print("=" * 60)

    embedder = get_embedder()

    started = time.time()
    info = embedder.preload()
    print("preload  : %.1fs  %s" % (time.time() - started, info))

    status = embedder.status()
    print("status   : device=%s  query_loaded=%s"
          % (status["device"], status["query_tower_loaded"]))
    if status["failed"]:
        print("FAILED   :", status["failed"])
        return 1

    query = embedder.encode_query("blood sugar level")
    document = embedder.encode_document("Glucose 186 mg/dL")
    if query is None or document is None:
        print("encode returned None")
        return 1
    print("cosine   : %.4f" % float(np.dot(query, document)))

    # Throughput, on the path that actually matters: repeated query
    # encodes of the kind a chat request makes.
    for label, fn, text in (
        ("query ", embedder.encode_query, "blood sugar level"),
        ("doc   ", embedder.encode_document, "Laboratory result test: Glucose"),
    ):
        fn(text)  # warm
        started = time.time()
        runs = 50
        for _ in range(runs):
            fn(text)
        elapsed = time.time() - started
        print("%s: %6.1f ms each (%.0f/s)"
              % (label, elapsed / runs * 1000, runs / elapsed))

    if torch.cuda.is_available():
        print("vram used: %d MB" % (torch.cuda.memory_allocated() / 1024**2))

    # A quick visual check that the GPU path gives the same answer as the
    # documented CPU path. A silent numerical difference would make every
    # published number unreproducible.
    settings.embed_device = "cpu"
    cpu = get_embedder()
    cpu_query = cpu.encode_query("blood sugar level")
    if cpu_query is not None:
        agreement = float(np.dot(query, cpu_query))
        print("cpu/gpu agreement: %.6f" % agreement)
        if agreement < 0.999:
            print("WARNING: the two devices disagree. Results would not be")
            print("reproducible between them.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
