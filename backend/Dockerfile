# SoloPrac backend
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    MODELS_ROOT=/models

WORKDIR /app

# Build deps, plus the native libraries WeasyPrint needs for CSS paged
# media (Pango, cairo, harfbuzz). The pdf layer is optional at runtime:
# if these are missing the API still serves, and pdf_available() reports
# false instead of the app failing to start.
# `libgl1` and `libglib2.0-0` are for the OCR step below: RapidOCR pulls in
# OpenCV, whose import fails on a headless image without libGL.so.1. The
# error surfaces as "ImportError: libGL.so.1: cannot open shared object
# file" from deep inside the OCR package, which reads like a broken wheel
# rather than a missing system library.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq5 \
        curl \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi-dev \
        shared-mime-info \
        fonts-dejavu-core \
        fonts-liberation \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Base dependencies first, so a code-only change does not reinstall them.
RUN pip install --upgrade pip && pip install .

# Embedding stack, as its own layer. torch AND torchvision must be
# installed together from the same CPU index: a torchvision resolved from
# PyPI is built against CUDA torch, its C++ ops then fail to register, and
# transformers reports the resulting ImportError as the deeply misleading
# "Could not import module 'BertModel'" - so the text encoder looks broken
# when the real fault is an ABI mismatch three layers down. The CPU index
# also avoids ~2.5 GB of unusable nvidia-* wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4" "torchvision>=0.19" \
    && pip install "transformers>=4.44" "open_clip_torch>=2.24"

# MedGemma, for scans OCR cannot read. Installed from the maintainer's
# prebuilt CPU wheel index: building llama.cpp from source here would need a
# toolchain the image does not carry. Optional at runtime - without it the
# capture path still abstains and routes to human review.
RUN pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        --only-binary=:all: "llama-cpp-python>=0.2.62" \
    || echo "llama-cpp-python unavailable; MedGemma stays off and OCR abstention still applies"

# OCR, for photographing a lab report.
#
# This was missing, and its absence was invisible in the worst possible
# way. `app/services/ocr.py` raises `OCRUnavailable`, `capture.py` catches
# it and treats the image as unreadable, and every single upload then
# abstained with confidence 0.00 - which is exactly the correct-looking
# behaviour for an unreadable photograph. Nothing errored, nothing was
# logged above a warning, and the UI cheerfully reported "HELD FOR REVIEW"
# for a perfectly legible report. The safety gate was working; the
# capability behind it simply was not there.
#
# RapidOCR rather than pytesseract: the ONNX models ship inside the wheel,
# so there is no tesseract binary, no language packs to fetch and no extra
# apt layer. It is CPU-only, which matches the rest of this image.
RUN pip install "rapidocr_onnxruntime>=1.3" \
    || echo "rapidocr unavailable; the capture path abstains and routes to human review"

COPY app ./app
COPY scripts ./scripts

# Build-time checks. A syntax error or an ABI mismatch must fail the build
# rather than the first request.
#
# The torchvision probe is deliberate. A CUDA-built torchvision against a
# CPU torch imports far enough to register no operators, and transformers
# surfaces that as "Could not import module 'BertModel'" - which reads like
# a broken text encoder and sends you looking in entirely the wrong place.
# Importing torchvision directly is the only check that names the real
# fault.
#
# The RapidOCR import is here for the same reason, and it is the check that
# would have caught the missing OCR dependency: with no engine installed,
# the application still starts, still serves, and still answers every
# upload with a confident "HELD FOR REVIEW". Only an import at build time
# turns that into a failed build instead of a silent capability gap.
RUN python -c "import app.main; print('import ok')" \
    && python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)" \
    && python -c "from transformers import AutoModel; print('transformers ok')" \
    && python -c "from rapidocr_onnxruntime import RapidOCR; print('rapidocr ok')"

EXPOSE 8000

# --workers 1 deliberately: the encoders are loaded once per process and
# held in memory, so N workers means N copies of ~430 MB of weights on a
# 16 GB box that is already running Postgres, Qdrant, Redis and SeaweedFS.
# Scale with threads until that becomes a measured problem.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
