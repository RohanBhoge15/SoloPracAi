"""Embeddings and cross-modal encoding.

The problem this solves, concretely: a doctor asks about "blood sugar" and
the record says "Glucose". Token overlap scores that 0.000. MedCPT places
them at cosine 0.825 in 768 dimensions, because it was trained on PubMed
queries rather than on literal string matches. `scripts/check_embeddings.py`
prints both numbers side by side.

Design constraints, all of them real on this host:

  - **No GPU.** The machine reports only a virtual display adapter, so
    everything runs on CPU. Models load lazily and only on first use,
    because a cold start of three encoders would take minutes and pin a
    worker.

  - **The query tower and the article tower are separate models.**
    MedCPT was trained with a distinct query and article encoder. Using the
    article tower to encode a clinician's question measurably degrades
    retrieval, so they are kept apart rather than shared.

  - **Only the query tower is preloaded.** It answers every request, so it
    is warmed at boot. The article tower is only needed when indexing, and
    holding ~430 MB of weights for indexing alone is a poor trade on a
    16 GB machine that also runs Postgres, Qdrant and SeaweedFS.

  - **Degrade, never fail.** If torch or a model directory is missing,
    `text_available()` returns False and the retriever falls back to
    lexical similarity. The product still works, it is just less clever.
    A hard failure here would take the whole API down.

The image side is the same idea applied to pixels: BiomedCLIP embeds a
radiograph into the same neighbourhood as the report text describing it,
which is what makes "find prior studies with similar findings" possible at
all. It loads from the local `open_clip_config.json` rather than the HF hub
id, so vision works with no network.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: MedCPT's context window. Truncating here rather than inside the model
#: makes a long clinical note degrade predictably instead of being
#: silently truncated by a default the caller cannot see.
MAX_TOKENS = 512

#: BiomedCLIP's context length, from its own open_clip_config.json.
CLIP_CONTEXT = 256

#: Prefix applied to text before it enters the joint space. BiomedCLIP's
#: own example notebook uses "this is a photo of "; the clinical phrasing
#: keeps the same shape while naming the modality. Pass `template=""` to
#: embed raw text and compare.
CLIP_PROMPT = "this is a medical image of "


def resolve_device(preference: str = "auto") -> str:
    """Pick the device to run on.

    "auto" uses CUDA when torch reports it, because the encoders are the
    only genuinely parallel part of this system and running MedCPT on CPU
    costs seconds per query. An explicit "cpu" is honoured, which is what
    the Docker image wants - the container gets no GPU, and asking for
    one there would raise rather than quietly falling back.
    """
    if preference and preference != "auto":
        return preference
    torch = _registry.torch
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _mean_pool(hidden, mask, torch) -> np.ndarray:
    """Mean pooling over non-padding tokens. MedCPT's convention.

    Returns shape (batch, dim); callers ravel, since a single query is
    passed as a one-element batch.
    """
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    summed = torch.sum(hidden * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return (summed / counts).detach().cpu().numpy().astype(np.float32)


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector
    return (vector / norm).astype(np.float32)


def _l2_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row on its own.

    `_l2` normalises the whole array, which is right for one vector and
    wrong for a stack of them: it would divide every row by the norm of
    the batch. MedCPT's cosine scores are only comparable if each vector
    is a unit vector on its own, so a batch needs the row-wise version.
    """
    if matrix.ndim == 1:
        return _l2(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    return (matrix / safe).astype(np.float32)


def _load_bert(path: str, torch):
    """Load a BERT tower as fp32, on either transformers major version.

    transformers 5 renamed the `torch_dtype` argument of `from_pretrained`
    to `dtype` and removed the old spelling; 4.x accepts only the old one.
    This project is developed against 5.x in a container but the only
    CUDA-enabled torch on the machine belongs to a 4.x install, so both
    spellings have to work. Probing the signature is cheap and exact,
    where as catching the TypeError would also swallow genuine load
    failures and report them as a version problem.
    """
    import inspect

    from transformers import AutoModel

    kwargs: Dict[str, object] = {}
    try:
        params = inspect.signature(AutoModel.from_pretrained).parameters
        # It is **kwargs on both, so fall back to probing the docs-free
        # way: try the modern name and only downgrade on the exact error.
        if "dtype" in params:
            kwargs["dtype"] = torch.float32
    except (TypeError, ValueError):  # pragma: no cover - defensive
        kwargs["dtype"] = torch.float32

    try:
        return AutoModel.from_pretrained(path, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModel.from_pretrained(path, torch_dtype=torch.float32)


class _ModelRegistry:
    """Loaded encoders behind a lock, populated on first use.

    Model loading is slow and not thread-safe. This makes it happen once,
    on one thread, so two concurrent first-requests do not each pay a
    15-second load.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._query_model = None
        self._article_model = None
        self._bge = None
        self._clip = None
        self._tokenizer = None
        self._failed: set[str] = set()
        self.load_seconds: Dict[str, float] = {}

    # â”€â”€ torch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def torch(self):
        if "torch" in self._failed:
            return None
        try:
            import torch

            return torch
        except ImportError:
            logger.warning("torch not installed; embeddings unavailable")
            self._failed.add("torch")
            return None

    # â”€â”€ MedCPT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _load_tower(self, settings: Settings, which: str):
        """Load one MedCPT tower. `which` is 'query' or 'article'."""
        attribute = f"_{which}_model"
        existing = getattr(self, attribute)
        if existing is not None:
            return existing
        if which in self._failed:
            return None

        with self._lock:
            existing = getattr(self, attribute)
            if existing is not None:
                return existing

            torch = self.torch
            if torch is None:
                self._failed.add(which)
                return None
            device = resolve_device(settings.embed_device)

            path = (
                settings.medcpt_query_path
                if which == "query"
                else settings.medcpt_article_path
            )
            if not Path(path).is_dir():
                logger.warning("MedCPT %s tower missing at %s", which, path)
                self._failed.add(which)
                return None

            try:
                from transformers import AutoTokenizer

                started = time.time()
                if self._tokenizer is None:
                    # The towers share a vocabulary, so one tokenizer serves both.
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        settings.medcpt_query_path
                    )
                model = _load_bert(path, torch)
                model = model.to(device)
                model.eval()
                setattr(self, attribute, model)
                self.load_seconds[f"medcpt_{which}"] = round(
                    time.time() - started, 2
                )
                logger.info(
                    "MedCPT %s tower loaded in %.1fs on %s",
                    which,
                    self.load_seconds[f"medcpt_{which}"],
                    device,
                )
                return model
            except Exception as exc:  # noqa: BLE001
                logger.warning("MedCPT %s tower failed: %s", which, exc)
                self._failed.add(which)
                return None

    def load_medcpt(self, settings: Settings, which: str = "query"):
        return self._load_tower(settings, which)

    # â”€â”€ BGE-M3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load_bge(self, settings: Settings):
        if self._bge is not None:
            return self._bge
        if "bge" in self._failed:
            return None
        with self._lock:
            if self._bge is not None:
                return self._bge
            if not Path(settings.bge_m3_path).is_dir():
                logger.info("BGE-M3 not present at %s", settings.bge_m3_path)
                self._failed.add("bge")
                return None
            try:
                from sentence_transformers import SentenceTransformer

                started = time.time()
                self._bge = SentenceTransformer(
                    settings.bge_m3_path, device=resolve_device(settings.embed_device)
                )
                self.load_seconds["bge_m3"] = round(time.time() - started, 2)
                logger.info("BGE-M3 loaded in %.1fs", self.load_seconds["bge_m3"])
                return self._bge
            except Exception as exc:  # noqa: BLE001
                logger.warning("BGE-M3 failed: %s", exc)
                self._failed.add("bge")
                return None

    # â”€â”€ BiomedCLIP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load_biomedclip(self, settings: Settings):
        """Load BiomedCLIP from disk, with no network access.

        The published snippet calls
        `create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-...')`,
        which re-downloads ~1.5 GB even when the weights are already on
        disk. `open_clip` supports a `local-dir:` schema that reads the
        shipped `open_clip_config.json`, merges the shipped
        `preprocess_cfg`, and picks up the checkpoint sitting beside it â€”
        no network at all.

        One catch: that config's `text_cfg.hf_model_name` still names a
        hub repo, so the BERT text tower's *architecture* would be fetched
        even though its weights come from the local checkpoint. The
        prepared directory points that field at a synthesised config-only
        shim instead. The text tower is a plain BERT-base, so its shape is
        fully determined and no hub weights are needed.
        """
        if self._clip is not None:
            return self._clip
        if "biomedclip" in self._failed:
            return None

        with self._lock:
            if self._clip is not None:
                return self._clip
            if self.torch is None:
                self._failed.add("biomedclip")
                return None

            root = Path(settings.biomedclip_path)
            if not (root / "open_clip_config.json").is_file():
                logger.info("BiomedCLIP not present at %s", root)
                self._failed.add("biomedclip")
                return None

            try:
                import open_clip
                from open_clip.tokenizer import HFTokenizer

                started = time.time()

                # Everything the app derives is written to a writable cache.
                # The model directory is an input and is mounted read-only,
                # so writing into it fails the load outright.
                prepared = _ensure_offline_clip_config(
                    root, settings.embedder_cache
                )

                # No checkpoint is placed in the prepared directory, so
                # `create_model_and_transforms` returns an initialised
                # architecture. The real weights are then loaded from the
                # read-only source path directly, which avoids copying or
                # symlinking 765 MB across a volume boundary.
                # `pretrained_text=False` is load-bearing, not a default.
                # Because the prepared directory holds no checkpoint,
                # open_clip sets `text_cfg.hf_model_pretrained = True`
                # (factory.py:476) and then tries to fetch the text tower's
                # weights from the shim directory, which contains only a
                # config — so the load dies with "no file named
                # model.safetensors". This flag is what stops that.
                model, _, preprocess = open_clip.create_model_and_transforms(
                    f"local-dir:{prepared}",
                    pretrained_text=False,
                    pretrained_image=False,
                )
                weights = _locate_checkpoint(root)
                if weights is None:
                    raise FileNotFoundError(f"no CLIP checkpoint in {root}")
                _load_biomedclip_weights(model, str(weights))
                model.eval()
                # open_clip logs "Model initialized randomly" above, because
                # the prepared directory holds no checkpoint. That is
                # expected and immediately undone by the load above; said
                # here so an operator reading the logs does not read it as a
                # failure.
                logger.info(
                    "BiomedCLIP weights loaded from %s", weights.name
                )

                # The log line above cannot distinguish a good load from a
                # bad one, and a randomly initialised tower is worse than
                # no tower at all: it encodes happily, so every caller sees
                # a vector and no error. Verified rather than assumed.
                if not _verify_clip_weights(model, self._clip or {}):
                    raise RuntimeError(
                        "BiomedCLIP weights did not load; refusing to serve "
                        "random image embeddings"
                    )

                # The wrapper, not AutoTokenizer: it applies BiomedCLIP's
                # 256-token context window and returns tensors. A bare HF
                # tokenizer returns ragged lists, which `encode_text` cannot
                # consume.
                tokenizer = HFTokenizer(str(root), context_length=CLIP_CONTEXT)

                self._clip = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "preprocess": preprocess,
                    "dim": _clip_embed_dim(prepared),
                }
                self.load_seconds["biomedclip"] = round(time.time() - started, 2)
                logger.info(
                    "BiomedCLIP loaded offline from %s in %.1fs",
                    root,
                    self.load_seconds["biomedclip"],
                )
                return self._clip
            except Exception as exc:  # noqa: BLE001
                logger.warning("BiomedCLIP failed: %s", exc)
                self._failed.add("biomedclip")
                return None

    # â”€â”€ shared â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reset_failures(self) -> None:
        """Clear the failure cache so a later attempt can retry."""
        with self._lock:
            self._failed.clear()


_registry = _ModelRegistry()


#: PubMedBERT ships a `vocab.txt` of ~28 895 tokens but allocates its word
#: embedding matrix at 30 522 rows â€” the standard BERT-base width. The
#: surplus rows are unreachable padding. Sizing the config from `vocab.txt`
#: therefore produces a shape mismatch on load, so the layer width is used.
BIOMEDBERT_VOCAB_SIZE = 30_522


def _ensure_text_config(cache_root: Path) -> Path:
    """Write a config-only shim so the BERT text tower builds offline.

    BiomedCLIP's text tower is
    `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` â€” a BERT-base.
    Only the *shape* is needed, because the real weights are loaded from
    the local checkpoint, so a synthesised config is enough to satisfy
    `AutoConfig` with no network access.

    `vocab_size` comes from `BIOMEDBERT_VOCAB_SIZE` and deliberately not
    from the model's own `vocab.txt`: the tokenizer file is smaller than
    the embedding layer, and trusting it yields a checkpoint that cannot
    be loaded at all.
    """
    shim_dir = cache_root / "biomedbert_text_cfg"
    shim_config = {
        "architectures": ["BertModel"],
        "model_type": "bert",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": BIOMEDBERT_VOCAB_SIZE,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    }
    _write_json_if_changed(shim_dir / "config.json", shim_config)
    return shim_dir


def _ensure_offline_clip_config(root: Path, cache_root: str) -> Path:
    """Return a writable directory `open_clip` can load with `local-dir:`.

    Two problems are solved here, and both come from `open_clip`'s
    published snippet assuming a network is available.

    The shipped directory is already almost usable: it has the config, the
    preprocess settings and the checkpoint. Only
    `text_cfg.hf_model_name` still names a hub repo, so the BERT text
    tower's *architecture* would be fetched even though its weights are
    local. That field is redirected to a synthesised shim.

    Nothing is written inside `root`. The model directory is an input and
    is mounted read-only in Docker, so a write there aborts the load with
    "Read-only file system" and the encoder silently never comes up â€” the
    exact failure this function was restructured to fix.

    The checkpoint is deliberately neither copied nor linked here. It
    stays where it is and is loaded from its original path, so preparing
    the config costs a few kilobytes instead of 765 MB.
    """
    cache = Path(cache_root) / "biomedclip"
    cache.mkdir(parents=True, exist_ok=True)

    with (root / "open_clip_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    shim = _ensure_text_config(Path(cache_root))
    model_cfg = config["model_cfg"]
    model_cfg.setdefault("text_cfg", {})
    model_cfg["text_cfg"]["hf_model_name"] = str(shim)
    # Weights come from the local checkpoint, not from a hub snapshot.
    model_cfg["text_cfg"]["hf_model_pretrained"] = False

    _write_json_if_changed(cache / "open_clip_config.json", config)
    return cache


def _write_json_if_changed(target: Path, payload: Any) -> None:
    """Write only when the content differs.

    Keeps repeat loads from rewriting a 765 MB-adjacent cache on every
    process start, and makes the operation idempotent under a read-only
    source directory.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(payload, indent=2)
    if target.is_file():
        try:
            if target.read_text(encoding="utf-8") == serialised:
                return
        except OSError:
            pass
    target.write_text(serialised, encoding="utf-8")
def _load_biomedclip_weights(model, checkpoint_path: str) -> None:
    """Load the CLIP checkpoint, tolerating exactly one known difference.

    Strict loading fails on a legitimate checkpoint: the published weights
    contain `position_ids` buffers for the BERT text tower, and
    transformers v5 no longer registers them as persistent buffers, so
    they arrive as unexpected keys.

    Those keys are safe to drop — they were derived from `position_ids`
    arithmetic and are recomputed at run time. Everything else is not, so
    the load stays strict and any *other* discrepancy raises rather than
    quietly producing a model with randomly initialised layers. A silently
    partial load would still encode images; it would just be wrong.
    """
    from open_clip.factory import load_state_dict

    state = load_state_dict(checkpoint_path, device="cpu")

    droppable = {
        key
        for key in state
        if key.endswith("position_ids") and "embeddings" in key
    }
    for key in droppable:
        state.pop(key)

    result = model.load_state_dict(state, strict=False)
    unexpected = set(result.unexpected_keys)
    missing = set(result.missing_keys)

    # Keys the model declares that the checkpoint legitimately omits.
    tolerated_missing = {
        key for key in missing if "logit_scale" in key or "visual.proj" in key
    }
    missing -= tolerated_missing
    unexpected -= {
        key
        for key in unexpected
        if key.endswith("position_ids") and "embeddings" in key
    }

    if missing or unexpected:
        raise RuntimeError(
            "BiomedCLIP checkpoint does not match the architecture: "
            f"missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]}"
        )
    if droppable:
        logger.debug(
            "dropped %d position_ids buffer(s) absent from this "
            "transformers version",
            len(droppable),
        )


def _verify_clip_weights(model, clip: Dict[str, Any]) -> bool:
    """Prove the CLIP weights are real rather than randomly initialised.

    `open_clip` logs "Model initialized randomly" whenever the directory it
    is given holds no checkpoint, which is exactly what this project does -
    it prepares a config-only directory on purpose and loads the checkpoint
    from the read-only source path afterwards. So that log line is expected
    here and says nothing on its own.

    The real failure is different and was invisible for a long time: a
    silently random vision tower still encodes images, so every call returns
    a vector and nothing raises. The only symptom is that image retrieval
    returns nonsense - and `image_available()` cheerfully reports True,
    because from the caller's point of view it is.

    So check the weights directly. A handful of known parameters are read
    out of the state dict; a randomly initialised tower has near-zero
    standard deviation, because those layers initialise from a narrow normal
    and only ever see a few hundred updates.
    """
    import torch as _torch

    parameters = model.state_dict()
    probes: List[str] = []
    for key in ("visual.conv1.weight", "visual.class_embedding",
                "visual.transformer.resblocks.0.attn.in_proj_weight",
                "text_model.encoder.layer.0.attention.self.query.weight"):
        if key in parameters:
            probes.append(key)
    # Fall back to whatever the vision tower does have, so the check does
    # not depend on the exact naming this open_clip version chose.
    if not probes:
        probes = [k for k in parameters if k.startswith("visual.")][:4]

    if not probes:
        return False

    for key in probes:
        tensor = parameters[key]
        if not hasattr(tensor, "float"):
            continue
        if float(tensor.float().std()) < 1e-3:
            logger.error(
                "BiomedCLIP vision/text weights look randomly initialised "
                "(%s std=%.2e). Image embeddings would be meaningless; "
                "refusing to report the model as available.",
                key,
                float(tensor.float().std()),
            )
            return False
    return True


def _locate_checkpoint(root: Path) -> Optional[Path]:
    """Find the CLIP weights, preferring safetensors as `open_clip` does."""
    preferred = [
        "open_clip_model.safetensors",
        "open_clip_pytorch_model.safetensors",
        "open_clip_pytorch_model.bin",
        "open_clip_pytorch_model.pth",
    ]
    found = [
        root / name
        for name in preferred
        if (root / name).is_file()
    ]
    if found:
        return found[0]
    for pattern in ("*.safetensors", "*.bin", "*.pth"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _clip_embed_dim(local_dir: Path) -> int:
    with (local_dir / "open_clip_config.json").open(encoding="utf-8") as handle:
        return int(json.load(handle)["model_cfg"].get("embed_dim", 512))


@dataclass
class Encoded:
    vector: np.ndarray
    dim: int


class Embedder:
    """Text and image encoding, with a lexical fallback.

    Instances are cheap and stateless apart from the process-wide model
    registry, so one is created per request rather than shared.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    # â”€â”€ availability â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def _uses_bge(self) -> bool:
        return (
            self.settings.text_encoder == "bge-m3"
            or self.settings.enable_multilingual
        )

    def text_available(self) -> bool:
        if self._uses_bge:
            return _registry.load_bge(self.settings) is not None
        return _registry.load_medcpt(self.settings, "query") is not None

    def indexer_available(self) -> bool:
        """Whether records can be embedded, not merely queried."""
        if self._uses_bge:
            return _registry.load_bge(self.settings) is not None
        return _registry.load_medcpt(self.settings, "article") is not None

    def image_available(self) -> bool:
        if not self.settings.enable_vision:
            return False
        return _registry.load_biomedclip(self.settings) is not None

    # â”€â”€ text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def encode_query(self, text: str) -> Optional[np.ndarray]:
        """Encode a clinician's question with the query tower."""
        return self._encode(text, query=True)

    def encode_document(self, text: str) -> Optional[np.ndarray]:
        """Encode a record with the article tower."""
        return self._encode(text, query=False)

    def _encode(self, text: str, query: bool) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None

        if self._uses_bge:
            vector = self._encode_bge(text)
            if vector is not None:
                return vector

        torch = _registry.torch
        if torch is None:
            return None
        model = _registry.load_medcpt(self.settings, "query" if query else "article")
        tokenizer = _registry._tokenizer
        if model is None or tokenizer is None:
            return None

        try:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            # The batch has to live on the same device as the weights, or
            # the first forward pass raises.
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            pooled = _mean_pool(
                out.last_hidden_state, inputs["attention_mask"], torch
            )
            return _l2(pooled[0])
        except Exception:  # noqa: BLE001
            logger.debug("medcpt encode failed", exc_info=True)
            return None

    def _encode_bge(self, text: str) -> Optional[np.ndarray]:
        model = _registry.load_bge(self.settings)
        if model is None:
            return None
        try:
            vector = model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vector[0], dtype=np.float32)
        except Exception:  # noqa: BLE001
            logger.debug("bge encode failed", exc_info=True)
            return None

    def encode_texts(
        self, texts: Sequence[str], query: bool = False
    ) -> Optional[np.ndarray]:
        """Encode a batch in a single forward pass.

        Returns None if any row fails, so a caller can fall back to
        per-row encoding rather than indexing a ragged array.
        """
        rows = [t for t in texts if t and t.strip()]
        if not rows:
            return None
        if len(rows) == 1:
            vector = self._encode(rows[0], query=query)
            return None if vector is None else vector.reshape(1, -1)

        matrix = self._encode_batch(rows, query=query)
        if matrix is not None:
            return matrix

        # One bad row must not cost the whole batch, and the fallback is
        # only reached when the batched path itself failed.
        vectors = [self._encode(text, query=query) for text in rows]
        if any(v is None for v in vectors):
            return None
        return np.vstack(vectors)

    def _encode_batch(
        self, rows: Sequence[str], query: bool
    ) -> Optional[np.ndarray]:
        """One tokenizer call, one forward pass, for the whole batch.

        This exists because the obvious implementation - a loop over
        `_encode` - throws away almost all of the available speed. On the
        GPU it measured 35.4 ms per record either way, so the batch was
        paying sixteen forward passes to save none of them.
        """
        if self._uses_bge:
            model = _registry.load_bge(self.settings)
            if model is not None:
                try:
                    return np.asarray(
                        model.encode(
                            list(rows),
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=len(rows),
                        ),
                        dtype=np.float32,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("bge batch encode failed", exc_info=True)
                    return None

        torch = _registry.torch
        if torch is None:
            return None
        model = _registry.load_medcpt(self.settings, "query" if query else "article")
        tokenizer = _registry._tokenizer
        if model is None or tokenizer is None:
            return None

        try:
            inputs = tokenizer(
                list(rows),
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            pooled = _mean_pool(
                out.last_hidden_state, inputs["attention_mask"], torch
            )
            return _l2_rows(pooled)
        except Exception:  # noqa: BLE001
            logger.debug("medcpt batch encode failed", exc_info=True)
            return None

    def dim(self) -> int:
        return 1024 if self._uses_bge else self.settings.qdrant_text_dim

    # â”€â”€ image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def encode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Embed a radiograph or scan with BiomedCLIP's vision tower."""
        if not self.settings.enable_vision:
            return None
        clip = _registry.load_biomedclip(self.settings)
        if clip is None:
            return None
        torch = _registry.torch
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = clip["preprocess"](image).unsqueeze(0)
            with torch.no_grad():
                features = clip["model"].encode_image(tensor)
            return _l2(features.cpu().numpy()[0])
        except Exception:  # noqa: BLE001
            logger.debug("biomedclip image encode failed", exc_info=True)
            return None

    def encode_image_caption(
        self, caption: str, template: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Embed text in the image (joint) space.

        This is what makes cross-modal search work: a query about
        'cardiomegaly' lands near a radiograph that shows it, because both
        are projected into one space by the same training run.

        The prompt template is not decoration. BiomedCLIP was trained with
        captions wrapped in a fixed prefix, so text encoded without one
        sits slightly off the distribution the image tower was aligned to.
        `scripts/check_vision.py` measures the effect rather than assuming
        it.
        """
        if not self.settings.enable_vision:
            return None
        clip = _registry.load_biomedclip(self.settings)
        if clip is None:
            return None
        torch = _registry.torch
        prompt = template if template is not None else CLIP_PROMPT
        text = f"{prompt}{caption}" if prompt else caption
        try:
            tokens = clip["tokenizer"]([text])
            with torch.no_grad():
                features = clip["model"].encode_text(tokens)
            return _l2(features.cpu().numpy()[0])
        except Exception:  # noqa: BLE001
            logger.debug("biomedclip text encode failed", exc_info=True)
            return None

    def image_dim(self) -> int:
        clip = _registry.load_biomedclip(self.settings) if self.settings.enable_vision else None
        if clip is None:
            return self.settings.qdrant_image_dim
        return int(clip["dim"])

    # â”€â”€ warm-up and reporting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def preload(self) -> Dict[str, Any]:
        """Load the query tower, off the request path.

        Called from the FastAPI lifespan. Without this, the first question
        after a restart blocks for ~15 s, which reads as a hang rather than
        as a cold start.
        """
        started = time.time()
        if self._uses_bge:
            self.text_available()
        else:
            _registry.load_medcpt(self.settings, "query")
        return {
            "ready": self.text_available(),
            "encoder": self.settings.text_encoder,
            "seconds": round(time.time() - started, 2),
        }

    def status(self) -> Dict[str, Any]:
        # `*_loaded` means "resident right now"; `*_available` means "can be
        # loaded if asked". Conflating the two reports a working indexer as
        # broken simply because nothing has needed to write a vector yet.
        return {
            "text_encoder": self.settings.text_encoder,
            "query_tower_loaded": _registry._query_model is not None,
            "indexer_available": self.indexer_available(),
            "article_tower_loaded": _registry._article_model is not None,
            "text_dim": self.dim(),
            "multilingual": self.settings.enable_multilingual,
            "vision_enabled": self.settings.enable_vision,
            "vision_ready": self.image_available(),
            "image_dim": self.settings.qdrant_image_dim,
            "device": resolve_device(self.settings.embed_device),
            "torch_present": _registry.torch is not None,
            "load_seconds": dict(_registry.load_seconds),
            "failed": sorted(_registry._failed),
            "models_root": self.settings.models_root,
        }


_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
