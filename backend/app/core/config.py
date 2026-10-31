"""Application settings.

All configuration is environment-driven with safe defaults so that
`docker compose up` works with no .env file at all.

The .env is looked up in the working directory first, then at the project
root, so the app finds it whether it is started from `backend/`, from the
repo root, or from inside a container.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: `__file__` is .../backend/app/core/config.py, so parents[2] is the backend
#: package root and parents[3] is the repository root that holds .env.
#:
#: This was previously `.parent.parent` off the *file*, which is .../backend/app
#: -- one level too shallow. The .env search path therefore never contained the
#: repository's actual .env, and every key silently loaded only when the
#: process happened to be started from the repository root. Running the
#: evaluation harness from eval/ got an empty provider chain and no error at
#: all, which is the worst way for this to fail.
_BACKEND = Path(__file__).resolve().parents[2]
_ROOT = _BACKEND.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_BACKEND / ".env"), str(_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SoloPrac"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ── database ─────────────────────────────────────────────────────
    postgres_user: str = Field(default="solum")
    postgres_password: str = Field(default="solum")
    postgres_db: str = Field(default="solum")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)

    # ── vector store ─────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default="solum-qdrant-local")
    qdrant_collection: str = Field(default="solum_events")
    #: Text embedding vector. MedCPT is 768-d; BGE-M3 is 1024-d. Whichever
    #: is active must match, or vectors cannot be compared.
    qdrant_text_vector: str = Field(default="medcpt")
    qdrant_text_dim: int = Field(default=768)
    qdrant_image_vector: str = Field(default="biomedclip")
    qdrant_image_dim: int = Field(default=512)

    # ── embedding models ─────────────────────────────────────────────
    #: Which encoder to use for clinical text. MedCPT is the default
    #: because it is trained on PubMed queries and is the smallest of the
    #: three, which matters on a CPU-only host.
    text_encoder: str = Field(default="medcpt")
    #: BGE-M3 understands Hindi and code-switching, which Indian records
    #: routinely contain. Costs ~2 GB RAM and is noticeably slower.
    enable_multilingual: bool = Field(default=False)
    #: Image embeddings. Off by default: 750 MB of weights and no CPU
    #: acceleration, so it is opt-in.
    enable_vision: bool = Field(default=False)
    #: Vision-language fallback for handwriting and poor scans. Needs
    #: llama-cpp-python; off by default for the same reason.
    enable_vision_llm: bool = Field(default=False)
    embed_batch_size: int = Field(default=8)
    #: "auto" picks CUDA when torch reports it and CPU otherwise. The
    #: encoders are the only genuinely parallel part of this system, and
    #: MedCPT on CPU costs seconds per query.
    embed_device: str = Field(default="auto")
    #: Load the query tower during startup instead of on first request.
    #: MedCPT takes ~15 s to load on this CPU-only host, and paying that
    #: inside a user request is indistinguishable from a hang. Loading it
    #: in the background at boot means the first real question is warm.
    preload_embeddings: bool = Field(default=True)
    #: Writable scratch space for encoder configs the app has to derive.
    #: Deliberately NOT inside the model directory: /models is mounted
    #: read-only, so anything written there fails at load time with
    #: "Read-only file system" and the encoder silently never comes up.
    embedder_cache: str = Field(default="./.cache/embedders")

    # ── redis ────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── local model paths (mounted read-only in docker) ──────────────
    models_root: str = Field(default="D:/models")
    medcpt_query_path: str = Field(default="D:/models/MedCPT-Query-Encoder")
    medcpt_article_path: str = Field(default="D:/models/MedCPT-Article-Encoder")
    bge_m3_path: str = Field(default="D:/models/bge-m3")
    biomedclip_path: str = Field(
        default="D:/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    medgemma_gguf: str = Field(default="D:/models/medgemma-gguf")
    #: Filenames inside medgemma_gguf. MedGemma ships the language model
    #: and its multimodal projector as two separate files.
    medgemma_file: str = Field(default="medgemma-4b-it_Q4_K_M.gguf")
    medgemma_mmproj: str = Field(default="mmproj-medgemma-4b-it-F16.gguf")
    medgemma_context: int = Field(default=2048)
    #: One thread per physical core. Oversubscribing on a box already
    #: running Postgres, Qdrant and Redis makes it slower, not faster.
    medgemma_threads: int = Field(default=4)
    #: Generation cap for the vision draft. This is the dominant cost on CPU:
    #: an mmap'd 4B at Q4_K_M runs around 4 tokens/s on 8 threads, so a 1024
    #: cap meant a 264-second "transcription" that was mostly padding. A lab
    #: sheet's structured reading fits well inside 400, and the cap is the
    #: difference between a usable opt-in and a hung request.
    medgemma_max_tokens: int = Field(default=400)

    embed_dim: int = Field(default=768)  # MedCPT query/article tower width

    # ── LLM ──────────────────────────────────────────────────────────
    #: Third and last provider. Kept because two working free tiers are
    #: better than one, but it is last for a measured reason: its free
    #: allowance is a hard 50 requests/day for the whole key, and that budget
    #: has been observed returning HTTP 429. Anything below it may be
    #: exhausted without the chat endpoint noticing.
    openrouter_api_key: str | None = Field(default=None)
    #: `liquid/lfm-2.5-2.6b:free` is a live free model verified to answer a
    #: clinical correction prompt correctly. The previously configured
    #: `inclusionai/ling-3.0-flash-sante:free` was retired by OpenRouter and
    #: returned 404, which is why the third link of the chain was dead.
    llm_model: str = Field(default="liquid/lfm-2.5-2.6b:free")
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1")
    #: Second provider. Groq's open models answered in 0.13 s against the
    #: primary's seconds, which is the difference between a demo that feels
    #: instant and one that does not. It sits below OpenCode because OpenCode
    #: is the configured primary and because Groq's free tier is rate-limited
    #: per day rather than per key.
    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="qwen/qwen3.8-27b")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    #: Primary provider, tried first. `space-bunny-free` is the free tier of
    #: OpenCode Zen and is the default so that the two fallback links are
    #: reached only after the primary genuinely exhausts its retries.
    opencode_api_key: str | None = Field(default=None)
    opencode_model: str = Field(default="space-bunny-free")
    opencode_base_url: str = Field(default="https://opencode.ai/zen/v1")
    #: Which provider is tried first. An unknown name falls back to the first
    #: configured provider rather than raising, because a typo in an env var
    #: should not take the chat endpoint down.
    llm_primary: str = Field(default="opencode")
    #: Set false to pin every request to the primary provider. Useful when
    #: measuring the primary in isolation, and as a kill switch if the
    #: fallback ever misbehaves in production.
    llm_fallback_enabled: bool = Field(default=True)
    #: Circuit breaker. Once a provider is exhausted, stop calling it for a
    #: cooldown instead of paying the retry cost on every single request.
    #: This is what stops an exhausted daily quota from being charged again
    #: on each turn: the failed calls are what burn a per-day budget.
    llm_breaker_enabled: bool = Field(default=True)
    llm_breaker_cooldown_s: float = Field(default=900.0)
    llm_breaker_max_cooldown_s: float = Field(default=3600.0)
    #: Hand the turn to the next provider if this one has not answered in
    #: this long. Guards the stall case, which a quota rejection is not:
    #: a 429 returns in milliseconds, so hedging on it would gain nothing,
    #: whereas a hung socket otherwise holds the doctor for the full timeout.
    #: 0 disables the hedge.
    llm_first_byte_timeout_s: float = Field(default=12.0)
    llm_timeout_s: float = Field(default=45.0)
    llm_max_retries: int = Field(default=3)
    llm_temperature: float = Field(default=0.0)
    llm_cache_ttl_s: int = Field(default=86_400)
    #: Hard gate. When false, outbound requests are not PII-checked. This
    #: exists only for local development and must never be false in
    #: a deployment that holds real patient data.
    enforce_deidentification: bool = Field(default=True)
    #: Block all outbound LLM calls (used by the eval harness when replaying
    #: frozen snapshots, so results reproduce without network access).
    offline_llm: bool = Field(default=False)

    # ── retrieval ────────────────────────────────────────────────────
    retrieval_k: int = Field(default=8)
    retrieval_alpha: float = Field(default=0.70)

    # ── capture / abstention ─────────────────────────────────────────
    ocr_confidence_threshold: float = Field(default=0.75)

    # ── identity ─────────────────────────────────────────────────────
    #: Must be overridden in any real deployment. Generated per install
    #: if unset, which invalidates existing sessions on restart - so the
    #: compose file sets it.
    token_secret: str = Field(default="dev-only-insecure-secret-change-me-0123456789")
    session_ttl_seconds: int = Field(default=60 * 60 * 24 * 7)

    # ── object storage (S3-compatible: SeaweedFS or MinIO) ───────────
    s3_endpoint: str = Field(default="http://localhost:8333")
    s3_public_endpoint: str = Field(default="http://localhost:8333")
    s3_access_key: str = Field(default="solum")
    s3_secret_key: str = Field(default="solum-secret")
    s3_bucket: str = Field(default="solum-docs")
    s3_region: str = Field(default="us-east-1")
    #: Signed download lifetime for a patient's inbox.
    download_url_ttl_seconds: int = Field(default=900)
    #: Avatars are small and re-fetched on every page view, so they get a
    #: longer lease than a clinical document.
    avatar_url_ttl_seconds: int = Field(default=86_400)

    # ── notifications ────────────────────────────────────────────────
    #: Redis Streams consumer group name.
    notification_stream: str = Field(default="solum:notifications")
    notification_maxlen: int = Field(default=10_000)
    #: Prefix for the per-recipient sets of acknowledged message ids.
    #: One set per account, suffixed with the account id. See
    #: NotificationBus.read_key for why the consumer group cannot carry this.
    notification_read_key: str = Field(default="solum:notifications:read")

    # ── CORS ─────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def llm_configured(self) -> bool:
        """Whether any provider can serve a request.

        Deliberately not keyed on one vendor. This previously required
        `openrouter_api_key` specifically, which meant a deployment holding
        only an OpenCode key reported `llm_configured: false` and the chat
        endpoint ran with no model at all - the primary was unusable purely
        because the last-resort link was missing.
        """
        return bool(self.llm_provider_chain) or (
            bool(self.openrouter_api_key) and not self.offline_llm
        )

    @property
    def llm_fallback_configured(self) -> bool:
        """Whether more than the primary is actually usable.

        Reported rather than assumed: a fallback that is configured but
        broken is worse than none, because it turns a visible primary
        failure into a silent secondary one.
        """
        return bool(self.llm_fallback_enabled and len(self.llm_provider_chain) > 1)

    @property
    def llm_provider_chain(self) -> list[tuple[str, str, str]]:
        """`(name, base_url, model)` for each provider, in attempt order.

        The order is fixed and documented: **OpenCode, then Groq, then
        OpenRouter.** `llm_primary` chooses which of these leads; the rest
        keep their relative order behind it. Insertion order into `both` is
        therefore meaningful and must not be rearranged casually - it is what
        makes a mis-set `LLM_PRIMARY` degrade to a sensible order rather than
        an arbitrary one.

        An unknown primary name falls back to OpenCode rather than raising,
        because a typo in an env var should not take the chat endpoint down.
        """
        if self.offline_llm:
            return []

        # Insertion order IS the chain order. Do not sort this dict.
        both: dict[str, tuple[str, str, str]] = {}
        if self.opencode_api_key:
            both["opencode"] = ("opencode", self.opencode_base_url, self.opencode_model)
        if self.groq_api_key:
            both["groq"] = ("groq", self.groq_base_url, self.groq_model)
        if self.openrouter_api_key:
            both["openrouter"] = ("openrouter", self.llm_base_url, self.llm_model)
        if not both:
            return []

        primary = self.llm_primary if self.llm_primary in both else "opencode"
        if primary not in both:  # no key usable under that name
            primary = next(iter(both))

        chain = [both[primary]]
        if self.llm_fallback_enabled:
            chain += [v for k, v in both.items() if k != primary]
        return chain


@lru_cache
def get_settings() -> Settings:
    return Settings()
