"""
Server configuration. Reads from env vars, with sensible defaults for local dev.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Paths ---
    root_dir: Path = Path(__file__).resolve().parent.parent
    weights_dir: Path = root_dir / "weights"
    stage1_weights: Path = weights_dir / "artifact_xception.pt"
    stage2_weights: Path = weights_dir / "au_mlp.pt"

    # --- Inference ---
    device: str = "cuda"  # falls back to cpu in code if unavailable
    image_size: int = 299  # Xception input

    # uncertainty band: stage1 P(fake) in [low, high] -> route to stage2
    stage1_uncertainty_low: float = 0.40
    stage1_uncertainty_high: float = 0.60

    # final decision threshold
    decision_threshold: float = 0.50

    # weight for combining stage1+stage2 (weighted avg)
    stage1_weight: float = 0.4
    stage2_weight: float = 0.6

    # --- Networking ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]  # tighten in prod

    # --- Cache ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 60 * 60 * 24  # 1 day

    # --- Behavior ---
    # If True, run with random predictions when weights aren't found.
    # Useful for wiring the extension before models are trained.
    dummy_mode_if_missing_weights: bool = True

    # download timeout when fetching image_url
    image_fetch_timeout: float = 5.0
    max_image_bytes: int = 10 * 1024 * 1024  # 10MB


settings = Settings()
