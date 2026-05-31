import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    model_path: str = os.getenv("LLADA_MODEL_PATH", "/home/yzx/models_weight/LLaDA/")
    model_name: str = os.getenv("LLADA_MODEL_NAME", "llada")
    device: str = os.getenv("LLADA_DEVICE", "cuda")
    dtype: str = os.getenv("LLADA_DTYPE", "bfloat16")
    host: str = os.getenv("LLADA_HOST", "0.0.0.0")
    port: int = int(os.getenv("LLADA_PORT", "8000"))
    default_steps: int = int(os.getenv("LLADA_STEPS", "128"))
    default_gen_length: int = int(os.getenv("LLADA_GEN_LENGTH", "128"))
    default_block_length: int = int(os.getenv("LLADA_BLOCK_LENGTH", "32"))
    default_temperature: float = float(os.getenv("LLADA_TEMPERATURE", "0"))
    default_cfg_scale: float = float(os.getenv("LLADA_CFG_SCALE", "0"))
    default_remasking: str = os.getenv("LLADA_REMASKING", "low_confidence")
    mask_id: int = int(os.getenv("LLADA_MASK_ID", "126336"))
    logits_eos_inf: bool = os.getenv("LLADA_LOGITS_EOS_INF", "false").lower() == "true"
    confidence_eos_eot_inf: bool = (
        os.getenv("LLADA_CONFIDENCE_EOS_EOT_INF", "false").lower() == "true"
    )
    debug_requests: bool = os.getenv("LLADA_DEBUG_REQUESTS", "true").lower() == "true"
    debug_full_request: bool = (
        os.getenv("LLADA_DEBUG_FULL_REQUEST", "false").lower() == "true"
    )
    debug_preview_chars: int = int(os.getenv("LLADA_DEBUG_PREVIEW_CHARS", "500"))


def get_config() -> ServerConfig:
    return ServerConfig()
