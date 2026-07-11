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
    detect_start_step: int = int(os.getenv("LLADA_DETECT_START_STEP", "56"))
    detect_interval: int = int(os.getenv("LLADA_DETECT_INTERVAL", "2"))
    detector_history_size: int = int(os.getenv("LLADA_DETECTOR_HISTORY_SIZE", "8"))
    detection_mode: str = os.getenv("LLADA_DETECTION_MODE", "line_unit")
    line_unit_steps: int = int(os.getenv("LLADA_LINE_UNIT_STEPS", "8"))
    line_unit_window: int = int(os.getenv("LLADA_LINE_UNIT_WINDOW", "2"))
    line_unit_max_blocks: int = int(os.getenv("LLADA_LINE_UNIT_MAX_BLOCKS", "4"))
    line_unit_force_blocks: int = int(os.getenv("LLADA_LINE_UNIT_FORCE_BLOCKS", "0"))
    line_unit_min_block_units: int = int(os.getenv("LLADA_LINE_UNIT_MIN_BLOCK_UNITS", "2"))
    line_unit_min_score: float = float(os.getenv("LLADA_LINE_UNIT_MIN_SCORE", "0.0"))
    detector_top_k: int = int(os.getenv("LLADA_DETECTOR_TOP_K", "4"))
    min_span_tokens: int = int(os.getenv("LLADA_MIN_SPAN_TOKENS", "8"))
    boundary_window: int = int(os.getenv("LLADA_BOUNDARY_WINDOW", "8"))
    structure_weight: float = float(os.getenv("LLADA_STRUCTURE_WEIGHT", "1.2"))
    min_boundary_score: float = float(os.getenv("LLADA_MIN_BOUNDARY_SCORE", "0.28"))
    stable_rounds: int = int(os.getenv("LLADA_STABLE_ROUNDS", "2"))
    prefer_structural_spans: bool = (
        os.getenv("LLADA_PREFER_STRUCTURAL_SPANS", "false").lower() == "true"
    )
    allow_partial_structures: bool = (
        os.getenv("LLADA_ALLOW_PARTIAL_STRUCTURES", "true").lower() == "true"
    )
    parallel_block_decode: bool = (
        os.getenv("LLADA_PARALLEL_BLOCK_DECODE", "true").lower() == "true"
    )
    agent_name_priority_decode: bool = (
        os.getenv("LLADA_AGENT_NAME_PRIORITY_DECODE", "true").lower() == "true"
    )
    agent_name_priority_window: int = int(os.getenv("LLADA_AGENT_NAME_PRIORITY_WINDOW", "96"))
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
