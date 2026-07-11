from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    n: int = 1
    top_p: Optional[float] = None
    steps: Optional[int] = None
    block_length: Optional[int] = None
    cfg_scale: Optional[float] = None
    remasking: Optional[str] = None
    detect_start_step: Optional[int] = None
    detect_interval: Optional[int] = None
    detector_history_size: Optional[int] = None
    detection_mode: Optional[str] = None
    parallel_block_decode: Optional[bool] = None
    agent_name_priority_decode: Optional[bool] = None
    agent_name_priority_window: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    n: int = 1
    top_p: Optional[float] = None
    steps: Optional[int] = None
    block_length: Optional[int] = None
    cfg_scale: Optional[float] = None
    remasking: Optional[str] = None
    detect_start_step: Optional[int] = None
    detect_interval: Optional[int] = None
    detector_history_size: Optional[int] = None
    detection_mode: Optional[str] = None
    parallel_block_decode: Optional[bool] = None
    agent_name_priority_decode: Optional[bool] = None
    agent_name_priority_window: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


def openai_error(message: str, status_code: int = 400) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": status_code,
        }
    }
