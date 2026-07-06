import time
import uuid
import threading
from typing import List, Optional, Dict, Any, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 配置区
# =========================
MODEL_PATH = "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct"
SERVED_MODEL_NAME = "llama3-8b-instruct"

HOST = "0.0.0.0"
PORT = 7002

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9

# 基础 Transformers generate 不适合多个请求同时抢 GPU。
# 这里用锁保证一次只跑一个请求，稳定优先。
GENERATE_LOCK = threading.Lock()


# =========================
# OpenAI-compatible 请求格式
# =========================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = SERVED_MODEL_NAME
    messages: List[ChatMessage]

    max_tokens: Optional[int] = Field(default=DEFAULT_MAX_TOKENS)
    temperature: Optional[float] = Field(default=DEFAULT_TEMPERATURE)
    top_p: Optional[float] = Field(default=DEFAULT_TOP_P)

    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


# =========================
# 加载模型，只执行一次
# =========================
print(f"Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    dtype = torch.float32

print(f"Loading model from: {MODEL_PATH}")
print(f"Using dtype: {dtype}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto",
    local_files_only=True,
)

model.eval()


def get_input_device():
    """
    device_map='auto' 时，模型可能被切到多张卡。
    输入一般放到 embedding 所在设备即可。
    """
    if hasattr(model, "hf_device_map"):
        for key in ["model.embed_tokens", "model.tok_embeddings", "transformer.wte"]:
            if key in model.hf_device_map:
                dev = model.hf_device_map[key]
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                return torch.device(dev)

        for _, dev in model.hf_device_map.items():
            if dev not in ["cpu", "disk"]:
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                return torch.device(dev)

    return next(model.parameters()).device


INPUT_DEVICE = get_input_device()
print(f"Input device: {INPUT_DEVICE}")


def get_eos_token_ids():
    """
    Llama 3 Instruct 常用 <|eot_id|> 作为一轮对话结束标记。
    只用 tokenizer.eos_token_id 有时不会及时停。
    """
    eos_ids = []

    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    return eos_ids


EOS_TOKEN_IDS = get_eos_token_ids()
print(f"EOS token ids: {EOS_TOKEN_IDS}")


# =========================
# FastAPI
# =========================
app = FastAPI()


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="This minimal server does not support stream=True yet.",
        )

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages = [m.model_dump() for m in req.messages]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(INPUT_DEVICE)

        prompt_tokens = input_ids.shape[-1]

        max_new_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
        temperature = DEFAULT_TEMPERATURE if req.temperature is None else req.temperature
        top_p = DEFAULT_TOP_P if req.top_p is None else req.top_p

        do_sample = temperature > 0

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "eos_token_id": EOS_TOKEN_IDS,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        with GENERATE_LOCK:
            with torch.no_grad():
                outputs = model.generate(**generation_kwargs)

        generated_ids = outputs[0][prompt_tokens:]
        completion_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # 简单支持 stop 字符串
        if req.stop is not None:
            stops = [req.stop] if isinstance(req.stop, str) else req.stop
            for s in stops:
                idx = completion_text.find(s)
                if idx != -1:
                    completion_text = completion_text[:idx]
                    break

        completion_text = completion_text.strip()

        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)