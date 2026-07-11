# LLaDA OpenAI-Compatible Server

This server loads LLaDA once at startup and reuses the model for multiple HTTP requests.

## Install

Install the original LLaDA dependencies, then add:

```bash
pip install fastapi uvicorn
```

## Start

Run from the repository root:

```bash
export LLADA_MODEL_PATH=/home/yzx/models_weight/LLaDA/
python -m llada_server.server
```

Useful environment variables:

```bash
export LLADA_HOST=0.0.0.0
export LLADA_PORT=8000
export LLADA_MODEL_NAME=llada
export LLADA_DEVICE=cuda
export LLADA_DTYPE=bfloat16
export LLADA_STEPS=128
export LLADA_GEN_LENGTH=128
export LLADA_BLOCK_LENGTH=32
export LLADA_DETECT_START_STEP=56
export LLADA_DETECTION_MODE=line_unit
export LLADA_PARALLEL_BLOCK_DECODE=true
```

The server uses `online_detect.generate_blockwise` by default. Agent/tool
prefetch events are printed to stdout only; no external prefetch action is
performed by `llada_server`.

## Chat Completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llada",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    "max_tokens": 128,
    "steps": 128,
    "block_length": 32,
    "temperature": 0
  }'
```

## Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="llada",
    messages=[{"role": "user", "content": "写一个快速排序"}],
    max_tokens=128,
    extra_body={
        "steps": 128,
        "block_length": 32,
        "detect_start_step": 56,
        "detection_mode": "line_unit",
    },
)
print(response.choices[0].message.content)
```

Streaming, tool calls, logprobs, and multiple choices with `n > 1` are not implemented.
