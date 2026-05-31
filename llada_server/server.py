import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from llada_server.config import get_config
from llada_server.engine import LLaDAEngine
from llada_server.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    openai_error,
)

config = get_config()
engine = LLaDAEngine(config)
generation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield


app = FastAPI(title="LLaDA OpenAI-Compatible Server", lifespan=lifespan)


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content=openai_error(str(exc), 400))


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.model_name}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(
        data=[
            ModelCard(
                id=config.model_name,
                created=0,
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    request_start = time.time()
    print(
        "HTTP chat request received:",
        f"id={request_id}",
        f"messages={len(request.messages)}",
        flush=True,
    )
    if request.stream:
        return JSONResponse(
            status_code=400,
            content=openai_error("stream=true is not supported by this server.", 400),
        )
    if request.n != 1:
        return JSONResponse(
            status_code=400,
            content=openai_error("Only n=1 is supported by this server.", 400),
        )

    messages = [message.model_dump() for message in request.messages]
    print("HTTP chat waiting for generation lock:", f"id={request_id}", flush=True)
    async with generation_lock:
        print("HTTP chat generation lock acquired:", f"id={request_id}", flush=True)
        result = await asyncio.to_thread(
            engine.generate_chat,
            messages,
            gen_length=request.max_tokens,
            steps=request.steps,
            block_length=request.block_length,
            temperature=request.temperature,
            cfg_scale=request.cfg_scale,
            remasking=request.remasking,
        )

    created = int(time.time())
    elapsed = time.time() - request_start
    print(
        "HTTP chat response ready:",
        f"id={request_id}",
        f"elapsed={elapsed:.2f}s",
        f"engine_elapsed={result.elapsed_seconds:.2f}s",
        flush=True,
    )
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model or config.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    request_id = f"cmpl-{uuid.uuid4().hex}"
    request_start = time.time()
    print(
        "HTTP completion request received:",
        f"id={request_id}",
        flush=True,
    )
    if request.stream:
        return JSONResponse(
            status_code=400,
            content=openai_error("stream=true is not supported by this server.", 400),
        )
    if request.n != 1:
        return JSONResponse(
            status_code=400,
            content=openai_error("Only n=1 is supported by this server.", 400),
        )

    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    choices = []
    prompt_tokens = 0
    completion_tokens = 0

    print("HTTP completion waiting for generation lock:", f"id={request_id}", flush=True)
    async with generation_lock:
        print("HTTP completion generation lock acquired:", f"id={request_id}", flush=True)
        for index, prompt in enumerate(prompts):
            result = await asyncio.to_thread(
                engine.generate_text,
                prompt,
                gen_length=request.max_tokens,
                steps=request.steps,
                block_length=request.block_length,
                temperature=request.temperature,
                cfg_scale=request.cfg_scale,
                remasking=request.remasking,
            )
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            choices.append(
                {
                    "text": result.text,
                    "index": index,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            )

    elapsed = time.time() - request_start
    print(
        "HTTP completion response ready:",
        f"id={request_id}",
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model or config.model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "llada_server.server:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
