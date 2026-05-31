import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from generate import generate
from llada_server.config import ServerConfig


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLaDAEngine:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        dtype = self._resolve_dtype(self.config.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.config.device).eval()

        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id == self.config.mask_id:
            raise ValueError(
                "tokenizer.pad_token_id equals mask_id; generate.py requires a different pad token."
            )

    def generate_chat(
        self,
        messages: Sequence[dict],
        *,
        gen_length: Optional[int] = None,
        steps: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        remasking: Optional[str] = None,
    ) -> GenerationResult:
        self._ensure_loaded()
        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=False,
        )
        if self.config.debug_requests:
            self._log_chat_request(messages, prompt)
        return self.generate_text(
            prompt,
            gen_length=gen_length,
            steps=steps,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            add_special_tokens=False,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        gen_length: Optional[int] = None,
        steps: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        remasking: Optional[str] = None,
        add_special_tokens: bool = False,
    ) -> GenerationResult:
        self._ensure_loaded()
        if self.config.debug_requests:
            self._log_prompt("completion", prompt)
        generation_args = self._generation_args(
            gen_length=gen_length,
            steps=steps,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
        )
        encoded = self.tokenizer(
            [prompt],
            add_special_tokens=add_special_tokens,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)

        try:
            start = time.time()
            print(
                "LLaDA request:",
                f"prompt_tokens={int(attention_mask.sum().item())}",
                f"gen_length={generation_args['gen_length']}",
                f"total_length={input_ids.shape[1] + generation_args['gen_length']}",
                f"steps={generation_args['steps']}",
                f"block_length={generation_args['block_length']}",
                flush=True,
            )
            out = generate(
                self.model,
                input_ids,
                attention_mask=None,
                mask_id=self.config.mask_id,
                logits_eos_inf=self.config.logits_eos_inf,
                confidence_eos_eot_inf=self.config.confidence_eos_eot_inf,
                **generation_args,
            )
            elapsed = time.time() - start
            _ = elapsed

            completion_ids = out[:, input_ids.shape[1] :]
            text = self.tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )[0]
            completion_tokens = int((completion_ids != self.config.mask_id).sum().item())
            prompt_tokens = int(attention_mask.sum().item())
            return GenerationResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        finally:
            del input_ids
            del attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generation_args(
        self,
        *,
        gen_length: Optional[int],
        steps: Optional[int],
        block_length: Optional[int],
        temperature: Optional[float],
        cfg_scale: Optional[float],
        remasking: Optional[str],
    ) -> dict:
        gen_length = gen_length or self.config.default_gen_length
        steps = steps or self.config.default_steps
        block_length = block_length or self.config.default_block_length
        temperature = (
            self.config.default_temperature if temperature is None else temperature
        )
        cfg_scale = self.config.default_cfg_scale if cfg_scale is None else cfg_scale
        remasking = remasking or self.config.default_remasking

        if gen_length <= 0:
            raise ValueError("gen_length/max_tokens must be greater than 0.")
        if steps <= 0:
            raise ValueError("steps must be greater than 0.")
        if block_length <= 0:
            raise ValueError("block_length must be greater than 0.")
        if block_length > gen_length:
            block_length = gen_length
        if gen_length % block_length != 0:
            raise ValueError("gen_length/max_tokens must be divisible by block_length.")
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            raise ValueError("steps must be divisible by gen_length / block_length.")
        if remasking not in {"low_confidence", "random"}:
            raise ValueError("remasking must be 'low_confidence' or 'random'.")

        return {
            "steps": steps,
            "gen_length": gen_length,
            "block_length": block_length,
            "temperature": temperature,
            "cfg_scale": cfg_scale,
            "remasking": remasking,
        }

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLaDA engine is not loaded.")

    def _log_chat_request(self, messages: Sequence[dict], prompt: str) -> None:
        print("LLaDA chat request:", flush=True)
        print(f"  messages_count={len(messages)}", flush=True)
        for index, message in enumerate(messages):
            content = message.get("content") or ""
            print(
                f"  message[{index}] role={message.get('role')} "
                f"chars={len(content)} tokens={len(self.tokenizer.encode(content, add_special_tokens=False))}",
                flush=True,
            )
            if self.config.debug_full_request:
                print(f"  message[{index}] content={content!r}", flush=True)
            else:
                print(
                    f"  message[{index}] preview={self._preview(content)!r}",
                    flush=True,
                )
        self._log_prompt("chat_template", prompt)

    def _log_prompt(self, label: str, prompt: str) -> None:
        token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        print(
            f"LLaDA prompt[{label}]: chars={len(prompt)} tokens={token_count}",
            flush=True,
        )
        if self.config.debug_full_request:
            print(f"LLaDA prompt[{label}] full={prompt!r}", flush=True)
        else:
            print(
                f"LLaDA prompt[{label}] preview={self._preview(prompt)!r}",
                flush=True,
            )

    def _preview(self, text: str) -> str:
        preview_chars = self.config.debug_preview_chars
        if len(text) <= preview_chars * 2:
            return text
        return f"{text[:preview_chars]} ... <{len(text) - preview_chars * 2} chars omitted> ... {text[-preview_chars:]}"

    @staticmethod
    def _resolve_dtype(dtype: str):
        dtype = dtype.lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32", "full"}:
            return torch.float32
        raise ValueError("LLADA_DTYPE must be bfloat16, float16, or float32.")
