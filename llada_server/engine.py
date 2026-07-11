import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from llada_server.config import ServerConfig
from online_detect.agent_prefetch import AgentPrefetchEvent
from online_detect.generate_blockwise import LLaDAOnlineBlockwiseGenerator
from online_detect.online_block_detector import OnlineBlockDetector


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float


class LLaDAEngine:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.generator = None

    def load(self) -> None:
        dtype = self._resolve_dtype(self.config.dtype)
        self.generator = LLaDAOnlineBlockwiseGenerator(
            self.config.model_path,
            device=self.config.device,
            torch_dtype=dtype,
        )
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model

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
        detect_start_step: Optional[int] = None,
        detect_interval: Optional[int] = None,
        detector_history_size: Optional[int] = None,
        detection_mode: Optional[str] = None,
        parallel_block_decode: Optional[bool] = None,
        agent_name_priority_decode: Optional[bool] = None,
        agent_name_priority_window: Optional[int] = None,
        request_id: Optional[str] = None,
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
            detect_start_step=detect_start_step,
            detect_interval=detect_interval,
            detector_history_size=detector_history_size,
            detection_mode=detection_mode,
            parallel_block_decode=parallel_block_decode,
            agent_name_priority_decode=agent_name_priority_decode,
            agent_name_priority_window=agent_name_priority_window,
            request_id=request_id,
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
        detect_start_step: Optional[int] = None,
        detect_interval: Optional[int] = None,
        detector_history_size: Optional[int] = None,
        detection_mode: Optional[str] = None,
        parallel_block_decode: Optional[bool] = None,
        agent_name_priority_decode: Optional[bool] = None,
        agent_name_priority_window: Optional[int] = None,
        request_id: Optional[str] = None,
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
        detection_args = self._detection_args(
            detect_start_step=detect_start_step,
            detect_interval=detect_interval,
            detector_history_size=detector_history_size,
            detection_mode=detection_mode,
            parallel_block_decode=parallel_block_decode,
            agent_name_priority_decode=agent_name_priority_decode,
            agent_name_priority_window=agent_name_priority_window,
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
                f"detect_start_step={detection_args['detect_start_step']}",
                f"detection_mode={detection_args['detection_mode']}",
                flush=True,
            )
            detector = self._build_detector(detection_args)
            online_result = self.generator.generate(
                input_ids,
                logits_eos_inf=self.config.logits_eos_inf,
                confidence_eos_eot_inf=self.config.confidence_eos_eot_inf,
                detector=detector,
                prefetch_callback=self._prefetch_logger(request_id),
                save_intermediate=False,
                parallel_block_decode=detection_args["parallel_block_decode"],
                agent_name_priority_decode=detection_args["agent_name_priority_decode"],
                agent_name_priority_window=detection_args["agent_name_priority_window"],
                detect_start_step=detection_args["detect_start_step"],
                detect_interval=detection_args["detect_interval"],
                **generation_args,
            )
            generate_elapsed = time.time() - start
            print(
                "LLaDA generate finished:",
                f"elapsed={generate_elapsed:.2f}s",
                f"detection_rounds={len(online_result['detection_log'])}",
                f"prefetch_events={len(online_result['prefetch_log'])}",
                f"frozen_boundaries={online_result['frozen_boundaries']}",
                flush=True,
            )

            decode_start = time.time()
            final_response_ids = online_result["final_response_sequence"]
            text = online_result["final_response_text"]
            completion_tokens = int((final_response_ids != self.config.mask_id).sum())
            prompt_tokens = int(attention_mask.sum().item())
            decode_elapsed = time.time() - decode_start
            total_elapsed = time.time() - start
            print(
                "LLaDA decode finished:",
                f"decode_elapsed={decode_elapsed:.2f}s",
                f"total_elapsed={total_elapsed:.2f}s",
                f"completion_tokens={completion_tokens}",
                f"output_chars={len(text)}",
                flush=True,
            )
            return GenerationResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                elapsed_seconds=total_elapsed,
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

    def _detection_args(
        self,
        *,
        detect_start_step: Optional[int],
        detect_interval: Optional[int],
        detector_history_size: Optional[int],
        detection_mode: Optional[str],
        parallel_block_decode: Optional[bool],
        agent_name_priority_decode: Optional[bool],
        agent_name_priority_window: Optional[int],
    ) -> dict:
        detect_start_step = detect_start_step or self.config.detect_start_step
        detect_interval = detect_interval or self.config.detect_interval
        detector_history_size = detector_history_size or self.config.detector_history_size
        detection_mode = detection_mode or self.config.detection_mode
        parallel_block_decode = (
            self.config.parallel_block_decode
            if parallel_block_decode is None
            else bool(parallel_block_decode)
        )
        agent_name_priority_decode = (
            self.config.agent_name_priority_decode
            if agent_name_priority_decode is None
            else bool(agent_name_priority_decode)
        )
        agent_name_priority_window = (
            agent_name_priority_window or self.config.agent_name_priority_window
        )

        if detect_start_step <= 0:
            raise ValueError("detect_start_step must be greater than 0.")
        if detect_interval <= 0:
            raise ValueError("detect_interval must be greater than 0.")
        if detector_history_size <= 0:
            raise ValueError("detector_history_size must be greater than 0.")
        if detection_mode not in {"line_unit", "token"}:
            raise ValueError("detection_mode must be 'line_unit' or 'token'.")
        if agent_name_priority_window <= 0:
            raise ValueError("agent_name_priority_window must be greater than 0.")

        return {
            "detect_start_step": int(detect_start_step),
            "detect_interval": int(detect_interval),
            "detector_history_size": int(detector_history_size),
            "detection_mode": detection_mode,
            "parallel_block_decode": parallel_block_decode,
            "agent_name_priority_decode": agent_name_priority_decode,
            "agent_name_priority_window": int(agent_name_priority_window),
        }

    def _build_detector(self, detection_args: dict) -> OnlineBlockDetector:
        history_size = (
            self.config.line_unit_steps
            if detection_args["detection_mode"] == "line_unit"
            else detection_args["detector_history_size"]
        )
        return OnlineBlockDetector(
            history_size=history_size,
            detection_mode=detection_args["detection_mode"],
            line_unit_window=self.config.line_unit_window,
            line_unit_max_blocks=self.config.line_unit_max_blocks,
            line_unit_force_blocks=self.config.line_unit_force_blocks,
            line_unit_min_block_units=self.config.line_unit_min_block_units,
            line_unit_min_score=self.config.line_unit_min_score,
            detect_top_k=self.config.detector_top_k,
            min_span_tokens=self.config.min_span_tokens,
            boundary_window=self.config.boundary_window,
            structure_weight=self.config.structure_weight,
            min_boundary_score=self.config.min_boundary_score,
            stable_rounds=self.config.stable_rounds,
            prefer_structural_spans=self.config.prefer_structural_spans,
            allow_partial_structures=self.config.allow_partial_structures,
        )

    def _prefetch_logger(self, request_id: Optional[str]):
        def log(event: AgentPrefetchEvent) -> None:
            print(
                "[Agent Prefetch]",
                f"request_id={request_id or '-'}",
                f"span={event.span_id}",
                f"agent={event.agent_name}",
                f"confidence={event.confidence:.4f}",
                f"preview={event.text_preview!r}",
                flush=True,
            )

        return log

    def _ensure_loaded(self) -> None:
        if self.generator is None or self.model is None or self.tokenizer is None:
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
