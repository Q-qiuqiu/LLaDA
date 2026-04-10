import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from openpyxl import Workbook
from transformers import AutoModel, AutoTokenizer


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise to logits."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Precompute how many tokens should be transferred at each denoising step
    for each sample in a block (same as original generate.py logic).
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def count_remaining_mask_ratio(x: torch.Tensor, prompt_len: int, mask_id: int) -> float:
    """Remaining mask ratio in generation region."""
    gen_region = x[:, prompt_len:]
    remain = (gen_region == mask_id).sum().item()
    total = gen_region.numel()
    return remain / max(total, 1)


class LLaDAValueRecorder:
    """Record value statistics from the first N denoising steps."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()

        if hasattr(self.model, "transformer"):
            self.base_model = self.model
        elif hasattr(self.model, "model"):
            self.base_model = self.model.model
        else:
            raise AttributeError("Cannot find transformer in model")

        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "blocks"):
            self.blocks = self.base_model.transformer.blocks
        elif hasattr(self.base_model, "blocks"):
            self.blocks = self.base_model.blocks
        else:
            raise AttributeError("Cannot find transformer blocks")

        self.num_layers = len(self.blocks)
        self.mask_id = 126336

        config = self.base_model.config
        self.num_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "effective_n_kv_heads", self.num_heads)

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

    def build_attention_bias(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        mask = attention_mask[:, None, None, :].to(dtype=torch.float32, device=device)
        mask = (1.0 - mask) * torch.finfo(torch.float32).min
        mask = mask.expand(attention_mask.shape[0], 1, seq_len, seq_len)
        return mask.to(dtype=dtype)

    def extract_qkv_and_apply_rope(
        self, block, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        x_normed = block.attn_norm(hidden_states)

        q = block.q_proj(x_normed)
        k = block.k_proj(x_normed)
        v = block.v_proj(x_normed)

        batch, seq_len, width = q.size()

        if getattr(block, "q_norm", None) is not None and getattr(block, "k_norm", None) is not None:
            q = block.q_norm(q).to(dtype=dtype)
            k = block.k_norm(k).to(dtype=dtype)

        q = q.view(batch, seq_len, self.num_heads, width // self.num_heads).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, width // self.num_heads).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, width // self.num_heads).transpose(1, 2)

        if hasattr(block, "rotary_emb"):
            q, k = block.rotary_emb(q, k)

        return q, k, v

    def compute_attention_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            seq_len = q.shape[-2]
            attn_bias = self.build_attention_bias(
                attention_mask=attention_mask,
                seq_len=seq_len,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            attn_weights = attn_weights + attn_bias

        return torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    def compute_step_avg_attention(
        self,
        all_hidden_states,
        layers_to_extract: List[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_list = []
        for layer_idx in layers_to_extract:
            hidden = all_hidden_states[layer_idx]
            block = self.blocks[layer_idx]

            q, k, _ = self.extract_qkv_and_apply_rope(block, hidden)
            if self.num_kv_heads != self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = self._repeat_kv(k, n_rep)

            attn = self.compute_attention_weights(q, k, attention_mask=attention_mask)
            attn_list.append(attn.mean(dim=1).squeeze(0))

        return torch.stack(attn_list, dim=0).mean(dim=0)

    @staticmethod
    def _decode_token_str(tokenizer, token_id: int) -> str:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        return text.replace("\x00", "")

    def _decode_with_mask(self, token_ids: List[int]) -> str:
        parts: List[str] = []
        buf: List[int] = []
        for tid in token_ids:
            if int(tid) == self.mask_id:
                if buf:
                    txt = self.tokenizer.decode(buf, skip_special_tokens=False).replace("\x00", "")
                    parts.append(txt)
                    buf = []
                parts.append("<MASK>")
            else:
                buf.append(int(tid))
        if buf:
            txt = self.tokenizer.decode(buf, skip_special_tokens=False).replace("\x00", "")
            parts.append(txt)
        return "".join(parts)

    @torch.no_grad()
    def generate_and_record_values(
        self,
        prompt: torch.Tensor,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        capture_steps: int = 5,
        layers_to_extract: Optional[List[int]] = None,
        save_intermediate: bool = False,
        output_file: str = "/home/yzx/LLaDA/yzx_test/value_outputs/denoise_value_log.txt",
    ) -> Dict[str, object]:
        prompt_len = int(prompt.shape[1])
        x = torch.full(
            (1, prompt_len + gen_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_len] = prompt.clone()
        prompt_index = x != self.mask_id
        attention_mask = torch.ones_like(x, dtype=torch.long, device=self.device)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        if layers_to_extract is None:
            layers_to_extract = list(range(max(0, self.num_layers - 2), self.num_layers))

        if save_intermediate:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(
                    f"steps={steps}, gen_length={gen_length}, block_length={block_length}, "
                    f"capture_steps={capture_steps}\n"
                )
                f.write("=" * 60 + "\n")

        captured_records: List[Dict[str, object]] = []
        global_step = 0

        for num_block in range(num_blocks):
            block_start = prompt_len + num_block * block_length
            block_end = prompt_len + (num_block + 1) * block_length
            block_mask_index = x[:, block_start:block_end] == self.mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == self.mask_id
                if not mask_index[:, block_start:block_end].any():
                    global_step += 1
                    continue

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)

                    outputs = self.base_model(x_, attention_mask=attention_mask_, output_hidden_states=True)
                    logits = outputs.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    all_hidden_states = tuple(h[:1] for h in outputs.hidden_states)
                else:
                    outputs = self.base_model(x, attention_mask=attention_mask, output_hidden_states=True)
                    logits = outputs.logits
                    all_hidden_states = outputs.hidden_states

                avg_attn = self.compute_step_avg_attention(
                    all_hidden_states=all_hidden_states,
                    layers_to_extract=layers_to_extract,
                    attention_mask=attention_mask,
                )

                probs = F.softmax(logits, dim=-1)
                top_confidence, top_token_ids = torch.max(probs, dim=-1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(x0.shape[0]):
                    k_this_step = int(num_transfer_tokens[j, i].item())
                    if k_this_step <= 0:
                        continue
                    _, select_index = torch.topk(confidence[j], k=k_this_step)
                    transfer_index[j, select_index] = True

                selected_positions = torch.where(transfer_index[0])[0]
                remain_ratio = count_remaining_mask_ratio(x, prompt_len, self.mask_id)

                capture_this_step = len(captured_records) < capture_steps
                if capture_this_step:
                    record = {
                        "global_step": int(global_step),
                        "block_index": int(num_block),
                        "step_in_block": int(i),
                        "sequence_before_update": x[0].detach().cpu().numpy().astype(np.int64),
                        "avg_attention": avg_attn.detach().float().cpu().numpy(),
                        "top_confidence": top_confidence[0].detach().float().cpu().numpy(),
                        "top_token_ids": top_token_ids[0].detach().cpu().numpy().astype(np.int64),
                        "selected_positions": selected_positions.detach().cpu().numpy().astype(np.int64),
                        "selected_token_ids": x0[0, selected_positions].detach().cpu().numpy().astype(np.int64),
                    }

                x[transfer_index] = x0[transfer_index]
                if capture_this_step:
                    record["sequence_after_update"] = x[0].detach().cpu().numpy().astype(np.int64)
                    captured_records.append(record)

                if save_intermediate:
                    response_ids = x[0, prompt_len:].detach().tolist()
                    intermediate_text = self._decode_with_mask(response_ids)
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"GlobalStep {global_step + 1}/{steps} "
                            f"Block {num_block + 1}/{num_blocks} "
                            f"StepInBlock {i + 1}/{steps_per_block} "
                            f"selected={int(selected_positions.numel())} "
                            f"remain_ratio={remain_ratio:.4f}\n"
                        )
                        f.write(intermediate_text + "\n")
                        f.write("-" * 60 + "\n")

                global_step += 1

        if len(captured_records) == 0:
            raise RuntimeError("No denoising step was captured. Please check generation settings.")

        attention_scores = np.stack([r["avg_attention"] for r in captured_records], axis=0)
        top_confidences = np.stack([r["top_confidence"] for r in captured_records], axis=0)
        top_token_ids = np.stack([r["top_token_ids"] for r in captured_records], axis=0)
        seq_before = np.stack([r["sequence_before_update"] for r in captured_records], axis=0)
        seq_after = np.stack([r["sequence_after_update"] for r in captured_records], axis=0)

        # Keep only response-region values.
        attention_scores = attention_scores[:, prompt_len:, prompt_len:]
        top_confidences = top_confidences[:, prompt_len:]
        top_token_ids = top_token_ids[:, prompt_len:]
        seq_before = seq_before[:, prompt_len:]
        seq_after = seq_after[:, prompt_len:]

        if len(captured_records) > 1:
            attention_deltas = attention_scores[1:] - attention_scores[:-1]
            confidence_deltas = top_confidences[1:] - top_confidences[:-1]
        else:
            attention_deltas = np.empty((0, *attention_scores.shape[1:]), dtype=np.float32)
            confidence_deltas = np.empty((0, top_confidences.shape[1]), dtype=np.float32)

        decoded_top_tokens = [
            [self._decode_token_str(self.tokenizer, tid) for tid in step_ids]
            for step_ids in top_token_ids
        ]

        meta = {
            "num_captured_steps": len(captured_records),
            "capture_steps_target": capture_steps,
            "prompt_length": prompt_len,
            "layers_to_extract": layers_to_extract,
            "global_steps": [r["global_step"] for r in captured_records],
            "block_indices": [r["block_index"] for r in captured_records],
            "steps_in_block": [r["step_in_block"] for r in captured_records],
            "selected_positions": [r["selected_positions"].tolist() for r in captured_records],
            "selected_token_ids": [r["selected_token_ids"].tolist() for r in captured_records],
            "selected_tokens": [
                [self._decode_token_str(self.tokenizer, t) for t in r["selected_token_ids"]]
                for r in captured_records
            ],
            "top_token_text_per_step": decoded_top_tokens,
        }

        return {
            "final_response_sequence": x[0, prompt_len:].detach().cpu().numpy().astype(np.int64),
            "attention_scores": attention_scores.astype(np.float32),
            "top_confidences": top_confidences.astype(np.float32),
            "top_token_ids": top_token_ids.astype(np.int64),
            "attention_deltas": attention_deltas.astype(np.float32),
            "confidence_deltas": confidence_deltas.astype(np.float32),
            "sequence_before_update": seq_before.astype(np.int64),
            "sequence_after_update": seq_after.astype(np.int64),
            "meta": meta,
        }

    @staticmethod
    def save_values(record: Dict[str, object], output_prefix: str) -> str:
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        xlsx_path = f"{output_prefix}.xlsx"

        attention_scores = record["attention_scores"]  # [S, L, L]
        top_confidences = record["top_confidences"]  # [S, L]
        top_token_ids = record["top_token_ids"]  # [S, L]
        attention_deltas = record["attention_deltas"]  # [S-1, L, L]
        confidence_deltas = record["confidence_deltas"]  # [S-1, L]
        meta = record["meta"]
        top_token_text_per_step = meta.get("top_token_text_per_step", [])
        prompt_len = int(meta.get("prompt_length", 0))

        wb = Workbook()
        ws = wb.active
        ws.title = "denoise_values"

        def token_cell(step_idx: int, pos_idx: int) -> str:
            token_id = int(top_token_ids[step_idx, pos_idx])
            token_text = ""
            if step_idx < len(top_token_text_per_step) and pos_idx < len(top_token_text_per_step[step_idx]):
                token_text = str(top_token_text_per_step[step_idx][pos_idx])
            token_text = token_text.replace("\n", "\\n")
            return f"{token_text} (id={token_id})"

        num_steps = int(attention_scores.shape[0])
        seq_len = int(attention_scores.shape[1])

        for step_idx in range(num_steps):
            ws.append([f"Step {step_idx + 1}", "", "", ""])
            ws.append(["sequence_token_index", "token", "attention_scores", "confidence"])
            for pos_idx in range(seq_len):
                seq_token_index = prompt_len + pos_idx + 1
                ws.append(
                    [
                        seq_token_index,
                        token_cell(step_idx, pos_idx),
                        json.dumps(attention_scores[step_idx, pos_idx, :].tolist(), ensure_ascii=False),
                        float(top_confidences[step_idx, pos_idx]),
                    ]
                )
            ws.append(["", "", "", ""])

        num_deltas = int(attention_deltas.shape[0])
        for pair_idx in range(num_deltas):
            left = pair_idx + 1
            right = pair_idx + 2
            ws.append([f"Step {left}-{right} Delta", "", "", ""])
            ws.append(["sequence_token_index", "token", "attention_delta", "confidence_delta"])
            # 差值 section 的 token 使用右侧步骤（例如 1-2 用第 2 步 token）
            token_step_idx = pair_idx + 1
            for pos_idx in range(seq_len):
                seq_token_index = prompt_len + pos_idx + 1
                ws.append(
                    [
                        seq_token_index,
                        token_cell(token_step_idx, pos_idx),
                        json.dumps(attention_deltas[pair_idx, pos_idx, :].tolist(), ensure_ascii=False),
                        float(confidence_deltas[pair_idx, pos_idx]),
                    ]
                )
            ws.append(["", "", "", ""])

        wb.save(xlsx_path)
        return xlsx_path


def main():
    prompt =""" 
    You are a coordinator agent responsible for decomposing a complex task into sub-tasks and assigning them to specialized agents.

    You have access to the following three agents:

    1. ResearchAgent

    * Description: Responsible for retrieving relevant background information, facts, and references from knowledge sources.
    * Input: query (string)
    * Output: concise factual summary

    2. PlanningAgent

    * Description: Responsible for creating structured plans, step-by-step strategies, or workflows.
    * Input: objective (string), context (string)
    * Output: ordered list of steps

    3. CodingAgent

    * Description: Responsible for writing code or performing calculations based on given specifications.
    * Input: specification (string)
    * Output: code snippet or computed result

    Task:
    "I want to build a simple web app that recommends books based on user preferences. It should include a recommendation algorithm and a basic UI."

    Your job:
    1. Analyze the task
    2. Decompose it into sub-tasks
    3. Assign each sub-task to the most appropriate agent

    Output format (STRICT):
    Produce a list of agent calls. Each call must be independent and follow this structure:
    {
    name: <AgentName>
    arguments: <key>: <value>
    ...
    }
    ---

    Requirements:

    * You MUST produce exactly three agent calls (one per agent)
    * Each agent call should correspond to a distinct sub-task
    * Keep arguments concise but sufficient for execution
    * Ensure that the three agent calls are logically independent (can be executed in parallel)
    ---
    """
    parser = argparse.ArgumentParser(description="Record first denoising-step values from LLaDA generation.")
    parser.add_argument("--model-path", type=str, default="/data/labshare/Param/llada/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--capture-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--save-intermediate", action="store_true")
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/yzx/LLaDA/yzx_test/denoise_value_log.txt",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="/home/yzx/LLaDA/yzx_test/value_outputs/denoise_value",
    )
    parser.set_defaults(save_intermediate=True)
    args = parser.parse_args()

    recorder = LLaDAValueRecorder(model_path=args.model_path, device=args.device)

    input_text = f"Question: {args.prompt}\nAnswer:"
    inputs = recorder.tokenizer(input_text, return_tensors="pt").to(args.device)

    record = recorder.generate_and_record_values(
        prompt=inputs.input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        capture_steps=args.capture_steps,
        save_intermediate=args.save_intermediate,
        output_file=args.output_file,
    )

    xlsx_path = recorder.save_values(record, args.output_prefix)
    print(f"Saved values to xlsx: {xlsx_path}")
    if args.save_intermediate:
        print(f"Saved intermediate log to: {args.output_file}")
    print(
        "Final generated response: "
        f"{recorder.tokenizer.decode(record['final_response_sequence'], skip_special_tokens=True)}"
    )


if __name__ == "__main__":
    main()
