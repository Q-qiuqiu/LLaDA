import argparse
import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from .agent_prefetch import AgentPrefetchEvent, AgentPrefetcher, draft_text, extract_agent_name
    from .online_block_detector import (
        DetectionResult,
        OnlineBlockDetector,
        StepSnapshot,
        _json_fragment_penalty,
        _normalize_scores,
        _structure_bonus,
        build_line_units,
    )
except ImportError:
    from agent_prefetch import AgentPrefetchEvent, AgentPrefetcher, draft_text, extract_agent_name
    from online_block_detector import (
        DetectionResult,
        OnlineBlockDetector,
        StepSnapshot,
        _json_fragment_penalty,
        _normalize_scores,
        _structure_bonus,
        build_line_units,
    )


CONFIG = {
    "prompt": """ 
    [
      {
        "role": "system",
        "content": "You are a helpful assistant to make travel plans for Bob.\n\nEXTERNAL RESOURCES:\n1. A database containing information about train tickets, attractions, and city transportation.\n2. A python notebook to execute python code for numerical operations and planning. \n\nTASK DESCRIPTION\nYou need to make a travel plan based on the given requirements, taking into account transportation between cities and daily schedules.\nThe final travel plan may include or be part of the following:\n1.go_to_place(origin:str,destination:str,departure_time,arrival_time): go to destination from origin. The origin and destination should be the name of a hotel or a spot instead of a city.\n2.visit(place:str,begin_time,end_time): visit somewhere from begin_time to end_time. The time should be expressed as \"%Y-%m-%d %H:%M\", e.g. 2023-07-02 16:00. You have to go somewhere before you can visit it.\n3.go_to_city(origin_city:str,destination_city:str,departure_time,arrival_time,ticket_number): go to destination city from origin city, using the ticket with the ticket_number(you can know the ticket number from the database).\n4.stay_in(city:str,begin_time,end_time): stay in somewhere from begin_time to end_time. The time should be expressed as \"%Y-%m-%d %H:%M\". Only when Bob is in some city can he visit it.\ne.g. \n<plan>go_to_place(\"Beijing Railway Hotel\",\"The Great Wall\",\"2023-07-02 7:00\",\"2023-07-02 8:05\")</plan>, <plan>visit(\"The Great Wall\",\"2023-07-02 8:05\",\"2023-07-05 17:00\")</plan>,<plan>go_to_city(\"Shanghai\",\"Beijing\",\"2023-07-02 16:00\",\"2023-07-02 22:30\",\"D1111\")</plan>, <plan>stay_in(\"Beijing\",\"2023-07-02 22:30\",\"2023-07-05 8:00\")</plan>\nYour ultimate goal is to give these plans, there is no need to do anything extra.\n\n--- Your Workflow ---\n1. You will first be given a task.\n2. You should understand the task and devise a plan to complete the task. This plan will contain a series of subtasks that need to be completed.\n\nPLAN AND SUBTASK:\nIf the task cannot be easily solved directly or requires the use of external resources, please assign it to another agent to complete (such as \"find the cheapest train from Beijing to Shanghai in 2023-7-1\"), otherwise you can complete it yourself. You may need to wait for results from other agents before proceeding to the next step of the task. If you need help from other agents, please clearly describe the task objectives, background, and precautions of the subtask. \nA subtask-structure has the following json component and surrounded with <subtask></subtask> as follows:\n<subtask>{\n\"subtask_name\": string, name of the subtask\n\"goal\": string, the main purpose of the subtask, and what will you do to reach this goal?\n\"criticism\": string, what potential problems may the current subtask and goal have?\n\"milestones\": list[string]. what milestones should be achieved to ensure the subtask is done? Make it detailed and specific.\n\"result_format\": optional, what the result should be.}</subtask>\n\n"
      },
      {
        "role": "system",
        "name": "example_user",
        "content": "Task Requirements: Bob is in Shanghai and going to travel in several cities, please make a ticket purchase plan and travel sequence for him.The demands are as follows:\n1. visit ['Beijing']. The order doesn't matter and he needs to return to Shanghai finally.\n2. He is free to travel from 2023.7.1 to 2023.7.20. The budget for transportation is 1000.0 CNY.\n3. Play at least 3 days in Beijing.\n4. If you arrive in a city before 12:00 noon, that day can be counted as a day of play. If it's past 12 o'clock, it doesn't count as a day.\n5. On the basis of completing the above conditions (especially the budget), spend as little time as possible.\n"
      },
      {
        "role": "system",
        "name": "example_assistant",
        "content": "Based on the requirements, we can know that Bob need to go to Beijing from Shanghai, stay in Beijing for 3 days and then go to Shanghai from Beijing.\nGiven the task, the first step is to find available train tickets that fit Bob's schedule and budget.\n<subtask>\n{\n\"subtask_name\": \"find_available_train_tickets\",\n\"goal\": \"Find train tickets from Shanghai to Beijing and back to Shanghai that fit within the travel dates, budget, and allow for at least 3 full days of play in Beijing. If the arrival is before 12:00 noon, it counts as a day of play.\",\n\"criticism\": \"Must ensure that the total cost of the round trip tickets does not exceed the budget of 1000.0 CNY and that the timings allow for at least 3 full days in Beijing for visit. So you need to allow time between train rides(arrival in a city and departure from the city). For each ticket, you must give me the ticket number, origin, destination, departure time, arrival time and the price.\",\n\"milestones\": [\"Identify a suitable train from Shanghai to Beijing.\", \"Identify a return train from Beijing to Shanghai ensuring at least 3 days in Beijing before departing.\", \"Ensure the total cost of both tickets is within the budget of 1000.0 CNY.\"]\n}\n</subtask>\nThen we can get the final plan consists of go_to_city and stay_in.\n<subtask>\n{\n\"subtask_name\": \"get the final plan\",\n\"goal\": \"Formulate a travel plan for Bob's trip from Shanghai to Beijing and back, ensuring it fits within his budget and time constraints, including at least 3 full days in Beijing.\",\n\"criticism\": \"The plan must be concise, focusing on efficient travel and stay arrangements while adhering to the budget and time constraints.\",\n\"milestones\": [\"Include suitable train journeys within the budget.\",\"Plan at least 3 full days in Beijing.\",\"Ensure the overall plan fits within the specified dates and budget.\"],\n\"result_format\": \"A schedule consisting with multiple <plan>go_to_place(...)</plan> and <plan>visit(...)</plan>.    1.go_to_place(origin:str,destination:str,departure_time,arrival_time): go to destination from origin.     2.visit(place:str,begin_time,end_time): visit somewhere from begin_time to end_time. The time should be expressed as %Y-%m-%d %H:%M, e.g. 2023-07-02 16:00.\"\n}\n</subtask>\n"
      },
      {
        "role": "user",
        "content": "Task Requirements:\nBob is in Guangzhou and going to travel in several cities, please make a ticket purchase plan and travel sequence for him.The demands are as follows:\n1. visit ['Beijing', 'Chengdu']. The order doesn't matter and he needs to return to Guangzhou finally.\n2. 10 days (2023-07-04 00:00 ~ 2023-07-14 00:00) for this trip.\n3. Play at least 3 days in Beijing, 3 days in Chengdu.\n4. Stay in any city for a minimum of 24 hours to count as one day.\n5. On the basis of completing the above conditions (especially the time limit), spend as little money as possible.\nCome up with an abstract plan to perform this task in a couple of steps. Give me the subtasks between <subtask> and </subtask>."
      }
    ]
    """,
    "model_path": "/data/labshare/Param/llada/",
    "device": "cuda",
    "use_chat_template": True,
    "gen_length": 512,
    "steps": 128,
    "block_length": 512,
    "temperature": 0.0,
    # 0.0 表示关闭 CFG。
    "cfg_scale": 0.0,
    # 从第几个全局 denoising step 开始做在线分块检测。
    "detect_start_step":56,
    # 每隔多少个 denoising step 做一次在线分块检测。
    "detect_interval": 2,
    # detector 聚合最近多少次检测 snapshot 的 attention/confidence 信号。
    "detector_history_size": 8,
    # 检测模式："line_unit" 表示按换行先切 unit，再聚合 unit 图分 block；"token" 表示直接 token 边界。
    "detection_mode": "line_unit",
    # line_unit 模式只用从 detect_start_step 开始的连续多少步进行识别。
    "line_unit_steps": 8,
    # line_unit 边界评分时，边界左右各看多少个 unit。
    "line_unit_window": 2,
    # line_unit 模式最多输出多少个 block。
    "line_unit_max_blocks": 4,
    # line_unit 模式强制输出多少个 block；0 表示自动选择。
    "line_unit_force_blocks": 0,
    # line_unit 模式每个 block 至少包含多少个 unit。
    "line_unit_min_block_units": 2,
    # line_unit 模式的最小边界分数。
    "line_unit_min_score": 0.0,
    # 每次检测最多选择多少个边界；最终 block 数最多约为 detector_top_k + 1。
    "detector_top_k": 4,
    # 每个候选 block 至少保留多少个有效 token，避免切出过碎的小块。
    "min_span_tokens": 8,
    # 计算边界分数时，边界左右各看多少个 token。
    "boundary_window": 8,
    # 结构提示在边界评分中的辅助权重；不是硬规则，attention/confidence 仍是主信号。
    "structure_weight": 1.2,
    # 边界分数低于该阈值不会被选为候选边界。
    "min_boundary_score": 0.28,
    # 连续多少次检测边界稳定后才冻结分块并切换到 block-wise transfer。
    "stable_rounds": 2,
    # 是否用完整 <subtask>...</subtask> / tool call 结构硬切；默认关闭，让 attention/confidence 主导。
    "prefer_structural_spans": False,
    # 当 prefer_structural_spans=True 时，是否允许多个不完整开始标签形成结构分块。
    "allow_partial_structures": True,
    # 是否写日志文件。
    "save_intermediate": True,
    # 日志输出路径。
    "output_file": "online_detect_log.txt",
    # freeze 后是否使用方案 A：复制完整上下文为 block batch，一次 forward 并行推进多个 block。
    "parallel_block_decode": True,
}


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def safe_topk(conf_row: torch.Tensor, k: int):
    finite = torch.isfinite(conf_row)
    available = int(finite.sum().item())
    if available <= 0:
        return None
    k = min(int(k), available)
    if k <= 0:
        return None
    return torch.topk(conf_row, k=k)


def ceil_transfer_count(remaining: int, steps_left: int) -> int:
    if remaining <= 0:
        return 0
    if steps_left <= 0:
        return remaining
    return max(1, (int(remaining) + int(steps_left) - 1) // int(steps_left))


class LastLayerAttentionProbe:
    def __init__(self, block, num_heads: int, num_kv_heads: int, head_dim: int):
        self.block = block
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.original_attention = block.attention
        self.enabled = False
        self.batch_size = 1
        self.prompt_len = 0
        self.q_cache: Optional[torch.Tensor] = None
        self.k_cache: Optional[torch.Tensor] = None
        self.attention_bias_cache: Optional[torch.Tensor] = None
        self.key_len = 0

        def wrapped_attention(q, k, v, attention_bias=None, layer_past=None, use_cache=False):
            if self.enabled:
                self._capture_qk(
                    q=q,
                    k=k,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                )
            return self.original_attention(
                q,
                k,
                v,
                attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
            )

        block.attention = wrapped_attention

    def enable(
        self,
        batch_size: int = 1,
        prompt_len: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.prompt_len = int(prompt_len)
        self.q_cache = None
        self.k_cache = None
        self.attention_bias_cache = None
        self.key_len = 0
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False
        self.q_cache = None
        self.k_cache = None
        self.attention_bias_cache = None
        self.key_len = 0

    def pop(self, boundary_windows: Sequence[Tuple[int, Sequence[int], Sequence[int]]]) -> Dict[int, float]:
        self.enabled = False
        if self.q_cache is None or self.k_cache is None:
            raise RuntimeError("Last-layer attention probe did not capture q/k states.")
        try:
            return self._compute_boundary_scores(boundary_windows)
        finally:
            self.disable()

    def close(self) -> None:
        self.block.attention = self.original_attention
        self.disable()

    def _mean_attention(
        self,
        q_rows: torch.Tensor,
        k_all: torch.Tensor,
        key_indices: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        key_len: int,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(q_rows, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            row_bias = attention_bias.index_select(dim=2, index=query_indices)[:, :, :, :key_len]
            scores = scores + row_bias
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_rows.dtype)
        return probs.index_select(dim=-1, index=key_indices).mean()

    def _capture_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        batch, query_len, _ = q.size()
        dtype = k.dtype

        if getattr(self.block, "q_norm", None) is not None and getattr(self.block, "k_norm", None) is not None:
            q = self.block.q_norm(q).to(dtype=dtype)
            k = self.block.k_norm(k).to(dtype=dtype)

        q = q.view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, _ = layer_past
            k = torch.cat((past_key, k), dim=-2)

        key_len = k.shape[-2]
        if hasattr(self.block, "rotary_emb"):
            q, k = self.block.rotary_emb(q, k)

        if self.num_kv_heads != self.num_heads:
            assert self.num_heads % self.num_kv_heads == 0
            k = k.repeat_interleave(
                self.num_heads // self.num_kv_heads,
                dim=1,
                output_size=self.num_heads,
            )

        if attention_bias is not None:
            attention_bias = self.block._cast_attn_bias(
                attention_bias[:, :, key_len - query_len: key_len, :key_len],
                dtype,
            )

        self.q_cache = q[: self.batch_size]
        self.k_cache = k[: self.batch_size]
        self.attention_bias_cache = None if attention_bias is None else attention_bias[: self.batch_size]
        self.key_len = int(key_len)

    def _compute_boundary_scores(
        self,
        boundary_windows: Sequence[Tuple[int, Sequence[int], Sequence[int]]],
    ) -> Dict[int, float]:
        q = self.q_cache
        k = self.k_cache
        if q is None or k is None:
            raise RuntimeError("q/k cache is empty.")
        if q.shape[0] != 1:
            raise ValueError("Online detector currently expects batch size 1 attention snapshots.")

        scores: Dict[int, float] = {}
        device = q.device
        attention_bias = self.attention_bias_cache
        for boundary, left_seq, right_seq in boundary_windows:
            if len(left_seq) == 0 or len(right_seq) == 0:
                continue
            left = torch.as_tensor(left_seq, dtype=torch.long, device=device) + self.prompt_len
            right = torch.as_tensor(right_seq, dtype=torch.long, device=device) + self.prompt_len
            left_q = q.index_select(dim=2, index=left)
            right_q = q.index_select(dim=2, index=right)
            left_intra = (
                self._mean_attention(left_q, k, left, attention_bias, self.key_len, left)
                if left.numel() >= 2
                else torch.zeros((), device=device, dtype=q.dtype)
            )
            right_intra = (
                self._mean_attention(right_q, k, right, attention_bias, self.key_len, right)
                if right.numel() >= 2
                else torch.zeros((), device=device, dtype=q.dtype)
            )
            cross_lr = self._mean_attention(left_q, k, right, attention_bias, self.key_len, left)
            cross_rl = self._mean_attention(right_q, k, left, attention_bias, self.key_len, right)
            intra = 0.5 * (left_intra + right_intra)
            cross = 0.5 * (cross_lr + cross_rl)
            scores[int(boundary)] = float((intra - cross).detach().float().cpu().item())
        return scores


class LLaDAOnlineBlockwiseGenerator:
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
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "block_groups"):
            self.blocks = [
                block
                for block_group in self.base_model.transformer.block_groups
                for block in block_group
            ]
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
        self.attn_probe = LastLayerAttentionProbe(
            self.blocks[-1],
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

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

    def extract_qkv_and_apply_rope(self, block, hidden_states: torch.Tensor):
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
        layers_to_extract: Sequence[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_list = []
        for layer_idx in layers_to_extract:
            hidden = all_hidden_states[layer_idx]
            block = self.blocks[layer_idx]
            q, k, _ = self.extract_qkv_and_apply_rope(block, hidden)
            if self.num_kv_heads != self.num_heads:
                k = self._repeat_kv(k, self.num_heads // self.num_kv_heads)
            attn = self.compute_attention_weights(q, k, attention_mask=attention_mask)
            attn_list.append(attn.mean(dim=1).squeeze(0))
        return torch.stack(attn_list, dim=0).mean(dim=0)

    def decode_token(self, token_id: int) -> str:
        if int(token_id) < 0:
            return ""
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\x00", "")

    def decode_with_mask(self, token_ids: Sequence[int]) -> str:
        parts: List[str] = []
        buf: List[int] = []
        for token_id in token_ids:
            if int(token_id) == self.mask_id:
                if buf:
                    parts.append(self.tokenizer.decode(buf, skip_special_tokens=False).replace("\x00", ""))
                    buf = []
                parts.append("<MASK>")
            else:
                buf.append(int(token_id))
        if buf:
            parts.append(self.tokenizer.decode(buf, skip_special_tokens=False).replace("\x00", ""))
        return "".join(parts)

    @staticmethod
    def ranges_from_boundaries(boundaries: Sequence[int], seq_len: int) -> List[Tuple[int, int]]:
        points = [0] + sorted(int(x) for x in boundaries if 0 < int(x) < seq_len) + [seq_len]
        return [(start, end) for start, end in zip(points[:-1], points[1:]) if start < end]

    @staticmethod
    def ranges_from_detection(detection: DetectionResult, seq_len: int) -> List[Tuple[int, int]]:
        if detection.spans:
            return [(max(0, span.start), min(seq_len, span.end)) for span in detection.spans if span.end > span.start]
        return LLaDAOnlineBlockwiseGenerator.ranges_from_boundaries(detection.boundaries, seq_len)

    def _build_snapshot(
        self,
        step_idx: int,
        prompt_len: int,
        x: torch.Tensor,
        mask_index: torch.Tensor,
        avg_attn: torch.Tensor,
        top_confidence: torch.Tensor,
        top_token_ids: torch.Tensor,
    ) -> StepSnapshot:
        response_mask = mask_index[0, prompt_len:].detach().cpu().numpy().astype(bool)
        response_seq = x[0, prompt_len:].detach().cpu().numpy().astype(np.int64)
        pred_ids = top_token_ids[0, prompt_len:].detach().cpu().numpy().astype(np.int64)
        pred_conf = top_confidence[0, prompt_len:].detach().float().cpu().numpy().astype(np.float64)
        draft_ids = np.where(response_mask, pred_ids, response_seq)
        draft_conf = np.where(response_mask, pred_conf, 1.0)
        token_texts = [self.decode_token(token_id) for token_id in draft_ids]
        if avg_attn.shape[0] == response_seq.shape[0]:
            response_attention = avg_attn
        else:
            response_attention = avg_attn[prompt_len:, prompt_len:]
        return StepSnapshot(
            step_idx=int(step_idx),
            attention=response_attention.detach().float().cpu().numpy(),
            confidence=draft_conf,
            token_ids=draft_ids,
            token_texts=token_texts,
            mask=response_mask,
        )

    def _draft_state(
        self,
        prompt_len: int,
        x: torch.Tensor,
        mask_index: torch.Tensor,
        top_confidence: torch.Tensor,
        top_token_ids: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        response_mask = mask_index[0, prompt_len:].detach().cpu().numpy().astype(bool)
        response_seq = x[0, prompt_len:].detach().cpu().numpy().astype(np.int64)
        pred_ids = top_token_ids[0, prompt_len:].detach().cpu().numpy().astype(np.int64)
        pred_conf = top_confidence[0, prompt_len:].detach().float().cpu().numpy().astype(np.float64)
        draft_ids = np.where(response_mask, pred_ids, response_seq)
        draft_conf = np.where(response_mask, pred_conf, 1.0)
        token_texts = [self.decode_token(token_id) for token_id in draft_ids]
        return response_mask, draft_ids, draft_conf, token_texts

    def _candidate_boundary_windows(
        self,
        token_texts: Sequence[str],
        mask: np.ndarray,
        detector: OnlineBlockDetector,
    ) -> List[Tuple[int, List[int], List[int]]]:
        keep = np.asarray(
            [bool(masked) and token.strip() != "" for token, masked in zip(token_texts, mask)],
            dtype=bool,
        )
        seq_len = len(token_texts)
        windows: List[Tuple[int, List[int], List[int]]] = []

        if detector.detection_mode == "line_unit":
            units = build_line_units(token_texts)
            for unit_boundary in range(1, len(units)):
                boundary = int(units[unit_boundary][0])
                left_units = units[max(0, unit_boundary - detector.line_unit_window): unit_boundary]
                right_units = units[unit_boundary: min(len(units), unit_boundary + detector.line_unit_window)]
                left = [idx for unit in left_units for idx in unit if keep[idx]]
                right = [idx for unit in right_units for idx in unit if keep[idx]]
                if left and right:
                    windows.append((boundary, left[-64:], right[:64]))
            return windows

        for boundary in range(1, seq_len):
            left = [
                idx
                for idx in range(max(0, boundary - detector.boundary_window), boundary)
                if keep[idx]
            ]
            right = [
                idx
                for idx in range(boundary, min(seq_len, boundary + detector.boundary_window))
                if keep[idx]
            ]
            if left and right:
                windows.append((boundary, left, right))
        return windows

    def _combine_boundary_scores(
        self,
        attention_scores: Dict[int, float],
        confidence: np.ndarray,
        token_texts: Sequence[str],
        windows: Sequence[Tuple[int, Sequence[int], Sequence[int]]],
        detector: OnlineBlockDetector,
    ) -> List[Tuple[int, float]]:
        if not windows:
            return []

        boundaries = [int(boundary) for boundary, _, _ in windows]
        attn = np.asarray([attention_scores.get(boundary, 0.0) for boundary in boundaries], dtype=np.float64)
        shifts: List[float] = []
        structures: List[float] = []
        penalties: List[float] = []
        conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
        for boundary, left, right in windows:
            left_arr = np.asarray(left, dtype=np.int64)
            right_arr = np.asarray(right, dtype=np.int64)
            left_conf = float(np.mean(conf[left_arr])) if left_arr.size else 0.0
            right_conf = float(np.mean(conf[right_arr])) if right_arr.size else 0.0
            shifts.append(abs(left_conf - right_conf))
            structures.append(_structure_bonus(token_texts, int(boundary)))
            penalties.append(_json_fragment_penalty(token_texts, int(boundary)))

        attention_part = _normalize_scores(attn)
        shift_part = _normalize_scores(np.asarray(shifts, dtype=np.float64))
        structure_part = _normalize_scores(np.asarray(structures, dtype=np.float64))
        penalty_part = _normalize_scores(np.asarray(penalties, dtype=np.float64))
        scores = (
            0.55 * attention_part
            + 0.25 * shift_part
            + float(detector.structure_weight) * 0.20 * structure_part
            - 0.20 * penalty_part
        )
        return [(boundary, float(score)) for boundary, score in zip(boundaries, scores)]

    def _prefetch_spans(
        self,
        prefetcher: AgentPrefetcher,
        detection: DetectionResult,
        snapshot: StepSnapshot,
    ) -> Tuple[List[AgentPrefetchEvent], List[Dict[str, object]]]:
        events: List[AgentPrefetchEvent] = []
        attempts: List[Dict[str, object]] = []
        ranges = self.ranges_from_detection(detection, len(snapshot.token_texts))
        for span_id, (start, end) in enumerate(ranges):
            span_mask = np.ones(end - start, dtype=bool)
            text = draft_text(snapshot.token_texts[start:end], span_mask)
            agent_name = extract_agent_name(text)
            conf = np.asarray(snapshot.confidence[start:end], dtype=np.float64)
            mean_confidence = float(np.mean(np.clip(conf, 0.0, 1.0))) if conf.size else 0.0
            known_prefetch_keys = set(prefetcher.started.keys())
            event = prefetcher.maybe_prefetch(
                span_id=span_id,
                token_texts=snapshot.token_texts[start:end],
                confidence=snapshot.confidence[start:end],
                mask=span_mask,
            )
            new_event = (
                event is not None
                and (int(span_id), event.agent_name) not in known_prefetch_keys
            )
            attempts.append(
                {
                    "span_id": int(span_id),
                    "start": int(start),
                    "end": int(end),
                    "agent_name": agent_name,
                    "mean_confidence": mean_confidence,
                    "prefetch_triggered": new_event,
                    "text_preview": text[:240],
                }
            )
            if new_event:
                events.append(event)
        return events, attempts

    def _prefetch_current_blocks(
        self,
        prefetcher: AgentPrefetcher,
        detection: DetectionResult,
        response_ids: Sequence[int],
        known_agent_names: Dict[int, str],
    ) -> Tuple[List[AgentPrefetchEvent], List[Dict[str, object]]]:
        events: List[AgentPrefetchEvent] = []
        attempts: List[Dict[str, object]] = []
        ranges = self.ranges_from_detection(detection, len(response_ids))
        for span_id, (start, end) in enumerate(ranges):
            if span_id in known_agent_names:
                continue
            block_ids = [int(token_id) for token_id in response_ids[start:end]]
            usable = np.asarray([token_id != self.mask_id for token_id in block_ids], dtype=bool)
            if not np.any(usable):
                continue
            token_texts = [self.decode_token(token_id) for token_id in block_ids]
            confidence = np.where(usable, 1.0, 0.0)
            text = draft_text(token_texts, usable)
            agent_name = extract_agent_name(text)
            if not agent_name:
                continue

            known_prefetch_keys = set(prefetcher.started.keys())
            event = prefetcher.maybe_prefetch(
                span_id=span_id,
                token_texts=token_texts,
                confidence=confidence,
                mask=usable,
            )
            new_event = (
                event is not None
                and (int(span_id), event.agent_name) not in known_prefetch_keys
            )
            known_agent_names[int(span_id)] = agent_name
            attempts.append(
                {
                    "span_id": int(span_id),
                    "start": int(start),
                    "end": int(end),
                    "agent_name": agent_name,
                    "prefetch_triggered": new_event,
                    "text_preview": text[:240],
                }
            )
            if new_event:
                events.append(event)
        return events, attempts

    @staticmethod
    def _write_json_line(file_obj, label: str, payload: Dict[str, object]) -> None:
        file_obj.write(label + "\n")
        file_obj.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def _decode_final_blocks(
        self,
        response_ids: Sequence[int],
        detection: Optional[DetectionResult],
    ) -> List[Dict[str, object]]:
        if detection is None:
            ranges = [(0, len(response_ids))]
        else:
            ranges = self.ranges_from_detection(detection, len(response_ids))
        blocks: List[Dict[str, object]] = []
        for block_id, (start, end) in enumerate(ranges):
            block_ids = list(response_ids[start:end])
            blocks.append(
                {
                    "block_id": int(block_id),
                    "start": int(start),
                    "end": int(end),
                    "token_count": int(end - start),
                    "text": self.decode_with_mask(block_ids),
                }
            )
        return blocks

    def _build_x0_and_confidence(
        self,
        logits: torch.Tensor,
        x_ref: torch.Tensor,
        mask_index: torch.Tensor,
        temperature: float,
        remasking: str,
        logits_eos_inf: bool,
        confidence_eos_eot_inf: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf

        probs = F.softmax(logits, dim=-1)
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            if logits_with_noise.shape[-1] > 126348:
                logits_with_noise[:, :, 126348] = -torch.inf

        if remasking == "low_confidence":
            x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            if confidence_eos_eot_inf:
                x0_p = x0_p.clone()
                x0_p[x0 == 126081] = -torch.inf
                if logits_with_noise.shape[-1] > 126348:
                    x0_p[x0 == 126348] = -torch.inf
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x_ref)
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        return x0, confidence

    def _parallel_block_transfer(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        block_start: int,
        block_end: int,
        gen_length: int,
        frozen_detection: DetectionResult,
        steps_left: int,
        temperature: float,
        remasking: str,
        logits_eos_inf: bool,
        confidence_eos_eot_inf: bool,
    ) -> Tuple[torch.Tensor, int, List[Dict[str, int]]]:
        ranges = self.ranges_from_detection(frozen_detection, gen_length)
        active_ranges: List[Tuple[int, int]] = []
        for start, end in ranges:
            abs_start = max(prompt_len + start, block_start)
            abs_end = min(prompt_len + end, block_end)
            if abs_start >= abs_end:
                continue
            remaining = int((x[:, abs_start:abs_end] == self.mask_id).sum().item())
            if remaining <= 0:
                continue
            active_ranges.append((abs_start, abs_end))

        transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        if not active_ranges:
            return transfer_index, 0, []

        batch_size = len(active_ranges)
        x_batch = x.expand(batch_size, -1).clone()
        attention_batch = attention_mask.expand(batch_size, -1)
        outputs = self.base_model(x_batch, attention_mask=attention_batch, output_hidden_states=False)
        x0, confidence = self._build_x0_and_confidence(
            logits=outputs.logits,
            x_ref=x_batch,
            mask_index=(x_batch == self.mask_id),
            temperature=temperature,
            remasking=remasking,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
        )

        debug: List[Dict[str, int]] = []
        for batch_idx, (abs_start, abs_end) in enumerate(active_ranges):
            remaining = int((x[:, abs_start:abs_end] == self.mask_id).sum().item())
            k_this_span = ceil_transfer_count(remaining, steps_left)
            out = safe_topk(confidence[batch_idx, abs_start:abs_end], k_this_span)
            if out is None:
                continue
            _, local_index = out
            positions = abs_start + local_index
            transfer_index[0, positions] = True
            x[0, positions] = x0[batch_idx, positions]
            debug.append(
                {
                    "batch_idx": int(batch_idx),
                    "start": int(abs_start - prompt_len),
                    "end": int(abs_end - prompt_len),
                    "selected": int(local_index.numel()),
                }
            )
        return transfer_index, int(transfer_index.sum().item()), debug

    def _fill_remaining_masks(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        remasking: str,
        logits_eos_inf: bool,
        confidence_eos_eot_inf: bool,
    ) -> int:
        mask_index = x == self.mask_id
        remaining = int(mask_index.sum().item())
        if remaining <= 0:
            return 0
        outputs = self.base_model(x, attention_mask=attention_mask, output_hidden_states=False)
        x0, _ = self._build_x0_and_confidence(
            logits=outputs.logits,
            x_ref=x,
            mask_index=mask_index,
            temperature=temperature,
            remasking=remasking,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
        )
        x[mask_index] = x0[mask_index]
        return remaining

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        steps: int = 128,
        gen_length: int = 512,
        block_length: int = 512,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        logits_eos_inf: bool = True,
        confidence_eos_eot_inf: bool = True,
        detect_start_step: int = 16,
        detect_interval: int = 4,
        detector: Optional[OnlineBlockDetector] = None,
        prefetch_callback: Optional[Callable[[AgentPrefetchEvent], None]] = None,
        layers_to_extract: Optional[Sequence[int]] = None,
        save_intermediate: bool = False,
        output_file: str = "online_detect_log.txt",
        parallel_block_decode: bool = True,
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

        # Kept for backward-compatible callers; online attention snapshots now use the last-layer probe.
        _ = layers_to_extract
        if detector is None:
            detector = OnlineBlockDetector()
        prefetcher = AgentPrefetcher(callback=prefetch_callback)

        detection_log: List[Dict[str, object]] = []
        prefetch_log: List[Dict[str, object]] = []
        agent_parse_log: List[Dict[str, object]] = []
        frozen_detection: Optional[DetectionResult] = None
        frozen_logged = False
        last_blockwise_decode: Optional[Dict[str, object]] = None
        detection_window_complete = False
        known_agent_names: Dict[int, str] = {}
        frozen_agent_parse_log: List[Dict[str, object]] = []
        final_mask_cleanup_count = 0

        if save_intermediate:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(
                    f"steps={steps}, gen_length={gen_length}, block_length={block_length}, "
                    f"detect_start_step={detect_start_step}, detect_interval={detect_interval}, "
                    f"detection_mode={detector.detection_mode}, history_size={detector.history_size}, "
                    f"parallel_block_decode={parallel_block_decode}\n"
                )
                f.write("=" * 80 + "\n")
                f.write("This log records detection rounds, agent-name parse attempts, and only the final blockwise decode sequence.\n")
                f.write("=" * 80 + "\n")

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
                current_step_1based = global_step + 1

                if frozen_detection is not None and parallel_block_decode and cfg_scale == 0.0:
                    transfer_index, selected_count, parallel_debug = self._parallel_block_transfer(
                        x=x,
                        attention_mask=attention_mask,
                        prompt_len=prompt_len,
                        block_start=block_start,
                        block_end=block_end,
                        gen_length=gen_length,
                        frozen_detection=frozen_detection,
                        steps_left=max(1, steps_per_block - i),
                        temperature=temperature,
                        remasking=remasking,
                        logits_eos_inf=logits_eos_inf,
                        confidence_eos_eot_inf=confidence_eos_eot_inf,
                    )
                    response_ids = x[0, prompt_len:].detach().tolist()
                    last_blockwise_decode = {
                        "step": current_step_1based,
                        "selected": selected_count,
                        "mode": "parallel_batch",
                        "parallel_blocks": parallel_debug,
                        "boundaries": frozen_detection.boundaries,
                        "response_after_update": self.decode_with_mask(response_ids),
                    }
                    events, parse_attempts = self._prefetch_current_blocks(
                        prefetcher=prefetcher,
                        detection=frozen_detection,
                        response_ids=response_ids,
                        known_agent_names=known_agent_names,
                    )
                    if parse_attempts:
                        parse_entry = {
                            "step": current_step_1based,
                            "attempts": parse_attempts,
                        }
                        frozen_agent_parse_log.append(parse_entry)
                        for event in events:
                            prefetch_log.append(event.__dict__.copy())
                        if save_intermediate:
                            with open(output_file, "a", encoding="utf-8") as f:
                                self._write_json_line(
                                    f,
                                    f"[Frozen Agent Name Parse @ Step {current_step_1based}]",
                                    parse_entry,
                                )
                                if events:
                                    self._write_json_line(
                                        f,
                                        f"[Frozen Agent Prefetch Events @ Step {current_step_1based}]",
                                        {"events": [event.__dict__.copy() for event in events]},
                                    )
                                f.write("-" * 80 + "\n")
                    global_step += 1
                    continue

                should_score_boundaries = (
                    frozen_detection is None
                    and not detection_window_complete
                    and current_step_1based >= int(detect_start_step)
                    and (current_step_1based - int(detect_start_step)) >= max(0, detector.history_size - 1)
                    and (
                        detector.detection_mode == "line_unit"
                        or (current_step_1based - int(detect_start_step)) % max(1, int(detect_interval)) == 0
                    )
                )

                if should_score_boundaries:
                    self.attn_probe.enable(batch_size=1, prompt_len=prompt_len)
                try:
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = self.mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                        outputs = self.base_model(x_, attention_mask=attention_mask_, output_hidden_states=False)
                        logits = outputs.logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        outputs = self.base_model(x, attention_mask=attention_mask, output_hidden_states=False)
                        logits = outputs.logits
                except Exception:
                    if should_score_boundaries:
                        self.attn_probe.disable()
                    raise

                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                probs = F.softmax(logits, dim=-1)
                if should_score_boundaries:
                    top_confidence, top_token_ids = torch.max(probs, dim=-1)
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = -torch.inf
                    if logits_with_noise.shape[-1] > 126348:
                        logits_with_noise[:, :, 126348] = -torch.inf

                if remasking == "low_confidence":
                    x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    if confidence_eos_eot_inf:
                        x0_p = x0_p.clone()
                        x0_p[x0 == 126081] = -torch.inf
                        if logits_with_noise.shape[-1] > 126348:
                            x0_p[x0 == 126348] = -torch.inf
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                snapshot = None
                if should_score_boundaries:
                    response_mask, draft_ids, draft_conf, token_texts = self._draft_state(
                        prompt_len=prompt_len,
                        x=x,
                        mask_index=mask_index,
                        top_confidence=top_confidence,
                        top_token_ids=top_token_ids,
                    )
                    boundary_windows = self._candidate_boundary_windows(
                        token_texts=token_texts,
                        mask=response_mask,
                        detector=detector,
                    )
                    attention_scores: Dict[int, float] = {}
                    if boundary_windows:
                        try:
                            attention_scores = self.attn_probe.pop(boundary_windows)
                        except Exception:
                            self.attn_probe.disable()
                            raise
                    else:
                        self.attn_probe.disable()
                    boundary_scores = self._combine_boundary_scores(
                        attention_scores=attention_scores,
                        confidence=draft_conf,
                        token_texts=token_texts,
                        windows=boundary_windows,
                        detector=detector,
                    )
                    detection = detector.detect_from_boundary_scores(
                        step_idx=current_step_1based,
                        token_texts=token_texts,
                        confidence=draft_conf,
                        mask=response_mask,
                        boundary_scores=boundary_scores,
                        freeze_when_stable=True,
                    )
                    snapshot = StepSnapshot(
                        step_idx=int(current_step_1based),
                        attention=np.empty((0, 0), dtype=np.float64),
                        confidence=draft_conf,
                        token_ids=draft_ids,
                        token_texts=token_texts,
                        mask=response_mask,
                    )
                    if detector.detection_mode == "line_unit":
                        detection_window_complete = True
                        detection.frozen = len(detection.spans) > 1
                        if detection.frozen:
                            detector._frozen_result = detection
                    response_before_update = self.decode_with_mask(
                        x[0, prompt_len:].detach().tolist()
                    )
                    events, parse_attempts = self._prefetch_spans(prefetcher, detection, snapshot)
                    detection_entry = {
                        "step": current_step_1based,
                        "block_index": int(num_block),
                        "step_in_block": int(i),
                        "frozen": detection.frozen,
                        "num_blocks": len(detection.spans),
                        "boundaries": detection.boundaries,
                        "boundary_scores": detection.boundary_scores,
                        "spans": [
                            {
                                "span_id": span_id,
                                "start": span.start,
                                "end": span.end,
                                "mean_confidence": span.mean_confidence,
                                "preview": span.text_preview,
                            }
                            for span_id, span in enumerate(detection.spans)
                        ],
                        "response_before_update": response_before_update,
                    }
                    detection_log.append(detection_entry)
                    parse_entry = {
                        "step": current_step_1based,
                        "attempts": parse_attempts,
                    }
                    agent_parse_log.append(parse_entry)
                    for event in events:
                        prefetch_log.append(event.__dict__.copy())
                    for attempt in parse_attempts:
                        agent_name = attempt.get("agent_name")
                        if agent_name:
                            known_agent_names[int(attempt["span_id"])] = str(agent_name)
                    just_frozen = detection.frozen and frozen_detection is None
                    if detection.frozen:
                        frozen_detection = detection
                    if save_intermediate:
                        with open(output_file, "a", encoding="utf-8") as f:
                            if frozen_detection is None or just_frozen or not frozen_logged:
                                self._write_json_line(f, f"[Detection @ Step {current_step_1based}]", detection_entry)
                                if just_frozen:
                                    frozen_logged = True
                            self._write_json_line(f, f"[Agent Name Parse @ Step {current_step_1based}]", parse_entry)
                            if events:
                                self._write_json_line(
                                    f,
                                    f"[Agent Prefetch Events @ Step {current_step_1based}]",
                                    {"events": [event.__dict__.copy() for event in events]},
                                )
                            f.write("-" * 80 + "\n")

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                if frozen_detection is None:
                    for j in range(x0.shape[0]):
                        k_this_step = int(num_transfer_tokens[j, i].item())
                        if k_this_step <= 0:
                            continue
                        out = safe_topk(confidence[j], k_this_step)
                        if out is None:
                            continue
                        _, select_index = out
                        transfer_index[j, select_index] = True
                else:
                    ranges = self.ranges_from_detection(frozen_detection, gen_length)
                    steps_left = max(1, steps_per_block - i)
                    for start, end in ranges:
                        abs_start = max(prompt_len + start, block_start)
                        abs_end = min(prompt_len + end, block_end)
                        if abs_start >= abs_end:
                            continue
                        remaining = int((x[:, abs_start:abs_end] == self.mask_id).sum().item())
                        k_this_span = ceil_transfer_count(remaining, steps_left)
                        for j in range(x0.shape[0]):
                            out = safe_topk(confidence[j, abs_start:abs_end], k_this_span)
                            if out is None:
                                continue
                            _, local_index = out
                            transfer_index[j, abs_start + local_index] = True

                selected_positions = torch.where(transfer_index[0])[0]
                x[transfer_index] = x0[transfer_index]
                if frozen_detection is not None:
                    response_ids = x[0, prompt_len:].detach().tolist()
                    last_blockwise_decode = {
                        "step": current_step_1based,
                        "selected": int(selected_positions.numel()),
                        "mode": "single_forward_blockwise",
                        "boundaries": frozen_detection.boundaries,
                        "response_after_update": self.decode_with_mask(response_ids),
                    }

                global_step += 1

        final_mask_cleanup_count = self._fill_remaining_masks(
            x=x,
            attention_mask=attention_mask,
            temperature=temperature,
            remasking=remasking,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
        )
        final_response_ids = x[0, prompt_len:].detach().cpu().numpy().astype(np.int64)
        final_response_list = final_response_ids.tolist()
        final_blocks = self._decode_final_blocks(final_response_list, frozen_detection)
        if save_intermediate:
            with open(output_file, "a", encoding="utf-8") as f:
                if final_mask_cleanup_count:
                    self._write_json_line(
                        f,
                        "[Final Mask Cleanup]",
                        {"filled_masks": int(final_mask_cleanup_count)},
                    )
                self._write_json_line(
                    f,
                    "[Final Blocks By Frozen Boundary]",
                    {
                        "frozen_step": None if frozen_detection is None else frozen_detection.step_idx,
                        "last_blockwise_step": None if last_blockwise_decode is None else last_blockwise_decode["step"],
                        "boundaries": [] if frozen_detection is None else frozen_detection.boundaries,
                        "blocks": final_blocks,
                    },
                )
                f.write("=" * 80 + "\n")
                self._write_json_line(
                    f,
                    "[Summary]",
                    {
                        "detection_rounds": len(detection_log),
                        "agent_parse_rounds": len(agent_parse_log),
                        "frozen_agent_parse_rounds": len(frozen_agent_parse_log),
                        "prefetch_events": len(prefetch_log),
                        "final_mask_cleanup_count": int(final_mask_cleanup_count),
                        "frozen_boundaries": [] if frozen_detection is None else frozen_detection.boundaries,
                        "last_blockwise_decode_mode": (
                            None if last_blockwise_decode is None else last_blockwise_decode.get("mode")
                        ),
                        "last_blockwise_step": (
                            None if last_blockwise_decode is None else last_blockwise_decode.get("step")
                        ),
                    },
                )
        return {
            "final_response_sequence": final_response_ids,
            "final_response_text": self.tokenizer.decode(final_response_ids, skip_special_tokens=True),
            "detection_log": detection_log,
            "prefetch_log": prefetch_log,
            "agent_parse_log": agent_parse_log,
            "frozen_agent_parse_log": frozen_agent_parse_log,
            "last_blockwise_decode": last_blockwise_decode,
            "final_blocks": final_blocks,
            "frozen_boundaries": [] if frozen_detection is None else frozen_detection.boundaries,
            "final_mask_cleanup_count": int(final_mask_cleanup_count),
        }


def _print_prefetch(event: AgentPrefetchEvent) -> None:
    print(
        f"[prefetch] span={event.span_id} agent={event.agent_name} "
        f"confidence={event.confidence:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Online block detection with block-wise LLaDA decoding.")
    parser.add_argument("--model-path", type=str, default=CONFIG["model_path"])
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--prompt", type=str, default=CONFIG["prompt"])
    parser.add_argument("--gen-length", type=int, default=CONFIG["gen_length"])
    parser.add_argument("--steps", type=int, default=CONFIG["steps"])
    parser.add_argument("--block-length", type=int, default=CONFIG["block_length"])
    parser.add_argument("--temperature", type=float, default=CONFIG["temperature"])
    parser.add_argument("--cfg-scale", type=float, default=CONFIG["cfg_scale"])
    parser.add_argument("--detect-start-step", type=int, default=CONFIG["detect_start_step"])
    parser.add_argument("--detect-interval", type=int, default=CONFIG["detect_interval"])
    parser.add_argument("--detector-history-size", type=int, default=CONFIG["detector_history_size"])
    parser.add_argument("--detection-mode", type=str, default=CONFIG["detection_mode"], choices=["line_unit", "token"])
    parser.add_argument("--line-unit-steps", type=int, default=CONFIG["line_unit_steps"])
    parser.add_argument("--line-unit-window", type=int, default=CONFIG["line_unit_window"])
    parser.add_argument("--line-unit-max-blocks", type=int, default=CONFIG["line_unit_max_blocks"])
    parser.add_argument("--line-unit-force-blocks", type=int, default=CONFIG["line_unit_force_blocks"])
    parser.add_argument("--line-unit-min-block-units", type=int, default=CONFIG["line_unit_min_block_units"])
    parser.add_argument("--line-unit-min-score", type=float, default=CONFIG["line_unit_min_score"])
    parser.add_argument("--detector-top-k", type=int, default=CONFIG["detector_top_k"])
    parser.add_argument("--min-span-tokens", type=int, default=CONFIG["min_span_tokens"])
    parser.add_argument("--boundary-window", type=int, default=CONFIG["boundary_window"])
    parser.add_argument("--structure-weight", type=float, default=CONFIG["structure_weight"])
    parser.add_argument("--min-boundary-score", type=float, default=CONFIG["min_boundary_score"])
    parser.add_argument("--stable-rounds", type=int, default=CONFIG["stable_rounds"])
    parser.add_argument("--prefer-structural-spans", action="store_true")
    parser.add_argument("--no-prefer-structural-spans", action="store_false", dest="prefer_structural_spans")
    parser.add_argument("--allow-partial-structures", action="store_true")
    parser.add_argument("--no-allow-partial-structures", action="store_false", dest="allow_partial_structures")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--no-use-chat-template", action="store_false", dest="use_chat_template")
    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument("--no-save-intermediate", action="store_false", dest="save_intermediate")
    parser.add_argument("--output-file", type=str, default=CONFIG["output_file"])
    parser.add_argument("--parallel-block-decode", action="store_true")
    parser.add_argument("--no-parallel-block-decode", action="store_false", dest="parallel_block_decode")
    parser.set_defaults(
        use_chat_template=CONFIG["use_chat_template"],
        save_intermediate=CONFIG["save_intermediate"],
        prefer_structural_spans=CONFIG["prefer_structural_spans"],
        allow_partial_structures=CONFIG["allow_partial_structures"],
        parallel_block_decode=CONFIG["parallel_block_decode"],
    )
    args = parser.parse_args()

    generator = LLaDAOnlineBlockwiseGenerator(args.model_path, device=args.device)
    if args.use_chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        input_text = generator.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = generator.tokenizer(input_text, add_special_tokens=False, return_tensors="pt").to(args.device)
    else:
        input_text = f"Question: {args.prompt}\nAnswer:"
        inputs = generator.tokenizer(input_text, return_tensors="pt").to(args.device)

    detector = OnlineBlockDetector(
        history_size=args.line_unit_steps if args.detection_mode == "line_unit" else args.detector_history_size,
        detection_mode=args.detection_mode,
        line_unit_window=args.line_unit_window,
        line_unit_max_blocks=args.line_unit_max_blocks,
        line_unit_force_blocks=args.line_unit_force_blocks,
        line_unit_min_block_units=args.line_unit_min_block_units,
        line_unit_min_score=args.line_unit_min_score,
        detect_top_k=args.detector_top_k,
        min_span_tokens=args.min_span_tokens,
        boundary_window=args.boundary_window,
        structure_weight=args.structure_weight,
        min_boundary_score=args.min_boundary_score,
        stable_rounds=args.stable_rounds,
        prefer_structural_spans=args.prefer_structural_spans,
        allow_partial_structures=args.allow_partial_structures,
    )
    result = generator.generate(
        prompt=inputs.input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        detect_start_step=args.detect_start_step,
        detect_interval=args.detect_interval,
        detector=detector,
        prefetch_callback=_print_prefetch,
        save_intermediate=args.save_intermediate,
        output_file=args.output_file,
        parallel_block_decode=args.parallel_block_decode,
    )
    print(f"Frozen boundaries: {result['frozen_boundaries']}")
    print(f"Detection rounds: {len(result['detection_log'])}")
    print(f"Prefetch events: {len(result['prefetch_log'])}")
    if result["last_blockwise_decode"] is not None:
        print(f"Last blockwise decode mode: {result['last_blockwise_decode'].get('mode')}")
    print(f"Final mask cleanup count: {result['final_mask_cleanup_count']}")
    print("Final generated response:")
    print(result["final_response_text"])


if __name__ == "__main__":
    main()
