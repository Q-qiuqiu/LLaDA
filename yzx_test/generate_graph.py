import torch
import numpy as np
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import os
import math
from tqdm import tqdm


def add_gumbel_noise(logits, temperature):
    """Add Gumbel noise to logits"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_tau_t(step_idx: int, total_steps: int, tau_min: float, tau_max: float) -> float:
    """Linear tau schedule used in DAPD"""
    if total_steps <= 1:
        return tau_max
    alpha = step_idx / (total_steps - 1)
    return tau_min + alpha * (tau_max - tau_min)


def build_mask_attention_scores(attn_matrix: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
    """
    attn_matrix: [L, L], already averaged over heads / chosen layers
    masked_positions: [M]
    return:
        scores: [M, M], s_ij = 0.5 * (a_ij + a_ji)
    """
    sub = attn_matrix.index_select(0, masked_positions).index_select(1, masked_positions)
    scores = 0.5 * (sub + sub.transpose(0, 1))
    return scores


def compute_proxy_degree(scores: torch.Tensor) -> torch.Tensor:
    """
    \tilde d_i = sum_j s_ij
    """
    deg = scores.sum(dim=-1)
    deg = deg - torch.diag(scores)
    return deg


def greedy_independent_set(edge_matrix: torch.Tensor, ranking_score: torch.Tensor) -> torch.Tensor:
    """
    Welsh-Powell style greedy maximal independent set
    edge_matrix: [M, M] bool
    ranking_score: [M]
    return:
        selected_local_idx: indices within masked_positions
    """
    order = torch.argsort(ranking_score, descending=True)
    selected = []

    for idx in order.tolist():
        ok = True
        for s in selected:
            if edge_matrix[idx, s]:
                ok = False
                break
        if ok:
            selected.append(idx)

    if len(selected) == 0:
        selected = [order[0].item()]

    return torch.tensor(selected, device=edge_matrix.device, dtype=torch.long)


def count_remaining_mask_ratio(x: torch.Tensor, prompt_len: int, mask_id: int) -> float:
    gen_region = x[:, prompt_len:]
    remain = (gen_region == mask_id).sum().item()
    total = gen_region.numel()
    return remain / max(total, 1)


class LLaDAAttentionExtractor:
    """
    LLaDA Mask Diffusion Model Attention Extractor + DAPD decoding
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

        # 适配你当前这份实现
        if hasattr(self.model, "transformer"):
            self.base_model = self.model
        elif hasattr(self.model, "model"):
            self.base_model = self.model.model
        else:
            raise AttributeError("Cannot find transformer in model")

        # 适配 block 位置
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

        if hasattr(config, "effective_n_kv_heads"):
            self.num_kv_heads = config.effective_n_kv_heads
        else:
            self.num_kv_heads = self.num_heads

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value tensors for Grouped Query Attention"""
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(
            batch, num_key_value_heads * n_rep, slen, head_dim
        )

    def build_attention_bias(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Convert [B, L] attention_mask into additive attention bias [B, 1, L, L].
        1 means valid token, 0 means padding.
        """
        if attention_mask is None:
            return None

        # [B, 1, 1, L]
        mask = attention_mask[:, None, None, :].to(dtype=torch.float32, device=device)
        mask = (1.0 - mask) * torch.finfo(torch.float32).min
        mask = mask.expand(attention_mask.shape[0], 1, seq_len, seq_len)
        return mask.to(dtype=dtype)

    def extract_qkv_and_apply_rope(self, block, hidden_states, layer_idx):
        """Extract Q/K/V and apply Rotary Position Embedding"""
        with torch.no_grad():
            dtype = hidden_states.dtype

            x_normed = block.attn_norm(hidden_states)

            q = block.q_proj(x_normed)
            k = block.k_proj(x_normed)
            v = block.v_proj(x_normed)

            B, T, C = q.size()

            if getattr(block, "q_norm", None) is not None and getattr(
                block, "k_norm", None
            ) is not None:
                q = block.q_norm(q).to(dtype=dtype)
                k = block.k_norm(k).to(dtype=dtype)

            q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.view(B, T, self.num_kv_heads, C // self.num_heads).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, C // self.num_heads).transpose(1, 2)

            if hasattr(block, "rotary_emb"):
                q, k = block.rotary_emb(q, k)

            return q, k, v

    def compute_attention_weights(
        self,
        q,
        k,
        head_dim,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Compute attention weights manually with optional padding mask"""
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if attention_mask is not None:
            seq_len = q.shape[-2]
            attn_bias = self.build_attention_bias(
                attention_mask=attention_mask,
                seq_len=seq_len,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            attn_weights = attn_weights + attn_bias

        attn_weights = torch.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        return attn_weights

    def compute_step_avg_attention(
        self,
        all_hidden_states,
        layers_to_extract,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Average attention over selected layers and heads at current decoding step.
        Return [L, L] tensor on current device.
        """
        attn_list = []

        for layer_idx in layers_to_extract:
            hidden = all_hidden_states[layer_idx]
            block = self.blocks[layer_idx]

            q, k, v = self.extract_qkv_and_apply_rope(block, hidden, layer_idx)

            if self.num_kv_heads != self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = self._repeat_kv(k, n_rep)

            attn_weights = self.compute_attention_weights(
                q, k, self.head_dim, attention_mask=attention_mask
            )  # [B,H,L,L]
            attn_weights = attn_weights.mean(dim=1).squeeze(0)  # [L,L]
            attn_list.append(attn_weights)

        avg_attn = torch.stack(attn_list, dim=0).mean(dim=0)  # [L,L]
        return avg_attn

    def extract_full_attention_matrix(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Extract full attention matrix (averaged over heads) for a layer"""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states[layer_idx]
            block = self.blocks[layer_idx]

            q, k, v = self.extract_qkv_and_apply_rope(block, hidden_states, layer_idx)

            if self.num_kv_heads != self.num_heads:
                assert self.num_heads % self.num_kv_heads == 0
                n_rep = self.num_heads // self.num_kv_heads
                k = self._repeat_kv(k, n_rep)

            attn_weights = self.compute_attention_weights(
                q, k, self.head_dim, attention_mask=attention_mask
            )
            attn_weights = attn_weights.mean(dim=1).squeeze(0)
            attn_weights = attn_weights.float().cpu().numpy()

            return attn_weights

    def prepare_input(self, data: Dict) -> Tuple[str, List[Tuple[int, int]]]:
        """Prepare input text from data dictionary"""
        if "question" not in data:
            raise ValueError("Expected 'question' field in data")

        question = data["question"]
        input_text = f"Question: {question}\nAnswer:"
        passage_ranges: List[Tuple[int, int]] = []
        return input_text, passage_ranges

    @torch.no_grad()
    def generate_with_attention(
        self,
        prompt,
        passage_ranges,
        layers_to_extract,
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        tau_min=0.01,
        tau_max=0.05,
        late_conf_threshold=0.9,
        save_intermediate: bool = True,
        output_file: str = "dapd_generation_log.txt",
    ):
        """
        DAPD decoding:
        - build attention-induced graph on masked tokens
        - edge score: s_ij = 0.5 * (a_ij + a_ji)
        - proxy degree: sum_j s_ij
        - rank by proxy_degree * confidence
        - greedy maximal independent set
        - when remaining mask ratio < 0.5, switch to confidence > 0.9
        """
        x = torch.full(
            (1, prompt.shape[1] + gen_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_index = x != self.mask_id

        # 构造 attention mask：prompt 可见，生成区一开始也置 1（因为是 mask token，不是 padding）
        attention_mask = torch.ones_like(x, dtype=torch.long, device=self.device)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        # 默认最后两层，更贴近论文
        if layers_to_extract is None:
            layers_to_extract = list(range(max(0, self.num_layers - 2), self.num_layers))

        layer_attentions_dict = {layer_idx: [] for layer_idx in layers_to_extract}
        token_generation_step: Dict[int, Tuple[int, int, int]] = {}

        if save_intermediate:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("")

        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            for i in range(steps_per_block):
                mask_index = x == self.mask_id

                # 当前 block 已经没有 mask，则提前结束
                if not mask_index[:, block_start:block_end].any():
                    break

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)

                    outputs = self.base_model(
                        x_,
                        attention_mask=attention_mask_,
                        output_hidden_states=True
                    )
                    logits = outputs.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                    all_hidden_states = tuple(h[:1] for h in outputs.hidden_states)
                    attn_mask_for_step = attention_mask
                else:
                    outputs = self.base_model(
                        x,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    logits = outputs.logits
                    all_hidden_states = outputs.hidden_states
                    attn_mask_for_step = attention_mask

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                        -1,
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # 后续 block 暂时不动
                x0_p[:, block_end:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                # 计算当前 step 的平均 attention matrix
                avg_attn = self.compute_step_avg_attention(
                    all_hidden_states=all_hidden_states,
                    layers_to_extract=layers_to_extract,
                    attention_mask=attn_mask_for_step,
                )  # [L, L]

                # 当前只在本 block 内的 masked positions 上做 DAPD
                masked_positions = torch.where(mask_index[0])[0]
                masked_positions = masked_positions[
                    (masked_positions >= block_start) & (masked_positions < block_end)
                ]

                if masked_positions.numel() == 0:
                    continue

                remain_ratio = count_remaining_mask_ratio(x, prompt.shape[1], self.mask_id)

                if remain_ratio < 0.5:
                    local_conf = confidence[0, masked_positions]
                    chosen_mask = local_conf > late_conf_threshold
                    selected_positions = masked_positions[chosen_mask]

                    # 避免一步都不选
                    if selected_positions.numel() == 0:
                        best_idx = torch.argmax(local_conf)
                        selected_positions = masked_positions[best_idx:best_idx + 1]
                else:
                    scores = build_mask_attention_scores(avg_attn, masked_positions)  # [M, M]
                    proxy_degree = compute_proxy_degree(scores)  # [M]

                    tau_t = get_tau_t(i, steps_per_block, tau_min, tau_max)
                    edge_matrix = scores > tau_t
                    edge_matrix.fill_diagonal_(False)

                    local_conf = confidence[0, masked_positions]
                    ranking_score = proxy_degree * local_conf

                    selected_local_idx = greedy_independent_set(edge_matrix, ranking_score)
                    selected_positions = masked_positions[selected_local_idx]

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                transfer_index[0, selected_positions] = True

                transferred_positions = torch.where(transfer_index[0])[0].cpu().numpy()
                transferred_tokens = x0[0, transfer_index[0]].cpu().numpy()

                # 保留你原来的 attention logging 逻辑
                for pos_idx, pos in enumerate(transferred_positions):
                    if pos >= prompt.shape[1]:
                        token_id = transferred_tokens[pos_idx]
                        token_generation_step[int(pos)] = (num_block, i, int(token_id))

                        for layer_idx in layers_to_extract:
                            hidden = all_hidden_states[layer_idx]
                            block = self.blocks[layer_idx]

                            q, k, v = self.extract_qkv_and_apply_rope(block, hidden, layer_idx)

                            if self.num_kv_heads != self.num_heads:
                                n_rep = self.num_heads // self.num_kv_heads
                                k = self._repeat_kv(k, n_rep)

                            attn_weights = self.compute_attention_weights(
                                q, k, self.head_dim, attention_mask=attn_mask_for_step
                            )
                            attn_weights = attn_weights.mean(dim=1).squeeze(0).float().cpu().numpy()

                            attn_row = attn_weights[pos, :]

                            passage_attentions = []
                            for start, end in passage_ranges:
                                passage_attn = attn_row[start:end]
                                passage_attentions.append(passage_attn)

                            layer_attentions_dict[layer_idx].append(
                                {
                                    "position": int(pos),
                                    "token_id": int(token_id),
                                    "passage_attentions": passage_attentions,
                                }
                            )

                x[transfer_index] = x0[transfer_index]

                if save_intermediate:
                    ids = x[0, prompt.shape[1]:].detach().tolist()
                    parts = []
                    buf = []

                    for tid in ids:
                        if tid == self.mask_id:
                            if buf:
                                txt = self.tokenizer.decode(buf, skip_special_tokens=False)
                                txt = txt.replace("\x00", "")
                                parts.append(txt)
                                buf = []
                            parts.append("<MASK>")
                        else:
                            buf.append(tid)

                    if buf:
                        txt = self.tokenizer.decode(buf, skip_special_tokens=False)
                        txt = txt.replace("\x00", "")
                        parts.append(txt)

                    intermediate_text = "".join(parts)

                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"Block {num_block+1} Step {i+1}/{steps_per_block} "
                            f"selected={len(selected_positions)} remain_ratio={remain_ratio:.4f}\n"
                        )
                        f.write(intermediate_text + "\n")
                        f.write("-" * 60 + "\n")

        generated_tokens = x[0, prompt.shape[1]:].cpu().numpy()
        return x, generated_tokens, layer_attentions_dict, token_generation_step

    def extract_attentions(
        self,
        data: Dict,
        max_new_tokens: int = 8,
        layers_to_extract: List[int] = None,
        steps: int = 8,
        block_length: int = 8,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        extract_full_matrix: bool = True,
        tau_min: float = 0.01,
        tau_max: float = 0.05,
        late_conf_threshold: float = 0.9,
    ) -> Dict:
        """Extract attention matrices and generation metadata"""
        input_text, passage_ranges = self.prepare_input(data)

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]
        attention_mask = inputs.get("attention_mask", None)

        if layers_to_extract is None:
            layers_to_extract = list(range(max(0, self.num_layers - 2), self.num_layers))

        output_seq, generated_tokens, layer_attentions_dict, token_gen_step = (
            self.generate_with_attention(
                prompt=input_ids,
                passage_ranges=passage_ranges,
                layers_to_extract=layers_to_extract,
                steps=steps,
                gen_length=max_new_tokens,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
                tau_min=tau_min,
                tau_max=tau_max,
                late_conf_threshold=late_conf_threshold,
            )
        )

        result: Dict[str, object] = {
            "answer_tokens": generated_tokens,
            "passage_ranges": np.array(passage_ranges),
            "input_length": input_length,
            "num_generated_tokens": len(generated_tokens),
            "question": data["question"],
            "gold_answer": data.get("answer", ""),
            "token_generation_step": token_gen_step,
        }

        for layer_idx in layers_to_extract:
            attention_data = layer_attentions_dict[layer_idx]
            attention_data.sort(key=lambda x: x["position"])
            layer_attentions = []
            for item in attention_data:
                layer_attentions.append(item["passage_attentions"])
            result[f"layer_{layer_idx}_attentions"] = layer_attentions

        if extract_full_matrix:
            full_sequence = output_seq
            full_attention_mask = torch.ones_like(full_sequence, dtype=torch.long, device=full_sequence.device)

            for layer_idx in layers_to_extract:
                full_attn_matrix = self.extract_full_attention_matrix(
                    full_sequence, layer_idx, attention_mask=full_attention_mask
                )
                result[f"layer_{layer_idx}_full_attention"] = full_attn_matrix

        return result

    def save_attentions(self, result: Dict, output_path: str):
        """Save attention data to .npz file"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        save_dict = {}
        for key, value in result.items():
            if key.endswith("_attentions"):
                save_dict[key] = np.array(value, dtype=object)
            else:
                save_dict[key] = value

        np.savez(output_path, **save_dict)


def process_batch_examples(
    data_json_path: str,
    model_path: str,
    output_dir: str,
    layers_to_extract: List[int] = None,
    max_new_tokens: int = 8,
    steps: int = 8,
    block_length: int = 8,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    extract_full_matrix: bool = True,
    tau_min: float = 0.01,
    tau_max: float = 0.05,
    late_conf_threshold: float = 0.9,
):
    """Process batch of examples from JSONL file"""
    data_list = []
    # with open(data_json_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         try:
    #             data = json.loads(line)
    #             data_list.append(data)
    #         except json.JSONDecodeError:
    #             continue

    # os.makedirs(output_dir, exist_ok=True)
    query = """You are a coordinator agent responsible for decomposing a complex task into sub-tasks and assigning them to specialized agents.

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

    data = {"question": query}
    data_list.append(data)
    extractor = LLaDAAttentionExtractor(model_path)

    for idx, data in enumerate(tqdm(data_list, desc="Processing")):
        example_id = data.get("id", f"example_{idx}")

        try:
            result = extractor.extract_attentions(
                data,
                max_new_tokens=max_new_tokens,
                layers_to_extract=layers_to_extract,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                extract_full_matrix=extract_full_matrix,
                tau_min=tau_min,
                tau_max=tau_max,
                late_conf_threshold=late_conf_threshold,
            )
            print("\n===== RESULT =====")

            # 解码生成的 token
            generated_text = extractor.tokenizer.decode(
                result["answer_tokens"],
                skip_special_tokens=True
            )
            print("Generated Answer:", generated_text)
            print("==================\n")
            # output_path = os.path.join(
            #     output_dir, f"{example_id}_attentions.npz"
            # )
            extractor.save_attentions(result, output_path)

        except Exception as e:
            print(f"Error processing {example_id}: {e}")


if __name__ == "__main__":
    data_json_path = "/home/yzx/LLaDA/yzx_test/data.jsonl"
    model_path = "/data/labshare/Param/llada/"
    output_dir = "/home/yzx/LLaDA/yzx_test/output"

    # 默认取最后两层，更贴近论文
    layers_to_extract = None

    # 建议先用 single-block，更接近论文 DAPD 主设置
    max_new_tokens = 128
    steps = 128
    block_length = 128

    temperature = 0.0
    cfg_scale = 0.0
    extract_full_matrix = True

    # LLaDA 数学/问答任务可先试这个
    tau_min = 0.01
    tau_max = 0.05
    late_conf_threshold = 0.9

    process_batch_examples(
        data_json_path=data_json_path,
        model_path=model_path,
        output_dir=output_dir,
        layers_to_extract=layers_to_extract,
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        extract_full_matrix=extract_full_matrix,
        tau_min=tau_min,
        tau_max=tau_max,
        late_conf_threshold=late_conf_threshold,
    )