import argparse
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
    perplexity score but reduces generation quality. Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute how many tokens should be transitioned at each denoising step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


class ManualAttentionExtractor:
    """
    Reconstruct attention weights manually from hidden states and attention
    projection layers because `output_attentions=True` is unsupported in LLaDA.
    """

    def __init__(self, model):
        self.base_model, self.blocks = self._resolve_backbone(model)
        self.config = self.base_model.config
        self.num_layers = len(self.blocks)
        self.num_heads = self._resolve_attr(
            self.config, "n_heads", "num_attention_heads", "num_heads"
        )
        self.hidden_size = self._resolve_attr(
            self.config, "d_model", "hidden_size", "dim"
        )
        self.num_kv_heads = self._resolve_attr(
            self.config,
            "effective_n_kv_heads",
            "num_key_value_heads",
            "n_kv_heads",
            default=self.num_heads,
        )
        self.head_dim = self.hidden_size // self.num_heads

    @staticmethod
    def _resolve_attr(config, *names, default=None):
        for name in names:
            if hasattr(config, name):
                return getattr(config, name)
        if default is not None:
            return default
        raise AttributeError(f"Missing config attributes: {names}")

    @staticmethod
    def _resolve_backbone(model):
        candidates = [model]
        if hasattr(model, "model"):
            candidates.append(model.model)
        if hasattr(model, "module"):
            candidates.append(model.module)

        for candidate in candidates:
            transformer = getattr(candidate, "transformer", None)
            if transformer is not None and hasattr(transformer, "blocks"):
                return candidate, transformer.blocks

        raise AttributeError("Cannot find `transformer.blocks` in the loaded model.")

    @staticmethod
    def _repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(
            batch, num_key_value_heads * n_rep, seq_len, head_dim
        )

    def extract_qkv_and_apply_rope(self, block, hidden_states):
        dtype = hidden_states.dtype
        x_normed = block.attn_norm(hidden_states)

        q = block.q_proj(x_normed)
        k = block.k_proj(x_normed)
        v = block.v_proj(x_normed)

        batch_size, seq_len, width = q.size()

        if getattr(block, "q_norm", None) is not None and getattr(block, "k_norm", None) is not None:
            q = block.q_norm(q).to(dtype=dtype)
            k = block.k_norm(k).to(dtype=dtype)

        q = q.view(batch_size, seq_len, self.num_heads, width // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, width // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, width // self.num_heads).transpose(1, 2)

        if hasattr(block, "rotary_emb"):
            q, k = block.rotary_emb(q, k)

        if self.num_kv_heads != self.num_heads:
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads.")
            k = self._repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = self._repeat_kv(v, self.num_heads // self.num_kv_heads)

        return q, k, v

    def compute_attention_weights(self, q, k, attention_mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(~key_mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return attn_weights

    @torch.no_grad()
    def extract_layer_attention(self, input_ids, layer_idx, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[layer_idx]
        block = self.blocks[layer_idx]
        q, k, _ = self.extract_qkv_and_apply_rope(block, hidden_states)
        attn_weights = self.compute_attention_weights(q, k, attention_mask=attention_mask)
        return attn_weights.mean(dim=1).squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def extract_layer_hidden_states(self, input_ids, layer_idx, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def extract_step_average_attention(
        self,
        input_ids,
        attention_mask=None,
        layer_indices=None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if layer_indices is None:
            layer_indices = list(range(max(0, self.num_layers - 4), self.num_layers))

        attn_matrices = []
        for layer_idx in layer_indices:
            hidden_states = outputs.hidden_states[layer_idx]
            block = self.blocks[layer_idx]
            q, k, _ = self.extract_qkv_and_apply_rope(block, hidden_states)
            attn_weights = self.compute_attention_weights(q, k, attention_mask=attention_mask)
            attn_matrices.append(attn_weights.mean(dim=1).squeeze(0))

        avg_attention = torch.stack(attn_matrices, dim=0).mean(dim=0)
        return avg_attention.float().cpu().numpy()


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=256,
    block_length=256,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    save_intermediate=False,
    tokenizer=None,
    output_file="generation_process.txt",
    return_history=False,
):
    """
    Sample from LLaDA using the reverse diffusion generation procedure.
    """
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    generation_history = []

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if return_history:
                generation_history.append(x.detach().cpu().clone())

            if save_intermediate and tokenizer is not None:
                ids = x[0, prompt.shape[1] :].detach().tolist()
                parts = []
                buf = []

                for tid in ids:
                    if tid == mask_id:
                        if buf:
                            text = tokenizer.decode(buf, skip_special_tokens=False)
                            parts.append(text.replace("\x00", ""))
                            buf = []
                        parts.append("<MASK>")
                    else:
                        buf.append(tid)

                if buf:
                    text = tokenizer.decode(buf, skip_special_tokens=False)
                    parts.append(text.replace("\x00", ""))

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"Block {num_block + 1} Step {i + 1}/{steps} "
                        f"transferred={int(transfer_index[0].sum().item())}\n"
                    )
                    f.write("".join(parts) + "\n")
                    f.write("-" * 50 + "\n")

    if return_history:
        return x, generation_history
    return x


def decode_token_texts(tokenizer, token_ids):
    return [
        tokenizer.decode([int(token_id)], skip_special_tokens=False)
        for token_id in token_ids
    ]


def format_token_labels(token_texts):
    labels = []
    for piece in token_texts:
        piece = piece.replace("\n", "\\n").replace("\t", "\\t")
        piece = piece if piece.strip() else repr(piece)
        labels.append(piece[:20])
    return labels


def categorize_token(token_text):
    if any(ch in token_text for ch in ("\n", "\r")):
        return "newline"
    stripped = token_text.strip()
    if not stripped:
        return "special"
    if token_text.startswith("<") and token_text.endswith(">"):
        return "special"
    if all(not ch.isalnum() for ch in stripped):
        return "punct"
    return "lexical"


def compute_residual_attention_scores(response_token_ids, response_token_texts, raw_scores):
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    token_types = [categorize_token(text) for text in response_token_texts]

    id_to_indices = {}
    for idx, token_id in enumerate(response_token_ids):
        id_to_indices.setdefault(int(token_id), []).append(idx)

    type_to_indices = {}
    for idx, token_type in enumerate(token_types):
        type_to_indices.setdefault(token_type, []).append(idx)

    baselines = np.zeros_like(raw_scores)
    baseline_sources = []
    for idx, token_id in enumerate(response_token_ids):
        same_id_indices = id_to_indices[int(token_id)]
        if len(same_id_indices) >= 2:
            baselines[idx] = raw_scores[same_id_indices].mean()
            baseline_sources.append("identity")
        else:
            token_type = token_types[idx]
            baselines[idx] = raw_scores[type_to_indices[token_type]].mean()
            baseline_sources.append(f"type:{token_type}")

    residual_scores = raw_scores - baselines
    return token_types, baselines, residual_scores, baseline_sources


def compute_residual_attention_graph(response_attention, response_token_ids, response_token_texts):
    raw_scores = response_attention.mean(axis=0)
    token_types, baselines, residual_scores, baseline_sources = compute_residual_attention_scores(
        response_token_ids=response_token_ids,
        response_token_texts=response_token_texts,
        raw_scores=raw_scores,
    )
    residual_matrix = response_attention - baselines[None, :]
    positive_residual = np.maximum(residual_matrix, 0.0)
    symmetric_graph = 0.5 * (positive_residual + positive_residual.T)
    np.fill_diagonal(symmetric_graph, 0.0)
    return {
        "raw_scores": raw_scores,
        "token_types": token_types,
        "baselines": baselines,
        "residual_scores": residual_scores,
        "baseline_sources": baseline_sources,
        "residual_matrix": residual_matrix,
        "graph": symmetric_graph,
    }


def min_max_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_hidden_convergence_scores(hidden_state_history, prompt_length):
    if not hidden_state_history:
        return np.array([], dtype=np.float64)

    stacked = np.stack(hidden_state_history, axis=0)[:, prompt_length:, :]
    if stacked.shape[0] < 2:
        return np.ones(stacked.shape[1], dtype=np.float64)

    step_drift = np.linalg.norm(np.diff(stacked, axis=0), axis=-1)
    total_drift = step_drift.sum(axis=0)
    convergence_scores = np.ones(step_drift.shape[1], dtype=np.float64)

    for idx in range(step_drift.shape[1]):
        if total_drift[idx] <= 1e-12:
            convergence_scores[idx] = 1.0
            continue
        future_drift = np.cumsum(step_drift[::-1, idx])[::-1]
        stable_step = 0
        for step_idx, remaining in enumerate(future_drift):
            if remaining / total_drift[idx] <= 0.2:
                stable_step = step_idx
                break
        convergence_scores[idx] = 1.0 - stable_step / max(step_drift.shape[0] - 1, 1)

    return convergence_scores


FORBIDDEN_HEADER_TEXTS = {
    "\n",
    "\r",
    "\t",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|mdm_mask|>",
    "<",
    ">",
    ":",
    '"',
    "</",
    ",",
    '",',
    "' '",
}


def is_pure_punct_token(token_text):
    stripped = token_text.strip()
    if not stripped:
        return False
    return all(not ch.isalnum() for ch in stripped)


def is_forbidden_header_token(token_text):
    if token_text in FORBIDDEN_HEADER_TEXTS:
        return True
    if token_text in {"\n", "\r", "\t"}:
        return True
    if token_text.startswith("<|") and token_text.endswith("|>"):
        return True
    if token_text.strip() == "":
        return True
    if is_pure_punct_token(token_text):
        return True
    return False


def branch_contains_head_pattern(token_texts):
    normalized = [text.strip() for text in token_texts]
    sequence = " ".join(normalized)
    has_open_tag = "< agent _call >" in sequence
    has_name = "name" in normalized
    has_agent = any(text.endswith("Agent") for text in normalized)
    return has_open_tag and has_name and has_agent


def branch_is_lexical_continuation(token_texts):
    lexical_count = sum(categorize_token(text) == "lexical" for text in token_texts)
    opening_tag_count = sum(text.strip() in {"<", "</"} for text in token_texts)
    return lexical_count >= max(4, len(token_texts) // 2) and opening_tag_count <= 1


def has_hard_structural_boundary(left_tokens, right_tokens):
    left_tail = [text.strip() for text in left_tokens[-6:]]
    right_head = [text.strip() for text in right_tokens[:6]]
    if ">" in left_tail and "<" in right_head:
        return True
    if "</" in left_tail and "<" in right_head:
        return True
    return False


def merge_super_branches(branch_spans, token_texts, graph, merge_tolerance=1.35):
    if len(branch_spans) <= 1:
        return branch_spans

    merged = [branch_spans[0]]
    for next_start, next_end in branch_spans[1:]:
        prev_start, prev_end = merged[-1]
        left_tokens = token_texts[prev_start:prev_end]
        right_tokens = token_texts[next_start:next_end]

        left_pattern = branch_contains_head_pattern(left_tokens)
        right_continuation = branch_is_lexical_continuation(right_tokens)
        hard_boundary = has_hard_structural_boundary(left_tokens, right_tokens)

        boundary_weight = float(graph[prev_end - 1, next_start]) if prev_end - 1 >= 0 else 0.0
        left_internal = float(graph[prev_start:prev_end, prev_start:prev_end].mean()) if prev_end > prev_start else 0.0
        right_internal = float(graph[next_start:next_end, next_start:next_end].mean()) if next_end > next_start else 0.0
        internal_ref = max((left_internal + right_internal) / 2.0, 1e-12)
        weak_boundary = boundary_weight <= internal_ref * merge_tolerance

        if left_pattern and right_continuation and not hard_boundary and weak_boundary:
            merged[-1] = (prev_start, next_end)
        else:
            merged.append((next_start, next_end))

    return merged


def slice_response_data(token_ids, token_texts, token_labels, prompt_length):
    return {
        "token_ids": token_ids[prompt_length:],
        "token_texts": token_texts[prompt_length:],
        "token_labels": token_labels[prompt_length:],
    }


def detect_branch_spans(graph, min_branch_size=8, max_branches=6, window_size=6):
    num_tokens = graph.shape[0]
    if num_tokens == 0:
        return []
    if num_tokens <= min_branch_size * 2:
        return [(0, num_tokens)]

    boundary_scores = []
    for boundary in range(num_tokens - 1):
        left_start = max(0, boundary - window_size + 1)
        left = np.arange(left_start, boundary + 1)
        right_end = min(num_tokens, boundary + 1 + window_size)
        right = np.arange(boundary + 1, right_end)
        if len(left) == 0 or len(right) == 0:
            boundary_scores.append(np.inf)
            continue
        cross_strength = graph[np.ix_(left, right)].mean()
        boundary_scores.append(float(cross_strength))

    finite_scores = np.array([score for score in boundary_scores if np.isfinite(score)], dtype=np.float64)
    if finite_scores.size == 0:
        return [(0, num_tokens)]

    threshold = float(np.quantile(finite_scores, 0.35))
    candidate_boundaries = []
    for boundary, score in enumerate(boundary_scores):
        if not np.isfinite(score) or score > threshold:
            continue
        left_ok = boundary == 0 or score <= boundary_scores[boundary - 1]
        right_ok = boundary == num_tokens - 2 or score <= boundary_scores[boundary + 1]
        if left_ok and right_ok:
            candidate_boundaries.append((score, boundary))

    selected = []
    for _, boundary in sorted(candidate_boundaries):
        proposed = sorted(selected + [boundary])
        start = 0
        valid = True
        for cut in proposed:
            if cut + 1 - start < min_branch_size:
                valid = False
                break
            start = cut + 1
        if valid and num_tokens - start >= min_branch_size:
            selected = proposed
        if len(selected) + 1 >= max_branches:
            break

    spans = []
    start = 0
    for boundary in selected:
        spans.append((start, boundary + 1))
        start = boundary + 1
    spans.append((start, num_tokens))
    return spans


def compute_branch_bridge_scores(graph, branch_start, branch_end):
    branch_indices = np.arange(branch_start, branch_end)
    outside_indices = np.concatenate(
        [np.arange(0, branch_start), np.arange(branch_end, graph.shape[0])]
    )
    if len(branch_indices) == 0:
        return np.array([], dtype=np.float64)

    branch_graph = graph[np.ix_(branch_indices, branch_indices)]
    internal_strength = branch_graph.sum(axis=1)
    if outside_indices.size == 0:
        external_strength = np.zeros(len(branch_indices), dtype=np.float64)
    else:
        external_strength = graph[np.ix_(branch_indices, outside_indices)].sum(axis=1)
    return np.sqrt(np.maximum(internal_strength, 0.0) * np.maximum(external_strength, 0.0))


def identify_header_span(
    graph,
    residual_scores,
    convergence_scores,
    token_texts,
    branch_start,
    branch_end,
    min_header_tokens=2,
    max_header_tokens=8,
):
    branch_indices = np.arange(branch_start, branch_end)
    if branch_indices.size == 0:
        return branch_start, branch_start

    bridge_scores = compute_branch_bridge_scores(graph, branch_start, branch_end)
    branch_salience = min_max_normalize(residual_scores[branch_indices])
    branch_bridge = min_max_normalize(bridge_scores)
    if convergence_scores.size == 0:
        branch_convergence = np.zeros(branch_indices.size, dtype=np.float64)
    else:
        branch_convergence = min_max_normalize(convergence_scores[branch_indices])

    token_weights = branch_salience + branch_bridge + branch_convergence
    valid_mask = np.array(
        [not is_forbidden_header_token(token_texts[idx]) for idx in branch_indices],
        dtype=np.float64,
    )
    weighted_scores = token_weights * valid_mask

    best = None
    max_span_len = min(max_header_tokens, branch_indices.size)
    min_span_len = min(min_header_tokens, max_span_len)
    if min_span_len <= 0:
        min_span_len = 1

    for span_len in range(min_span_len, max_span_len + 1):
        for start_offset in range(0, branch_indices.size - span_len + 1):
            end_offset = start_offset + span_len
            span_mask = valid_mask[start_offset:end_offset]
            valid_count = int(span_mask.sum())
            if valid_count == 0:
                continue
            if valid_count < max(1, math.ceil(span_len * 0.5)):
                continue
            span_score = float(weighted_scores[start_offset:end_offset].sum() / valid_count)
            candidate = (span_score, valid_count, -start_offset, -span_len, start_offset, end_offset)
            if best is None or candidate > best:
                best = candidate

    if best is not None:
        _, _, _, _, start_offset, end_offset = best
        return branch_start + start_offset, branch_start + end_offset

    lexical_offsets = [
        offset
        for offset, idx in enumerate(branch_indices)
        if categorize_token(token_texts[idx]) == "lexical"
    ]
    if lexical_offsets:
        start_offset = lexical_offsets[0]
        end_offset = min(branch_indices.size, start_offset + min_header_tokens)
        return branch_start + start_offset, branch_start + end_offset

    return branch_start, min(branch_end, branch_start + 1)


def analyze_response_structure(
    extractor,
    tokenizer,
    sequence_ids,
    attention_mask,
    prompt_length,
    layer_idx,
    generation_history=None,
    min_branch_size=8,
    max_branches=6,
):
    token_ids = sequence_ids[0].detach().cpu().tolist()
    token_texts = decode_token_texts(tokenizer, token_ids)
    token_labels = format_token_labels(token_texts)
    response_data = slice_response_data(token_ids, token_texts, token_labels, prompt_length)

    attention_matrix = extractor.extract_layer_attention(
        input_ids=sequence_ids,
        layer_idx=layer_idx,
        attention_mask=attention_mask,
    )
    response_attention = np.asarray(attention_matrix[prompt_length:, prompt_length:], dtype=np.float64)

    if response_attention.size == 0:
        return None

    graph_data = compute_residual_attention_graph(
        response_attention=response_attention,
        response_token_ids=response_data["token_ids"],
        response_token_texts=response_data["token_texts"],
    )
    graph = graph_data["graph"]
    branch_spans = detect_branch_spans(
        graph,
        min_branch_size=min_branch_size,
        max_branches=max_branches,
    )
    branch_spans = merge_super_branches(
        branch_spans=branch_spans,
        token_texts=response_data["token_texts"],
        graph=graph,
    )

    hidden_state_history = []
    for snapshot in generation_history or []:
        hidden_states = extractor.extract_layer_hidden_states(
            input_ids=snapshot.to(sequence_ids.device),
            layer_idx=layer_idx,
            attention_mask=attention_mask,
        )
        hidden_state_history.append(hidden_states)
    convergence_scores = compute_hidden_convergence_scores(hidden_state_history, prompt_length)

    branch_entries = []
    for branch_idx, (start, end) in enumerate(branch_spans, start=1):
        header_start, header_end = identify_header_span(
            graph=graph,
            residual_scores=graph_data["residual_scores"],
            convergence_scores=convergence_scores,
            token_texts=response_data["token_texts"],
            branch_start=start,
            branch_end=end,
        )
        branch_graph = graph[start:end, start:end]
        within_strength = float(branch_graph.mean()) if branch_graph.size > 0 else 0.0
        if start == 0 and end == graph.shape[0]:
            external_strength = 0.0
        else:
            outside_indices = np.concatenate([np.arange(0, start), np.arange(end, graph.shape[0])])
            external_strength = (
                float(graph[np.ix_(np.arange(start, end), outside_indices)].mean())
                if outside_indices.size > 0
                else 0.0
            )
        header_score = (
            float(graph_data["residual_scores"][header_start:header_end].mean())
            if header_end > header_start
            else 0.0
        )
        branch_entries.append(
            {
                "branch_idx": branch_idx,
                "start": start,
                "end": end,
                "header_start": header_start,
                "header_end": header_end,
                "within_strength": within_strength,
                "external_strength": external_strength,
                "header_score": header_score,
            }
        )

    reveal_order = sorted(
        branch_entries,
        key=lambda entry: (entry["header_score"], entry["within_strength"] - entry["external_strength"]),
        reverse=True,
    )

    return {
        "layer_idx": layer_idx,
        "graph": graph,
        "response_data": response_data,
        "branch_entries": branch_entries,
        "reveal_order": reveal_order,
    }


def save_structure_summary(
    model,
    tokenizer,
    sequence_ids,
    attention_mask,
    prompt_length,
    layer_idx,
    output_path,
    generation_history=None,
    min_branch_size=8,
    max_branches=6,
):
    extractor = ManualAttentionExtractor(model)
    structure = analyze_response_structure(
        extractor=extractor,
        tokenizer=tokenizer,
        sequence_ids=sequence_ids,
        attention_mask=attention_mask,
        prompt_length=prompt_length,
        layer_idx=layer_idx,
        generation_history=generation_history,
        min_branch_size=min_branch_size,
        max_branches=max_branches,
    )

    if structure is None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"layer={layer_idx}\n")
            f.write("no response tokens\n")
        return

    graph = structure["graph"]
    response_token_labels = structure["response_data"]["token_labels"]
    branch_entries = structure["branch_entries"]
    reveal_order = structure["reveal_order"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"layer={layer_idx}\n")
        f.write("step1_residual_graph:\n")
        f.write(f"num_tokens={len(response_token_labels)} positive_edge_mass={float(graph.sum()):.6f}\n")
        f.write("step2_branches:\n")
        for entry in branch_entries:
            branch_tokens = " ".join(repr(tok) for tok in response_token_labels[entry["start"] : entry["end"]])
            f.write(
                f"branch{entry['branch_idx']}: span=[{entry['start']},{entry['end']}) "
                f"within={entry['within_strength']:.6f} outside={entry['external_strength']:.6f} "
                f"tokens={branch_tokens}\n"
            )
        f.write("step3_header_spans:\n")
        for entry in branch_entries:
            header_tokens = " ".join(
                repr(tok) for tok in response_token_labels[entry["header_start"] : entry["header_end"]]
            )
            f.write(
                f"branch{entry['branch_idx']}: header=[{entry['header_start']},{entry['header_end']}) "
                f"header_score={entry['header_score']:.6f} tokens={header_tokens}\n"
            )
        f.write("step4_reveal_headers_first:\n")
        for rank, entry in enumerate(reveal_order, start=1):
            f.write(
                f"rank{rank}: branch{entry['branch_idx']} "
                f"reveal_span=[{entry['header_start']},{entry['header_end']})\n"
            )
        f.write("step5_parallel_branch_expansion:\n")
        for entry in reveal_order:
            remaining = [
                idx
                for idx in range(entry["start"], entry["end"])
                if idx < entry["header_start"] or idx >= entry["header_end"]
            ]
            f.write(
                f"branch{entry['branch_idx']}: remaining_positions={remaining}\n"
            )


def save_attention_summary(
    attention_matrix,
    token_ids,
    token_labels,
    token_texts,
    output_path,
    layer_idx,
    prompt_length,
    top_k=10,
):
    attention_matrix = np.asarray(attention_matrix, dtype=np.float64)
    response_attention = attention_matrix[prompt_length:, prompt_length:]
    response_data = slice_response_data(token_ids, token_texts, token_labels, prompt_length)
    response_token_labels = response_data["token_labels"]

    if response_attention.size == 0 or len(response_data["token_ids"]) == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"layer={layer_idx}\n")
            f.write("no response tokens\n")
        return

    raw_scores = response_attention.mean(axis=0)
    token_types, baselines, residual_scores, baseline_sources = compute_residual_attention_scores(
        response_token_ids=response_data["token_ids"],
        response_token_texts=response_data["token_texts"],
        raw_scores=raw_scores,
    )

    top_k = min(top_k, len(raw_scores))
    raw_top_indices = np.argsort(raw_scores)[::-1][:top_k]
    residual_top_indices = np.argsort(residual_scores)[::-1][:top_k]

    type_scores = {"newline": 0.0, "punct": 0.0, "special": 0.0, "lexical": 0.0}
    total_score = float(raw_scores.sum())
    if total_score > 0:
        for idx, score in enumerate(raw_scores):
            token_type = token_types[idx]
            type_scores[token_type] += float(score)
        for key in type_scores:
            type_scores[key] /= total_score

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"layer={layer_idx}\n")
        f.write("raw_top_tokens:\n")
        for rank, idx in enumerate(raw_top_indices, start=1):
            f.write(
                f"top{rank}: idx={idx} token={response_token_labels[idx]!r} "
                f"token_id={response_data['token_ids'][idx]} type={token_types[idx]} "
                f"raw={raw_scores[idx]:.6f}\n"
            )
        f.write("residual_top_tokens:\n")
        for rank, idx in enumerate(residual_top_indices, start=1):
            f.write(
                f"top{rank}: idx={idx} token={response_token_labels[idx]!r} "
                f"token_id={response_data['token_ids'][idx]} type={token_types[idx]} "
                f"raw={raw_scores[idx]:.6f} baseline={baselines[idx]:.6f} "
                f"residual={residual_scores[idx]:.6f} baseline_source={baseline_sources[idx]}\n"
            )
        f.write("type_distribution:\n")
        f.write(
            "newline={:.2f}, punct={:.2f}, special={:.2f}, lexical={:.2f}\n".format(
                type_scores["newline"],
                type_scores["punct"],
                type_scores["special"],
                type_scores["lexical"],
            )
        )
        f.write("all_response_tokens:\n")
        for idx, (token_id, token_label) in enumerate(
            zip(response_data["token_ids"], response_token_labels)
        ):
            f.write(
                f"idx={idx} token={token_label!r} token_id={token_id} type={token_types[idx]} "
                f"raw={raw_scores[idx]:.6f} baseline={baselines[idx]:.6f} "
                f"residual={residual_scores[idx]:.6f} baseline_source={baseline_sources[idx]}\n"
            )


def plot_attention_heatmap(
    attention_matrix,
    token_labels,
    output_path,
    title,
    max_tokens=80,
    start_idx=0,
):
    attention_matrix = np.asarray(attention_matrix)
    seq_len = attention_matrix.shape[0]
    if start_idx < 0 or start_idx >= seq_len:
        raise ValueError(f"start_idx {start_idx} out of range for seq_len {seq_len}")

    end_idx = seq_len
    if max_tokens is not None:
        end_idx = min(seq_len, start_idx + max_tokens)

    matrix_view = attention_matrix[start_idx:end_idx, start_idx:end_idx]
    labels_view = token_labels[start_idx:end_idx]
    view_len = matrix_view.shape[0]

    fig_size = max(8, min(24, view_len * 0.35))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(matrix_view, cmap="Blues", aspect="auto", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04, label="Attention")
    plt.title(title)
    plt.xticks(range(view_len), labels_view, rotation=90, fontsize=8)
    plt.yticks(range(view_len), labels_view, fontsize=8)
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def symmetrize_attention_matrix(attention_matrix):
    attention_matrix = np.asarray(attention_matrix, dtype=np.float64)
    return 0.5 * (attention_matrix + attention_matrix.T)


def identify_attention_sinks(attention_matrix, sink_threshold_std=2.0):
    attention_matrix = np.asarray(attention_matrix, dtype=np.float64)
    if attention_matrix.ndim != 2 or attention_matrix.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), 0.0

    incoming_mean = attention_matrix.mean(axis=0)
    mean_value = float(incoming_mean.mean())
    std_value = float(incoming_mean.std())
    threshold = mean_value + sink_threshold_std * std_value

    if std_value < 1e-12:
        sink_indices = np.array([], dtype=np.int64)
    else:
        sink_indices = np.where(incoming_mean > threshold)[0].astype(np.int64)

    return sink_indices, incoming_mean, threshold


def filter_sink_tokens(attention_matrix, token_labels, sink_threshold_std=2.0):
    attention_matrix = np.asarray(attention_matrix, dtype=np.float64)
    token_labels = list(token_labels)

    sink_indices, incoming_mean, threshold = identify_attention_sinks(
        attention_matrix,
        sink_threshold_std=sink_threshold_std,
    )

    keep_mask = np.ones(attention_matrix.shape[0], dtype=bool)
    keep_mask[sink_indices] = False
    filtered_matrix = attention_matrix[np.ix_(keep_mask, keep_mask)]
    filtered_labels = [label for idx, label in enumerate(token_labels) if keep_mask[idx]]

    return {
        "filtered_matrix": filtered_matrix,
        "filtered_labels": filtered_labels,
        "sink_indices": sink_indices,
        "sink_labels": [token_labels[idx] for idx in sink_indices.tolist()],
        "incoming_mean": incoming_mean,
        "threshold": threshold,
    }


def normalize_selected_steps(selected_steps, total_steps):
    requested_steps = [int(step) for step in selected_steps]
    if not requested_steps:
        return [total_steps]

    if min(requested_steps) < 1:
        raise ValueError("selected steps must be >= 1")

    max_requested = max(requested_steps)
    if max_requested <= total_steps:
        normalized_steps = []
        for step in requested_steps:
            if step not in normalized_steps:
                normalized_steps.append(step)
        return normalized_steps

    if max_requested == 1:
        return [1]

    scaled_steps = []
    for step in requested_steps:
        scaled = 1 + round((step - 1) * (total_steps - 1) / (max_requested - 1))
        scaled = min(max(scaled, 1), total_steps)
        if scaled not in scaled_steps:
            scaled_steps.append(scaled)
    return scaled_steps


def save_selected_step_average_attention_heatmaps(
    model,
    tokenizer,
    generation_history,
    attention_mask,
    prompt_length,
    output_dir,
    prefix="generated",
    max_tokens=80,
    num_avg_layers=4,
    selected_steps=None,
    sink_threshold_std=2.0,
):
    extractor = ManualAttentionExtractor(model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not generation_history:
        return

    layer_indices = list(range(max(0, extractor.num_layers - num_avg_layers), extractor.num_layers))
    total_steps = len(generation_history)
    if selected_steps is None:
        selected_steps = [1, total_steps]
    selected_steps = normalize_selected_steps(selected_steps, total_steps)

    for step_idx in selected_steps:
        sequence_ids = generation_history[step_idx - 1].to(model.device)
        token_ids = sequence_ids[0].detach().cpu().tolist()
        token_texts = decode_token_texts(tokenizer, token_ids)
        token_labels = format_token_labels(token_texts)
        response_labels = token_labels[prompt_length:]

        if len(response_labels) == 0:
            continue

        avg_attention = extractor.extract_step_average_attention(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
        )
        response_attention = np.asarray(
            avg_attention[prompt_length:, prompt_length:],
            dtype=np.float64,
        )

        if response_attention.size == 0:
            continue

        sink_filtered_raw = filter_sink_tokens(
            response_attention,
            response_labels,
            sink_threshold_std=sink_threshold_std,
        )
        filtered_interaction_input = sink_filtered_raw["filtered_matrix"]
        filtered_labels = sink_filtered_raw["filtered_labels"]

        if filtered_interaction_input.size == 0 or len(filtered_labels) == 0:
            continue

        interaction_score = symmetrize_attention_matrix(filtered_interaction_input)
        file_prefix = f"{prefix}_step_{step_idx:03d}_last{len(layer_indices)}layers_avg"

        np.save(output_dir / f"{file_prefix}_raw_attention.npy", response_attention)
        np.save(output_dir / f"{file_prefix}_sink_filtered_attention.npy", filtered_interaction_input)
        np.save(output_dir / f"{file_prefix}_interaction.npy", interaction_score)
        with open(output_dir / f"{file_prefix}_sink_filter.txt", "w", encoding="utf-8") as f:
            f.write(f"step={step_idx}\n")
            f.write(f"sink_threshold_std={sink_threshold_std}\n")
            f.write(f"sink_threshold={sink_filtered_raw['threshold']:.6f}\n")
            f.write(f"num_filtered={len(sink_filtered_raw['sink_indices'])}\n")
            for idx, label in zip(
                sink_filtered_raw["sink_indices"].tolist(),
                sink_filtered_raw["sink_labels"],
            ):
                f.write(
                    f"idx={idx} token={label!r} incoming_mean={sink_filtered_raw['incoming_mean'][idx]:.6f}\n"
                )

        plot_attention_heatmap(
            attention_matrix=response_attention,
            token_labels=response_labels,
            output_path=output_dir / f"{file_prefix}_raw_attention.png",
            title=(
                f"Step {step_idx} Raw Attention "
                f"(avg last {len(layer_indices)} layers, avg all heads)"
            ),
            max_tokens=max_tokens,
            start_idx=0,
        )
        plot_attention_heatmap(
            attention_matrix=filtered_interaction_input,
            token_labels=filtered_labels,
            output_path=output_dir / f"{file_prefix}_sink_filtered_attention.png",
            title=(
                f"Step {step_idx} Sink-Filtered Attention "
                f"(avg last {len(layer_indices)} layers, avg all heads)"
            ),
            max_tokens=max_tokens,
            start_idx=0,
        )
        plot_attention_heatmap(
            attention_matrix=interaction_score,
            token_labels=filtered_labels,
            output_path=output_dir / f"{file_prefix}_interaction.png",
            title=(
                f"Step {step_idx} Symmetrized Interaction "
                f"(sink-filtered, avg last {len(layer_indices)} layers, avg all heads)"
            ),
            max_tokens=max_tokens,
            start_idx=0,
        )


def save_manual_attention_heatmaps(
    model,
    tokenizer,
    sequence_ids,
    attention_mask,
    prompt_length,
    layers,
    output_dir,
    prefix="sample",
    max_tokens=80,
    summary_top_k=10,
):
    extractor = ManualAttentionExtractor(model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_ids = sequence_ids[0].detach().cpu().tolist()
    token_texts = decode_token_texts(tokenizer, token_ids)
    token_labels = format_token_labels(token_texts)

    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= extractor.num_layers:
            raise ValueError(
                f"layer {layer_idx} out of range, model has {extractor.num_layers} layers"
            )

        attn_matrix = extractor.extract_layer_attention(
            input_ids=sequence_ids,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
        )
        np.save(output_dir / f"{prefix}_layer_{layer_idx}_attention.npy", attn_matrix)
        save_attention_summary(
            attention_matrix=attn_matrix,
            token_ids=token_ids,
            token_labels=token_labels,
            token_texts=token_texts,
            output_path=output_dir / f"{prefix}_layer_{layer_idx}_attention.txt",
            layer_idx=layer_idx,
            prompt_length=prompt_length,
            top_k=summary_top_k,
        )
        plot_attention_heatmap(
            attention_matrix=attn_matrix,
            token_labels=token_labels,
            output_path=output_dir / f"{prefix}_layer_{layer_idx}_attention.png",
            title=f"Layer {layer_idx} Response Attention",
            max_tokens=max_tokens,
            start_idx=prompt_length,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate with LLaDA and manually reconstruct attention heatmaps."
    )
    parser.add_argument("--model-path", type=str, default="/data/labshare/Param/llada/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 30, 31])
    parser.add_argument("--step-avg-last-layers", type=int, default=4)
    parser.add_argument(
        "--selected-heatmap-steps",
        type=int,
        nargs="+",
        default=[1, 32, 64, 96, 128],
    )
    parser.add_argument("--sink-threshold-std", type=float, default=2.0)
    parser.add_argument("--max-plot-tokens", type=int, default=512)
    parser.add_argument("--summary-top-k", type=int, default=30)
    parser.add_argument("--structure-layer", type=int, default=None)
    parser.add_argument("--structure-min-branch-size", type=int, default=8)
    parser.add_argument("--structure-max-branches", type=int, default=6)
    parser.add_argument(
        "--heatmap-dir",
        type=str,
        default="/home/yzx/LLaDA/yzx_test/attention_heatmaps",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save the denoising text trajectory to a log file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/yzx/LLaDA/yzx_test/denoise_log_128_128_128.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != 126336

    prompt_text = '''
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
    '''

    messages = [{"role": "user", "content": prompt_text}]
    prompts = [
        tokenizer.apply_chat_template(
            [message], add_generation_prompt=True, tokenize=False
        )
        for message in messages
    ]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded_outputs["input_ids"].to(device)
    attention_mask = encoded_outputs["attention_mask"].to(device)

    if args.save_intermediate and os.path.exists(args.output_file):
        os.remove(args.output_file)

    out, generation_history = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking="low_confidence",
        save_intermediate=args.save_intermediate,
        tokenizer=tokenizer,
        output_file=args.output_file,
        return_history=True,
    )

    generated_text = tokenizer.batch_decode(
        out[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    for text in generated_text:
        print(text)
        print("-" * 50)

    final_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], args.gen_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=-1,
    )

    save_selected_step_average_attention_heatmaps(
        model=model,
        tokenizer=tokenizer,
        generation_history=generation_history,
        attention_mask=final_attention_mask,
        prompt_length=input_ids.shape[1],
        output_dir=Path(args.heatmap_dir),
        prefix="generated",
        max_tokens=args.max_plot_tokens,
        num_avg_layers=args.step_avg_last_layers,
        selected_steps=args.selected_heatmap_steps,
        sink_threshold_std=args.sink_threshold_std,
    )

    structure_layer = args.structure_layer
    if structure_layer is None:
        structure_layer = args.layers[-1]
    save_structure_summary(
        model=model,
        tokenizer=tokenizer,
        sequence_ids=out,
        attention_mask=final_attention_mask,
        prompt_length=input_ids.shape[1],
        layer_idx=structure_layer,
        output_path=Path(args.heatmap_dir) / f"generated_layer_{structure_layer}_structure.txt",
        generation_history=generation_history,
        min_branch_size=args.structure_min_branch_size,
        max_branches=args.structure_max_branches,
    )


if __name__ == "__main__":
    main()
