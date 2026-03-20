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
        self.model = model
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
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    save_intermediate=False,
    tokenizer=None,
    output_file="generation_process.txt",
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

    return x


def format_token_labels(tokenizer, token_ids):
    labels = []
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        piece = piece.replace("\n", "\\n").replace("\t", "\\t")
        piece = piece if piece.strip() else repr(piece)
        labels.append(piece[:20])
    return labels


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
):
    extractor = ManualAttentionExtractor(model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_ids = sequence_ids[0].detach().cpu().tolist()
    token_labels = format_token_labels(tokenizer, token_ids)

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
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max-plot-tokens", type=int, default=512)
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
    <agent_call>
    name: <AgentName>
    arguments: <key>: <value>
    ...
    </agent_call>
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

    out = generate(
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

    save_manual_attention_heatmaps(
        model=model,
        tokenizer=tokenizer,
        sequence_ids=out,
        attention_mask=final_attention_mask,
        prompt_length=input_ids.shape[1],
        layers=args.layers,
        output_dir=args.heatmap_dir,
        prefix="generated",
        max_tokens=args.max_plot_tokens,
    )


if __name__ == "__main__":
    main()
