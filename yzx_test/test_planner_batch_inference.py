import argparse
import json
import os
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from generate_graph import LLaDAAttentionExtractor


def load_queries(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {input_path}")
    return data


def build_input_ids(extractor: LLaDAAttentionExtractor, sample: Dict[str, Any], device: str) -> torch.Tensor:
    prompt = sample.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Each sample must contain a non-empty 'prompt' message list")

    input_text = extractor.tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = extractor.tokenizer(
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    return inputs.input_ids


def decode_response(extractor: LLaDAAttentionExtractor, generated_tokens) -> str:
    text = extractor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text.replace("\x00", "").strip()


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_batch_inference(args: argparse.Namespace) -> None:
    data = load_queries(args.input_json)
    if args.limit is not None:
        data = data[: args.limit]

    extractor = LLaDAAttentionExtractor(args.model_path, device=args.device)
    results: List[Dict[str, Any]] = []

    for sample in tqdm(data, desc="Running planner queries"):
        query_id = sample.get("query_id")
        if query_id is None:
            raise ValueError("Each sample must contain 'query_id'")

        prompt_ids = build_input_ids(extractor, sample, args.device)
        _, generated_tokens, _, _ = extractor.generate_with_attention(
            prompt=prompt_ids,
            passage_ranges=[],
            layers_to_extract=None,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking="low_confidence",
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            late_conf_threshold=args.late_conf_threshold,
            save_intermediate=False,
        )

        results.append(
            {
                "query_id": query_id,
                "response": decode_response(extractor, generated_tokens),
            }
        )

        if args.save_every > 0 and len(results) % args.save_every == 0:
            save_results(results, args.output_json)

    save_results(results, args.output_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch test planner queries with one model load and save {query_id, response}."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="/home/yzx/LLaDA/yzx_test/planner_type1_0_50_1.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/yzx/LLaDA/yzx_test/planner_type1_0_50_1_predictions.json",
    )
    parser.add_argument("--model-path", type=str, default="/data/labshare/Param/llada/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--tau-min", type=float, default=0.01)
    parser.add_argument("--tau-max", type=float, default=0.05)
    parser.add_argument("--late-conf-threshold", type=float, default=0.9)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Persist partial results every N samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_batch_inference(parse_args())
