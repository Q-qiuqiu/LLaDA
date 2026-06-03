# Online Block Detection

This directory is an independent development copy for online detection experiments.

## Files

- `generate_blockwise.py`: online generation entry. It records step attention/confidence/token drafts, detects response blocks during denoising, freezes stable boundaries, then switches transfer selection from global top-k to per-block top-k.
- `online_block_detector.py`: xlsx-free detector. It accepts in-memory `StepSnapshot` objects and returns `DetectionResult`.
- `agent_prefetch.py`: lightweight parser/state machine for early agent/tool name prefetch events.
- `*_reference.py`: copied reference files from `yzx_test` for comparison. They are not imported by the new online path.

## Example

Edit the `CONFIG` dictionary at the top of `generate_blockwise.py`, especially `prompt` and the generation/detection hyperparameters. Then run from the repository root:

```bash
python online_detect/generate_blockwise.py
```

Command-line arguments can still override the in-code config:

```bash
python online_detect/generate_blockwise.py \
  --model-path /data/labshare/Param/llada/ \
  --device cuda \
  --prompt "Write subtasks for a multi-agent travel planning workflow." \
  --steps 128 \
  --gen-length 512 \
  --block-length 512 \
  --detect-start-step 24 \
  --detect-interval 4 \
  --save-intermediate \
  --output-file online_detect/online_detect_log.txt
```

The implementation currently keeps one full-sequence model forward per denoising step. The parallelism experiment is in the transfer schedule: after stable block detection, every detected block receives its own token transfer budget in the same step. This is the low-risk stage before splitting spans into separate batched forwards.

## Log Contents

When `--save-intermediate` is enabled, `--output-file` records:

- every online detection round, including step index, frozen status, block count, boundaries, span previews, and the response sequence at detection time;
- every agent-name parse attempt for each detected span, including extracted `agent_name`/`name`/`tool_name`/`subtask_name` when present;
- prefetch events that passed confidence filtering;
- after freezing, no more detection/parse rounds are written;
- at the end, only the final response split by the frozen boundaries is written.

By default, detection is attention/confidence led. Structural text is only an auxiliary score: complete tags such as `<subtask>`, incomplete fragments such as `<sub`, repeated early drafts such as `subsub`, and keys such as `subtask_name` can increase a boundary score, while likely JSON-internal field cuts are penalized. Set `prefer_structural_spans=True` only when you want complete tags to hard-override the attention/confidence boundary scorer.
