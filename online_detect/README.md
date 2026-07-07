# Online Block Detection

This directory is an independent development copy for online detection experiments.

## Files

- `generate_blockwise.py`: online generation entry. It records step attention/confidence/token drafts, detects response blocks during denoising, freezes stable boundaries, then can switch to batched per-block decoding.
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

With `parallel_block_decode=True`, the implementation uses scheme A after freezing: it copies the full sequence once per detected block, runs one batched model forward, lets each batch item update only its own block span, then merges those selected tokens back into the main sequence. Use `--no-parallel-block-decode` to fall back to the older single-forward block-wise transfer schedule. With `agent_name_priority_decode=True`, blocks whose agent/subtask/tool name has not been parsed yet spend their transfer budget on the detected name field or the block prefix before normal block-wise decoding resumes.

## Log Contents

When `--save-intermediate` is enabled, `--output-file` records:

- every online detection round, including step index, frozen status, block count, boundaries, span previews, and the response sequence at detection time;
- every agent-name parse attempt for each detected span, including extracted `agent_name`/`name`/`tool_name`/`subtask_name` when present;
- prefetch events that passed confidence filtering;
- after freezing, no more detection/parse rounds are written;
- at the end, only the final response split by the frozen boundaries is written.

By default, `detection_mode="line_unit"` matches the offline `yzx_test` style more closely: it uses only the configured eight-step window starting at `detect_start_step`, aggregates attention/confidence over those steps, splits draft text into newline units, merges structural shell units such as `<subtask>`, `{`, `}`, and `</subtask>` into neighboring content units, builds a unit graph, chooses block boundaries on that unit graph, then freezes and uses the resulting token spans for block-wise transfer. Structural text is only an auxiliary score unless `prefer_structural_spans=True`.
