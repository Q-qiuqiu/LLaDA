import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StepSnapshot:
    step_idx: int
    attention: np.ndarray
    confidence: np.ndarray
    token_ids: np.ndarray
    token_texts: Sequence[str]
    mask: np.ndarray


@dataclass
class DetectedSpan:
    start: int
    end: int
    score: float
    mean_confidence: float
    text_preview: str


@dataclass
class DetectionResult:
    step_idx: int
    frozen: bool
    boundaries: List[int]
    boundary_scores: List[Tuple[int, float]]
    spans: List[DetectedSpan]


def _clean_token(token: str) -> str:
    return token.split(" (id=")[0]


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _keep_token(token: str, is_masked: bool) -> bool:
    if not is_masked:
        return False
    text = _clean_token(token)
    if text.strip() == "":
        return False
    if "<|endoftext|>" in text or "<|eot_id|>" in text:
        return False
    return True


def _keep_text_token(token: str) -> bool:
    text = _clean_token(token)
    if text.strip() == "":
        return False
    if "<|endoftext|>" in text or "<|eot_id|>" in text:
        return False
    return True


def _structure_bonus(tokens: Sequence[str], boundary: int, context_tokens: int = 10) -> float:
    left_start = max(0, boundary - context_tokens)
    right_end = min(len(tokens), boundary + context_tokens)
    left_text = "".join(_clean_token(token) for token in tokens[left_start:boundary])
    right_text = "".join(_clean_token(token) for token in tokens[boundary:right_end])
    around_text = left_text + right_text
    left_norm = re.sub(r"\s+", "", left_text.lower())
    right_norm = re.sub(r"\s+", "", right_text.lower())
    around_norm = re.sub(r"\s+", "", around_text.lower())

    bonus = 0.0
    if any(marker in left_text for marker in ("</subtask>", "</tool_call>", "</agent_call>")):
        bonus += 1.0
    if any(marker in right_text for marker in ("<subtask>", "<tool_call>", "<agent_call>")):
        bonus += 1.0
    if re.search(r"<?/?sub[a-z_]{0,8}>?", right_norm):
        bonus += 0.8
    if re.search(r"(?:^|[<{,\"'\n])sub(?:task|tas|ta|t|b|bb|bbb|sub){0,3}", right_norm):
        bonus += 0.65
    if re.search(r"<?/?(?:tool|agent)[a-z_]{0,12}>?", right_norm):
        bonus += 0.65
    if re.search(r'"(?:subtask_name|name|agent_name|tool_name|server_name|arguments|params)"\s*:', right_text):
        bonus += 0.55
    if re.search(r'"?(?:subtask_name|agent_name|tool_name|server_name|name)"?\s*:', right_text):
        bonus += 0.45
    if any(marker in around_text for marker in ("</subtask><subtask>", "</tool_call><tool_call>", "}{")):
        bonus += 0.7
    if any(marker in around_norm for marker in ("</subtask><sub", "}</sub", "}<sub", "><sub")):
        bonus += 0.7
    if left_text.rstrip().endswith((".", "!", "?", ";", ":", "}", "]", ">")):
        bonus += 0.2
    if left_norm.endswith(("}", "}</subtask>", "}</tool_call>", "}</agent_call>")):
        bonus += 0.35
    return bonus


def _json_fragment_penalty(tokens: Sequence[str], boundary: int, context_tokens: int = 14) -> float:
    left_start = max(0, boundary - context_tokens)
    right_end = min(len(tokens), boundary + context_tokens)
    left_text = "".join(_clean_token(token) for token in tokens[left_start:boundary])
    right_text = "".join(_clean_token(token) for token in tokens[boundary:right_end])
    near_text = left_text + right_text

    # Avoid cutting inside values/lists of the same JSON-ish object unless a
    # clear new block/tag starts on the right.
    if re.search(r"<?/?(?:sub|tool|agent)", re.sub(r"\s+", "", right_text.lower())):
        return 0.0
    penalty = 0.0
    if left_text.count('"') % 2 == 1 or right_text.count('"') % 2 == 1:
        penalty += 0.5
    if left_text.rstrip().endswith((",", ":")):
        penalty += 0.4
    if re.search(r'"(?:goal|criticism|milestones|result_format)"\s*:', near_text):
        penalty += 0.35
    if "[" in left_text and "]" not in left_text:
        penalty += 0.25
    return penalty


def _token_char_offsets(token_texts: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token in token_texts:
        text = _clean_token(token)
        start = cursor
        cursor += len(text)
        offsets.append((start, cursor))
        parts.append(text)
    return "".join(parts), offsets


def _char_to_token_boundary(offsets: Sequence[Tuple[int, int]], char_pos: int, side: str) -> int:
    if not offsets:
        return 0
    if side == "left":
        for idx, (_, end) in enumerate(offsets):
            if end > char_pos:
                return idx
        return len(offsets) - 1
    for idx, (_, end) in enumerate(offsets):
        if end >= char_pos:
            return idx + 1
    return len(offsets)


def _find_tag_positions(text: str, offsets: Sequence[Tuple[int, int]], tag: str) -> List[int]:
    positions: List[int] = []
    search_from = 0
    while True:
        char_pos = text.find(tag, search_from)
        if char_pos < 0:
            break
        positions.append(_char_to_token_boundary(offsets, char_pos, "left"))
        search_from = char_pos + len(tag)
    return positions


def find_structural_spans(
    token_texts: Sequence[str],
    allow_partial_structures: bool = True,
) -> List[Tuple[int, int]]:
    """
    Prefer complete outer call blocks over local token-boundary cuts.
    This keeps JSON fields inside one <subtask>...</subtask> or tool/agent call
    instead of splitting on high-confidence field boundaries.
    """
    seq_len = len(token_texts)
    if seq_len == 0:
        return []

    text, offsets = _token_char_offsets(token_texts)
    tag_pairs = [
        ("<subtask>", "</subtask>"),
        ("<tool_call>", "</tool_call>"),
        ("<TOOL_CALL>", "</TOOL_CALL>"),
        ("<agent_call>", "</agent_call>"),
        ("<AGENT_CALL>", "</AGENT_CALL>"),
    ]

    spans: List[Tuple[int, int]] = []
    open_starts: List[int] = []
    for open_tag, close_tag in tag_pairs:
        open_starts.extend(_find_tag_positions(text, offsets, open_tag))
        search_from = 0
        while True:
            open_pos = text.find(open_tag, search_from)
            if open_pos < 0:
                break
            close_pos = text.find(close_tag, open_pos + len(open_tag))
            if close_pos < 0:
                break
            close_end = close_pos + len(close_tag)
            start_token = _char_to_token_boundary(offsets, open_pos, "left")
            end_token = _char_to_token_boundary(offsets, close_end, "right")
            if end_token > start_token:
                spans.append((start_token, end_token))
            search_from = close_end

    if allow_partial_structures and len(spans) < 2:
        partial_starts = sorted(set(pos for pos in open_starts if 0 <= pos < seq_len))
        if len(partial_starts) >= 2:
            partial_starts[0] = 0
            points = partial_starts + [seq_len]
            return [
                (start, end)
                for start, end in zip(points[:-1], points[1:])
                if end > start
            ]

    if not spans:
        return []

    spans = sorted(spans)
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if not merged or start >= merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    # Attach leading prompt-like fragments to the first structural block and
    # trailing residue to the last block so the generation region is fully covered.
    merged[0] = (0, merged[0][1])
    last_start, _ = merged[-1]
    merged[-1] = (last_start, seq_len)
    return [(start, end) for start, end in merged if end > start]


def spans_to_boundaries(spans: Sequence[Tuple[int, int]], seq_len: int) -> List[int]:
    return [int(end) for _, end in spans[:-1] if 0 < int(end) < int(seq_len)]


def build_line_units(token_texts: Sequence[str]) -> List[List[int]]:
    units: List[List[int]] = []
    cur: List[int] = []
    for idx, token in enumerate(token_texts):
        text = _clean_token(token)
        if "\n" in text or text == "\\n":
            if cur:
                units.append(cur)
                cur = []
            stripped = text.replace("\\n", "").replace("\n", "").strip()
            if stripped and _keep_text_token(stripped):
                cur.append(idx)
            continue
        if not _keep_text_token(text):
            continue
        cur.append(idx)
    if cur:
        units.append(cur)
    if not units:
        units = [[idx] for idx, token in enumerate(token_texts) if _keep_text_token(token)]
    return merge_line_units(units, token_texts)


def _unit_text(unit: Sequence[int], token_texts: Sequence[str]) -> str:
    return "".join(_clean_token(token_texts[idx]) for idx in unit).strip()


def _is_opening_shell(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return False
    if compact in {"<subtask>", "<tool_call>", "<agent_call>", "{", "[", "<subtask>{", "<tool_call>{"}:
        return True
    if compact.startswith(("<sub", "<tool", "<agent")) and '"' not in compact and ":" not in compact:
        return True
    if compact.endswith(("{", "[", ":")) and len(compact) <= 24:
        return True
    return False


def _is_closing_shell(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return False
    if compact in {"}", "]", "</subtask>", "</tool_call>", "</agent_call>", "}</subtask>", "}</tool_call>"}:
        return True
    if compact.startswith(("}", "]", "</")) and len(compact) <= 32:
        return True
    return False


def merge_line_units(
    units: Sequence[Sequence[int]],
    token_texts: Sequence[str],
    min_tokens: int = 4,
) -> List[List[int]]:
    merged: List[List[int]] = []
    pending_prefix: List[int] = []
    for raw_unit in units:
        unit = list(raw_unit)
        text = _unit_text(unit, token_texts)
        if _is_opening_shell(text):
            pending_prefix.extend(unit)
            continue
        if pending_prefix:
            unit = pending_prefix + unit
            pending_prefix = []
        if _is_closing_shell(text) and merged:
            merged[-1].extend(unit)
        else:
            merged.append(unit)
    if pending_prefix:
        if merged:
            merged[-1].extend(pending_prefix)
        else:
            merged.append(pending_prefix)

    idx = 0
    while idx < len(merged):
        if len(merged[idx]) >= min_tokens or len(merged) == 1:
            idx += 1
            continue
        if idx == 0:
            merged[idx + 1] = merged[idx] + merged[idx + 1]
            del merged[idx]
            continue
        merged[idx - 1].extend(merged[idx])
        del merged[idx]
    return merged


def aggregate_units(
    graph: np.ndarray,
    confidence: np.ndarray,
    token_texts: Sequence[str],
    units: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n = len(units)
    unit_graph = np.zeros((n, n), dtype=np.float64)
    unit_conf = np.zeros(n, dtype=np.float64)
    unit_texts: List[str] = []
    for i, group_i in enumerate(units):
        idx_i = np.asarray(group_i, dtype=np.int64)
        unit_conf[i] = float(np.mean(confidence[idx_i])) if idx_i.size else 0.0
        unit_texts.append("".join(_clean_token(token_texts[idx]) for idx in group_i).strip())
        for j, group_j in enumerate(units):
            idx_j = np.asarray(group_j, dtype=np.int64)
            block = graph[np.ix_(idx_i, idx_j)]
            unit_graph[i, j] = float(np.mean(block)) if block.size else 0.0
    return unit_graph, unit_conf, unit_texts


def _unit_boundary_scores(
    unit_graph: np.ndarray,
    unit_conf: np.ndarray,
    unit_texts: Sequence[str],
    units: Sequence[Sequence[int]],
    token_texts: Sequence[str],
    window: int,
    structure_weight: float,
) -> np.ndarray:
    scores = np.zeros(max(0, unit_graph.shape[0] - 1), dtype=np.float64)
    if scores.size == 0:
        return scores
    for boundary in range(1, unit_graph.shape[0]):
        left = np.arange(max(0, boundary - window), boundary)
        right = np.arange(boundary, min(unit_graph.shape[0], boundary + window))
        if left.size == 0 or right.size == 0:
            continue
        cross = float(np.mean(unit_graph[np.ix_(left, right)]))
        intra_parts: List[float] = []
        if left.size >= 2:
            intra_parts.append(float(np.mean(unit_graph[np.ix_(left, left)])))
        if right.size >= 2:
            intra_parts.append(float(np.mean(unit_graph[np.ix_(right, right)])))
        intra = float(np.mean(intra_parts)) if intra_parts else 0.0
        conf_shift = abs(float(np.mean(unit_conf[left])) - float(np.mean(unit_conf[right])))
        token_boundary = int(units[boundary][0])
        struct = _structure_bonus(token_texts, token_boundary)
        penalty = _json_fragment_penalty(token_texts, token_boundary)
        scores[boundary - 1] = (
            0.60 * max(0.0, intra - cross)
            + 0.20 * conf_shift
            + 0.20 * float(structure_weight) * struct
            - 0.20 * penalty
        )
    return scores


def choose_unit_boundaries(
    scores: np.ndarray,
    max_blocks: int,
    force_blocks: int,
    min_block_units: int,
    min_score: float,
) -> List[int]:
    if scores.size == 0:
        return []
    max_splits = max(0, int(max_blocks) - 1)
    if force_blocks and force_blocks > 0:
        max_splits = max(0, int(force_blocks) - 1)
        threshold = -float("inf")
    else:
        threshold = max(float(min_score), float(np.mean(scores) + 0.25 * np.std(scores)))
    ranked = sorted(range(scores.shape[0]), key=lambda idx: float(scores[idx]), reverse=True)
    selected: List[int] = []
    num_units = scores.shape[0] + 1

    def valid_with(boundary: int) -> bool:
        points = [0] + sorted(selected + [boundary]) + [num_units]
        return all((end - start) >= min_block_units for start, end in zip(points[:-1], points[1:]))

    for idx in ranked:
        boundary = idx + 1
        if float(scores[idx]) < threshold:
            continue
        if not valid_with(boundary):
            continue
        selected.append(boundary)
        if len(selected) >= max_splits:
            break
    return sorted(selected)


def detect_line_unit_boundaries(
    graph: np.ndarray,
    confidence: np.ndarray,
    token_texts: Sequence[str],
    unit_window: int,
    structure_weight: float,
    max_blocks: int,
    force_blocks: int,
    min_block_units: int,
    min_score: float,
) -> Tuple[List[int], List[Tuple[int, float]]]:
    units = build_line_units(token_texts)
    if len(units) <= 1:
        return [], []
    unit_graph, unit_conf, unit_texts = aggregate_units(graph, confidence, token_texts, units)
    scores = _unit_boundary_scores(
        unit_graph=unit_graph,
        unit_conf=unit_conf,
        unit_texts=unit_texts,
        units=units,
        token_texts=token_texts,
        window=unit_window,
        structure_weight=structure_weight,
    )
    unit_boundaries = choose_unit_boundaries(
        scores=scores,
        max_blocks=max_blocks,
        force_blocks=force_blocks,
        min_block_units=min_block_units,
        min_score=min_score,
    )
    token_boundaries = [int(units[boundary][0]) for boundary in unit_boundaries]
    debug = [
        (int(units[boundary][0]), float(scores[boundary - 1]))
        for boundary in unit_boundaries
    ]
    return token_boundaries, debug


def aggregate_attention_graph(
    snapshots: Sequence[StepSnapshot],
    local_beta: float = 0.003,
    sym_mode: str = "avg",
    confidence_mode: str = "outer",
    weight_mode: str = "confidence_mean",
) -> Tuple[np.ndarray, np.ndarray]:
    if not snapshots:
        raise ValueError("snapshots must not be empty")

    seq_len = int(snapshots[0].attention.shape[0])
    weights = _step_weights(snapshots, weight_mode)
    idx = np.arange(seq_len, dtype=np.float64)
    local_gate = np.exp(-float(local_beta) * np.abs(idx[:, None] - idx[None, :]))

    graph = np.zeros((seq_len, seq_len), dtype=np.float64)
    confidence = np.zeros(seq_len, dtype=np.float64)
    for weight, snapshot in zip(weights, snapshots):
        mat = np.asarray(snapshot.attention, dtype=np.float64)
        if sym_mode == "avg":
            mat = 0.5 * (mat + mat.T)
        elif sym_mode == "max":
            mat = np.maximum(mat, mat.T)
        else:
            raise ValueError(f"unsupported sym_mode: {sym_mode}")

        conf = np.clip(np.asarray(snapshot.confidence, dtype=np.float64), 0.0, 1.0)
        if confidence_mode == "none":
            conf_gate = 1.0
        elif confidence_mode == "outer":
            conf_gate = np.sqrt(np.outer(conf, conf))
        elif confidence_mode == "row":
            conf_gate = conf[:, None]
        else:
            raise ValueError(f"unsupported confidence_mode: {confidence_mode}")

        graph += float(weight) * mat * conf_gate * local_gate
        confidence += float(weight) * conf

    graph = np.maximum(graph, 0.0)
    np.fill_diagonal(graph, np.diag(graph) + 1e-8)
    return graph, confidence


def _step_weights(snapshots: Sequence[StepSnapshot], weight_mode: str) -> np.ndarray:
    if weight_mode == "uniform":
        weights = np.ones(len(snapshots), dtype=np.float64)
    elif weight_mode == "linear":
        weights = np.arange(1, len(snapshots) + 1, dtype=np.float64)
    elif weight_mode == "confidence_mean":
        weights = np.asarray(
            [max(float(np.mean(np.clip(s.confidence, 0.0, 1.0))), 1e-8) for s in snapshots],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported weight_mode: {weight_mode}")
    return weights / weights.sum()


def score_token_boundaries(
    graph: np.ndarray,
    confidence: np.ndarray,
    token_texts: Sequence[str],
    keep_mask: np.ndarray,
    window: int,
    structure_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    boundaries: List[int] = []
    attention_contrast: List[float] = []
    confidence_shift: List[float] = []
    confidence_valley: List[float] = []
    structure_bonus: List[float] = []
    json_penalty: List[float] = []

    seq_len = int(graph.shape[0])
    for boundary in range(1, seq_len):
        left = np.asarray(
            [idx for idx in range(max(0, boundary - window), boundary) if keep_mask[idx]],
            dtype=np.int64,
        )
        right = np.asarray(
            [idx for idx in range(boundary, min(seq_len, boundary + window)) if keep_mask[idx]],
            dtype=np.int64,
        )
        if left.size == 0 or right.size == 0:
            continue

        cross = float(np.mean(graph[np.ix_(left, right)]))
        intra_parts: List[float] = []
        if left.size >= 2:
            intra_parts.append(float(np.mean(graph[np.ix_(left, left)])))
        if right.size >= 2:
            intra_parts.append(float(np.mean(graph[np.ix_(right, right)])))
        intra = float(np.mean(intra_parts)) if intra_parts else 0.0

        left_conf = float(np.mean(confidence[left]))
        right_conf = float(np.mean(confidence[right]))
        near = np.concatenate([left[-min(left.size, 2) :], right[: min(right.size, 2)]])
        near_conf = float(np.mean(confidence[near])) if near.size else 0.0

        boundaries.append(boundary)
        attention_contrast.append(intra - cross)
        confidence_shift.append(abs(left_conf - right_conf))
        confidence_valley.append(1.0 - near_conf)
        structure_bonus.append(_structure_bonus(token_texts, boundary))
        json_penalty.append(_json_fragment_penalty(token_texts, boundary))

    if not boundaries:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    attention_part = _normalize_scores(np.asarray(attention_contrast, dtype=np.float64))
    shift_part = _normalize_scores(np.asarray(confidence_shift, dtype=np.float64))
    valley_part = _normalize_scores(np.asarray(confidence_valley, dtype=np.float64))
    structure_part = _normalize_scores(np.asarray(structure_bonus, dtype=np.float64))
    penalty_part = _normalize_scores(np.asarray(json_penalty, dtype=np.float64))
    scores = (
        0.50 * attention_part
        + 0.25 * shift_part
        + 0.15 * valley_part
        + float(structure_weight) * 0.20 * structure_part
        - 0.20 * penalty_part
    )
    return np.asarray(boundaries, dtype=np.int64), scores


def choose_boundaries(
    boundaries: np.ndarray,
    scores: np.ndarray,
    keep_mask: np.ndarray,
    top_k: int,
    min_span_tokens: int,
    min_score: float,
) -> List[int]:
    if boundaries.size == 0 or top_k <= 0:
        return []

    def kept_count(start: int, end: int) -> int:
        return int(np.sum(keep_mask[start:end]))

    ranked = sorted(range(scores.shape[0]), key=lambda idx: float(scores[idx]), reverse=True)
    selected: List[int] = []
    seq_len = int(keep_mask.shape[0])
    for score_idx in ranked:
        if float(scores[score_idx]) < float(min_score):
            continue
        boundary = int(boundaries[score_idx])
        points = [0] + sorted(selected + [boundary]) + [seq_len]
        if any(kept_count(start, end) < min_span_tokens for start, end in zip(points[:-1], points[1:])):
            continue
        selected.append(boundary)
        if len(selected) >= top_k:
            break
    return sorted(selected)


def boundaries_close(left: Sequence[int], right: Sequence[int], tolerance: int) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(int(a) - int(b)) <= int(tolerance) for a, b in zip(left, right))


class OnlineBlockDetector:
    def __init__(
        self,
        history_size: int = 4,
        detect_top_k: int = 8,
        min_span_tokens: int = 8,
        boundary_window: int = 8,
        local_beta: float = 0.003,
        structure_weight: float = 1.2,
        min_boundary_score: float = 0.28,
        stable_rounds: int = 2,
        boundary_tolerance: int = 2,
        min_mean_confidence: float = 0.05,
        detection_mode: str = "line_unit",
        line_unit_window: int = 2,
        line_unit_max_blocks: int = 4,
        line_unit_force_blocks: int = 0,
        line_unit_min_block_units: int = 2,
        line_unit_min_score: float = 0.0,
        prefer_structural_spans: bool = False,
        allow_partial_structures: bool = True,
    ) -> None:
        self.history_size = int(history_size)
        self.detect_top_k = int(detect_top_k)
        self.min_span_tokens = int(min_span_tokens)
        self.boundary_window = int(boundary_window)
        self.local_beta = float(local_beta)
        self.structure_weight = float(structure_weight)
        self.min_boundary_score = float(min_boundary_score)
        self.stable_rounds = int(stable_rounds)
        self.boundary_tolerance = int(boundary_tolerance)
        self.min_mean_confidence = float(min_mean_confidence)
        self.detection_mode = str(detection_mode)
        self.line_unit_window = int(line_unit_window)
        self.line_unit_max_blocks = int(line_unit_max_blocks)
        self.line_unit_force_blocks = int(line_unit_force_blocks)
        self.line_unit_min_block_units = int(line_unit_min_block_units)
        self.line_unit_min_score = float(line_unit_min_score)
        self.prefer_structural_spans = bool(prefer_structural_spans)
        self.allow_partial_structures = bool(allow_partial_structures)
        self.snapshots: List[StepSnapshot] = []
        self._last_boundaries: Optional[List[int]] = None
        self._stable_count = 0
        self._frozen_result: Optional[DetectionResult] = None

    @property
    def frozen_result(self) -> Optional[DetectionResult]:
        return self._frozen_result

    def add_snapshot(self, snapshot: StepSnapshot) -> None:
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.history_size:
            self.snapshots = self.snapshots[-self.history_size :]

    def detect(self, freeze_when_stable: bool = True) -> DetectionResult:
        if self._frozen_result is not None:
            return self._frozen_result
        if not self.snapshots:
            raise RuntimeError("No snapshots have been added.")

        graph, confidence = aggregate_attention_graph(self.snapshots, local_beta=self.local_beta)
        source = self.snapshots[-1]
        keep_mask = np.asarray(
            [_keep_token(token, bool(masked)) for token, masked in zip(source.token_texts, source.mask)],
            dtype=bool,
        )
        boundaries, scores = score_token_boundaries(
            graph=graph,
            confidence=confidence,
            token_texts=source.token_texts,
            keep_mask=keep_mask,
            window=self.boundary_window,
            structure_weight=self.structure_weight,
        )
        line_debug: List[Tuple[int, float]] = []
        if self.detection_mode == "line_unit":
            selected, line_debug = detect_line_unit_boundaries(
                graph=graph,
                confidence=confidence,
                token_texts=source.token_texts,
                unit_window=self.line_unit_window,
                structure_weight=self.structure_weight,
                max_blocks=self.line_unit_max_blocks,
                force_blocks=self.line_unit_force_blocks,
                min_block_units=self.line_unit_min_block_units,
                min_score=self.line_unit_min_score,
            )
        else:
            selected = []

        structural_spans = (
            find_structural_spans(
                source.token_texts,
                allow_partial_structures=self.allow_partial_structures,
            )
            if self.prefer_structural_spans and not selected
            else []
        )
        if structural_spans:
            selected = spans_to_boundaries(structural_spans, len(source.token_texts))
        elif not selected:
            selected = choose_boundaries(
                boundaries=boundaries,
                scores=scores,
                keep_mask=keep_mask,
                top_k=self.detect_top_k,
                min_span_tokens=self.min_span_tokens,
                min_score=self.min_boundary_score,
            )

        if self._last_boundaries is not None and boundaries_close(
            selected, self._last_boundaries, self.boundary_tolerance
        ):
            self._stable_count += 1
        else:
            self._stable_count = 1
        self._last_boundaries = list(selected)

        score_map = {int(b): float(s) for b, s in zip(boundaries, scores)}
        score_map.update({int(b): float(s) for b, s in line_debug})
        spans = self._build_spans(selected, confidence, source.token_texts)
        frozen = bool(freeze_when_stable and self._stable_count >= self.stable_rounds and len(spans) > 1)
        result = DetectionResult(
            step_idx=int(source.step_idx),
            frozen=frozen,
            boundaries=list(selected),
            boundary_scores=[(int(boundary), float(score_map.get(int(boundary), 0.0))) for boundary in selected],
            spans=spans,
        )
        if frozen:
            self._frozen_result = result
        return result

    def _build_spans(
        self,
        boundaries: Sequence[int],
        confidence: np.ndarray,
        token_texts: Sequence[str],
    ) -> List[DetectedSpan]:
        points = [0] + [int(x) for x in boundaries] + [len(token_texts)]
        spans: List[DetectedSpan] = []
        for start, end in zip(points[:-1], points[1:]):
            span_conf = np.asarray(confidence[start:end], dtype=np.float64)
            mean_conf = float(np.mean(span_conf)) if span_conf.size else 0.0
            if mean_conf < self.min_mean_confidence:
                continue
            preview = "".join(_clean_token(token) for token in token_texts[start:min(end, start + 32)])
            spans.append(
                DetectedSpan(
                    start=int(start),
                    end=int(end),
                    score=float(mean_conf),
                    mean_confidence=mean_conf,
                    text_preview=preview[:200],
                )
            )
        return spans
