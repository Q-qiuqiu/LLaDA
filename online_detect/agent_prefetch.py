import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AgentPrefetchEvent:
    span_id: int
    agent_name: str
    confidence: float
    text_preview: str


def _clean_token(token: str) -> str:
    return token.split(" (id=")[0]


def draft_text(token_texts: Sequence[str], mask: Sequence[bool], unknown: str = "") -> str:
    parts = []
    for text, is_masked in zip(token_texts, mask):
        if is_masked:
            parts.append(_clean_token(text))
        elif unknown:
            parts.append(unknown)
    return "".join(parts)


def extract_agent_name(text: str) -> Optional[str]:
    patterns = [
        r'"agent_name"\s*:\s*"([^"]{1,128})"',
        r'"name"\s*:\s*"([^"]{1,128})"',
        r'"tool_name"\s*:\s*"([^"]{1,128})"',
        r'"subtask_name"\s*:\s*"([^"]{1,128})"',
        r"\bagent_name\s*:\s*([A-Za-z_][A-Za-z0-9_.-]{0,127})",
        r"\bname\s*:\s*([A-Za-z_][A-Za-z0-9_.-]{0,127})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Try a partial JSON object if the span has already become valid JSON.
    left = text.find("{")
    right = text.rfind("}")
    if 0 <= left < right:
        try:
            payload = json.loads(text[left : right + 1])
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("agent_name", "name", "tool_name", "subtask_name"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


class AgentPrefetcher:
    def __init__(
        self,
        callback: Optional[Callable[[AgentPrefetchEvent], None]] = None,
        min_name_confidence: float = 0.65,
    ) -> None:
        self.callback = callback
        self.min_name_confidence = float(min_name_confidence)
        self.started: Dict[Tuple[int, str], AgentPrefetchEvent] = {}

    def maybe_prefetch(
        self,
        span_id: int,
        token_texts: Sequence[str],
        confidence: Sequence[float],
        mask: Sequence[bool],
    ) -> Optional[AgentPrefetchEvent]:
        text = draft_text(token_texts, mask)
        agent_name = extract_agent_name(text)
        if not agent_name:
            return None

        conf = np.asarray(confidence, dtype=np.float64)
        masked = np.asarray(mask, dtype=bool)
        usable_conf = conf[masked] if np.any(masked) else conf
        mean_conf = float(np.mean(np.clip(usable_conf, 0.0, 1.0))) if usable_conf.size else 0.0
        if mean_conf < self.min_name_confidence:
            return None

        key = (int(span_id), agent_name)
        if key in self.started:
            return self.started[key]

        event = AgentPrefetchEvent(
            span_id=int(span_id),
            agent_name=agent_name,
            confidence=mean_conf,
            text_preview=text[:240],
        )
        self.started[key] = event
        if self.callback is not None:
            self.callback(event)
        return event

