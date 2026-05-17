"""
llm.py — OpenAI-compatible HTTP client.

Responsibilities:
  - Test endpoint reachability.
  - Build the request body with correct OpenAI param names AND common
    locally-served extensions (top_k, min_p, repetition_penalty).
  - Stream chat completions and surface text deltas to the caller.
  - Detect <think> reasoning blocks in the stream and split them off.
  - Handle context-window compaction: keep the system message and most
    recent turns; crop from the middle when the conversation exceeds the
    configured threshold.
"""
import json
import time
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from .utils import est_messages_tokens, est_tokens


# Per-user send locks. Two messages from the same user are serialized;
# different users are fully concurrent.
_user_send_locks: Dict[str, threading.Lock] = {}
_user_send_guard = threading.Lock()


def _user_send_lock(username: str) -> threading.Lock:
    with _user_send_guard:
        if username not in _user_send_locks:
            _user_send_locks[username] = threading.Lock()
        return _user_send_locks[username]


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def test_endpoint(endpoint: str, api_key: str = "",
                  timeout: float = 4.0) -> Tuple[bool, str, List[str]]:
    """
    Returns (ok, message, available_models).
    Tries /models first, falls back to a HEAD on the base URL.
    """
    endpoint = (endpoint or "").rstrip("/")
    if not endpoint:
        return False, "No endpoint configured.", []
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Try /models (OpenAI-compatible servers typically expose this)
    try:
        r = requests.get(endpoint + "/models", headers=headers, timeout=timeout)
        if r.status_code == 200:
            try:
                data = r.json()
                ids = []
                for item in (data.get("data") or []):
                    mid = item.get("id") if isinstance(item, dict) else None
                    if mid:
                        ids.append(mid)
                return True, "Connected.", ids
            except ValueError:
                return True, "Connected (no model list).", []
        elif r.status_code in (401, 403):
            return False, f"Auth error ({r.status_code}). Check API key.", []
    except requests.RequestException as exc:
        # Fall through to second probe
        last = str(exc)
    else:
        last = f"HTTP {r.status_code}"

    # Fall back to a base-URL ping
    try:
        r = requests.get(endpoint, headers=headers, timeout=timeout)
        if r.status_code < 500:
            return True, f"Reachable (status {r.status_code}).", []
        return False, f"Server error: HTTP {r.status_code}", []
    except requests.RequestException as exc:
        return False, f"Unreachable: {exc.__class__.__name__}", []


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

def compact_messages(system_msg: str, history: List[Dict[str, str]],
                     context_size: int, threshold: float = 0.8,
                     reserve_for_reply: int = 512
                     ) -> Tuple[List[Dict[str, str]], bool]:
    """
    Ensure system + history fits within `context_size` tokens.
    Crop from the middle of `history` (keeping earliest + most recent turns)
    until we drop under the threshold. Returns (new_history, was_compacted).
    """
    budget = int(context_size * threshold) - reserve_for_reply
    if budget < 256:
        budget = 256

    sys_tokens = est_tokens(system_msg) + 4
    used = sys_tokens + est_messages_tokens(history)
    if used <= budget:
        return history, False

    # Keep first 2 messages (often the earliest exchange establishes character)
    # and as many of the most-recent messages as fit. Drop from the middle.
    keep_head = 2 if len(history) > 6 else 0
    head = history[:keep_head]
    tail = history[keep_head:]

    # Walk back from the end, accumulating tail messages until we hit budget.
    acc: List[Dict[str, str]] = []
    running = sys_tokens + est_messages_tokens(head)
    for m in reversed(tail):
        t = est_tokens(m.get("content", "")) + 4
        if running + t > budget:
            break
        acc.append(m)
        running += t
    acc.reverse()

    new_history = head + acc
    # If we dropped messages, drop a leading 'assistant' so the first
    # message after the head/middle gap is a 'user' turn (cleaner for the model).
    while len(new_history) > len(head) and \
            new_history[len(head)].get("role") == "assistant":
        new_history.pop(len(head))

    return new_history, True


# ---------------------------------------------------------------------------
# Request body builder
# ---------------------------------------------------------------------------

def build_payload(model: str, messages: List[Dict[str, str]],
                  sampling: Dict[str, Any], stream: bool = True) -> Dict[str, Any]:
    """
    Build an OpenAI-compatible chat completion payload.

    OpenAI native params:
      temperature, top_p, frequency_penalty, presence_penalty, max_tokens
    Common locally-served extensions (llama.cpp, vLLM, text-generation-webui):
      top_k, min_p, repetition_penalty
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # OpenAI native
    if "temperature" in sampling:
        payload["temperature"] = float(sampling["temperature"])
    if "top_p" in sampling:
        payload["top_p"] = float(sampling["top_p"])
    if "frequency_penalty" in sampling:
        payload["frequency_penalty"] = float(sampling["frequency_penalty"])
    if "presence_penalty" in sampling:
        payload["presence_penalty"] = float(sampling["presence_penalty"])
    if "max_tokens" in sampling:
        payload["max_tokens"] = int(sampling["max_tokens"])

    # Locally-served extensions — sent at the TOP LEVEL so servers like
    # text-generation-webui, llama.cpp server, and vLLM all pick them up.
    if "top_k" in sampling:
        payload["top_k"] = int(sampling["top_k"])
    if "min_p" in sampling:
        payload["min_p"] = float(sampling["min_p"])
    if "repeat_penalty" in sampling:
        # Send under BOTH common names: 'repetition_penalty' (HF / vLLM) and
        # 'repeat_penalty' (llama.cpp). Servers ignore unknown keys.
        payload["repetition_penalty"] = float(sampling["repeat_penalty"])
        payload["repeat_penalty"] = float(sampling["repeat_penalty"])

    return payload


# ---------------------------------------------------------------------------
# Streaming chat completion
# ---------------------------------------------------------------------------

class StreamCancelled(Exception):
    pass


def stream_chat(endpoint: str, api_key: str, payload: Dict[str, Any],
                cancel_event: Optional[threading.Event] = None,
                timeout: float = 120.0
                ) -> Generator[Dict[str, Any], None, None]:
    """
    Yield events:
      {"type": "delta", "text": "..."}
      {"type": "done",  "finish_reason": "stop"}
      {"type": "error", "message": "..."}
    Detection of <think>...</think> reasoning is done by the consumer.
    """
    endpoint = (endpoint or "").rstrip("/")
    url = endpoint + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with requests.post(url, headers=headers, json=payload,
                           stream=True, timeout=timeout) as r:
            if r.status_code >= 400:
                body = r.text[:400]
                yield {"type": "error",
                       "message": f"HTTP {r.status_code}: {body}"}
                return

            for raw in r.iter_lines(decode_unicode=True):
                if cancel_event is not None and cancel_event.is_set():
                    yield {"type": "error", "message": "cancelled"}
                    return
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    yield {"type": "done", "finish_reason": "stop"}
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta") or {}
                # OpenAI-compatible: content lives in delta.content
                piece = delta.get("content")
                # Some servers expose a separate reasoning_content field for
                # Qwen/DeepSeek-style thinking. Surface it as a tagged delta.
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield {"type": "delta", "text": f"<think>{reasoning}</think>"}
                if piece:
                    yield {"type": "delta", "text": piece}
                finish = ch.get("finish_reason")
                if finish:
                    yield {"type": "done", "finish_reason": finish}
                    return
    except requests.RequestException as exc:
        yield {"type": "error",
               "message": f"{exc.__class__.__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Conversation orchestrator
# ---------------------------------------------------------------------------

def run_completion(*, config: Dict[str, Any], user_rec: Dict[str, Any],
                   history: List[Dict[str, str]], system_msg: str,
                   cancel_event: Optional[threading.Event] = None
                   ) -> Generator[Dict[str, Any], None, None]:
    """
    Top-level streaming generator used by the server's SSE endpoint.
    Emits an initial 'meta' event describing context state, then deltas, then 'end'.
    """
    llm_cfg = config.get("llm", {})
    sampling = config.get("sampling", {})

    context_size = int(llm_cfg.get("context_size", 8196))
    threshold = float(llm_cfg.get("compaction_threshold", 0.8))

    new_hist, was_compacted = compact_messages(
        system_msg, history, context_size, threshold=threshold,
        reserve_for_reply=int(sampling.get("max_tokens", 1024)),
    )

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.extend(new_hist)

    used_tokens = est_messages_tokens(messages)
    yield {
        "type": "meta",
        "compacted": was_compacted,
        "tokens_used": used_tokens,
        "context_size": context_size,
        "threshold": threshold,
        "kept_messages": len(new_hist),
        "dropped": max(0, len(history) - len(new_hist)),
    }

    payload = build_payload(
        model=llm_cfg.get("model_name", "local-model"),
        messages=messages,
        sampling=sampling,
        stream=True,
    )

    endpoint = llm_cfg.get("endpoint", "")
    api_key = llm_cfg.get("api_key", "")

    for ev in stream_chat(endpoint, api_key, payload,
                          cancel_event=cancel_event):
        yield ev
