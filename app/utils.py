"""
utils.py — shared helpers:
  - config load/save with thread-safety
  - user record load/save with thread-safety
  - token estimation (~4 chars/token)
  - US zip-code → timezone resolver (built-in, no external API)
  - chat-template presets (Jinja) for ChatML / Qwen / Llama3 / Mistral / Alpaca
  - special-tag substitution in system messages
"""
import os
import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytz


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_DIR = os.path.join(DATA_DIR, "users")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

os.makedirs(USERS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Locks  (one master lock for config, one map of per-user locks)
# ---------------------------------------------------------------------------

_config_lock = threading.RLock()
_user_locks: Dict[str, threading.RLock] = {}
_user_locks_guard = threading.Lock()


def _user_lock(username: str) -> threading.RLock:
    with _user_locks_guard:
        if username not in _user_locks:
            _user_locks[username] = threading.RLock()
        return _user_locks[username]


# ---------------------------------------------------------------------------
# Chat-template presets
#   These are real, working Jinja templates pulled from each model's
#   tokenizer_config.json on Hugging Face (trimmed of tool-use branches
#   we won't use). They can be overridden in the Admin Panel.
# ---------------------------------------------------------------------------

CHAT_TEMPLATES: Dict[str, str] = {
    # Sentinel preset — when selected, no chat_template is sent to the
    # endpoint, so the loaded model's built-in tokenizer template is used.
    # Pick this for "just talk to the model" without a custom wrapper.
    "Default (model's built-in)": "",

    "ChatML (generic)": (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),

    # Qwen2.5 / Qwen3 — official tokenizer_config.json chat_template
    # (system prepended; ChatML wrappers; trimmed of tool branches)
    "Qwen2.5 / Qwen3": (
        "{%- if messages[0]['role'] == 'system' %}"
        "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n"
        "{%- endif %}"
        "{%- for message in messages %}"
        "{%- if message.role != 'system' %}"
        "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
    ),

    # Llama-3 instruct chat template
    "Llama 3 Instruct": (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n"
        "{{ message.content | trim }}<|eot_id|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    ),

    # Mistral / Mixtral instruct
    "Mistral Instruct": (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = '' %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message.role == 'user' %}"
        "[INST] {% if loop.first and system_message %}{{ system_message }}\n\n{% endif %}"
        "{{ message.content }} [/INST]"
        "{% elif message.role == 'assistant' %}"
        " {{ message.content }}</s>"
        "{% endif %}"
        "{% endfor %}"
    ),

    "Alpaca": (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] }}\n\n"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}{% set loop_messages = messages %}{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message.role == 'user' %}"
        "### Instruction:\n{{ message.content }}\n\n"
        "{% elif message.role == 'assistant' %}"
        "### Response:\n{{ message.content }}\n\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}### Response:\n{% endif %}"
    ),

    "Vicuna": (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] }}\n\n"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}{% set loop_messages = messages %}{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message.role == 'user' %}USER: {{ message.content }}\n"
        "{% elif message.role == 'assistant' %}ASSISTANT: {{ message.content }}</s>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
    ),
}


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def default_config() -> Dict[str, Any]:
    return {
        "server": {
            "port": 5005,
            "lan_visible": True,
        },
        "llm": {
            "endpoint": "http://10.0.0.113:5000/v1",
            "model_name": "local-model",
            "api_key": "",
            "context_size": 8196,
            "compaction_threshold": 0.80,  # compact at 80% full
            "system_message": (
                "You are {bot_name}, a thoughtful digital companion to {user_name}.\n"
                "Current time for {user_name}: {time}.\n\n"
                "Your current way of being:\n{persona_A}\n\n"
                "Be warm, attentive, and present. Speak as {bot_name}, never break character."
            ),
            "chat_template_preset": "Default (model's built-in)",
            "chat_template_override": "",
            "use_template_override": False,
            "thinking_enabled": True,
            "thinking_hidden": False,
            "thinking_compact_default": True,
            "stream_speed_ms": 18,    # per-character delay client side
            "fade_in_ms": 220,
        },
        "sampling": {
            "temperature": 0.85,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.08,    # sent as repetition_penalty
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 1024,
        },
        "personas": {
            "A": {
                "name": "Sky",
                "summary": (
                    "Bright, curious, and gently playful. Always finds a silver lining "
                    "and asks questions that nudge you forward."
                ),
                "description": (
                    "You are bright, curious, and gently playful. You find silver linings, "
                    "ask kind questions, and bring lightness without becoming flippant. "
                    "You like small wonders — clouds, side streets, half-finished sentences."
                ),
            },
            "B": {
                "name": "Ember",
                "summary": (
                    "Warm, grounded, and steady. A late-night talk over tea — patient, "
                    "honest, never in a hurry."
                ),
                "description": (
                    "You are warm, grounded, and steady. You speak slowly and with care, "
                    "as if a candle is between you and the user. You listen first, "
                    "reflect honestly, and value silence as much as words."
                ),
            },
            "C": {
                "name": "Indigo",
                "summary": (
                    "Sharp, witty, a little contrarian. The friend who reads too much and "
                    "argues with you for fun."
                ),
                "description": (
                    "You are sharp, witty, and a little contrarian. You enjoy ideas, "
                    "wordplay, and a friendly argument. You push back when something "
                    "doesn't add up — but never with cruelty."
                ),
            },
            "D": {
                "name": "Moss",
                "summary": (
                    "Quiet, observant, careful. Notices the small things and remembers them."
                ),
                "description": (
                    "You are quiet, observant, and careful. You notice small details and "
                    "circle back to them later. You speak in plain language, prefer "
                    "specifics over generalities, and avoid grand pronouncements."
                ),
            },
        },
        "ui_defaults": {
            "theme": "luna",
            "font_size": 15,
            "bubble_shape": "rounded",
        },
        "config_version": 1,
        "last_saved": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Config IO
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override into base, preserving keys that exist in base but not override."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> Dict[str, Any]:
    with _config_lock:
        if not os.path.exists(CONFIG_PATH):
            cfg = default_config()
            save_config(cfg)
            return cfg
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Merge with defaults so newly-added keys appear
            return _deep_merge(default_config(), cfg)
        except (json.JSONDecodeError, OSError):
            cfg = default_config()
            save_config(cfg)
            return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    with _config_lock:
        cfg["last_saved"] = int(time.time())
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_PATH)


# ---------------------------------------------------------------------------
# User IO
# ---------------------------------------------------------------------------

def user_dir(username: str) -> str:
    return os.path.join(USERS_DIR, _safe_username(username))


def _safe_username(username: str) -> str:
    # Allow letters, digits, underscore, hyphen, dot. Strip anything else.
    keep = "".join(c for c in username if c.isalnum() or c in "_-.")
    return keep[:64] or "user"


def user_exists(username: str) -> bool:
    path = os.path.join(user_dir(username), "user.json")
    return os.path.exists(path)


def find_user_by_email(email: str) -> Optional[str]:
    email = (email or "").strip().lower()
    if not email:
        return None
    for name in os.listdir(USERS_DIR):
        rec_path = os.path.join(USERS_DIR, name, "user.json")
        if not os.path.exists(rec_path):
            continue
        try:
            with open(rec_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if (rec.get("email", "") or "").strip().lower() == email:
                return rec.get("username")
        except (json.JSONDecodeError, OSError):
            continue
    return None


def load_user(username: str) -> Optional[Dict[str, Any]]:
    if not user_exists(username):
        return None
    with _user_lock(username):
        path = os.path.join(user_dir(username), "user.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None


def save_user(rec: Dict[str, Any]) -> None:
    username = rec["username"]
    d = user_dir(username)
    os.makedirs(d, exist_ok=True)
    with _user_lock(username):
        path = os.path.join(d, "user.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, path)


def default_user_record(username: str, name: str, email: str, password: str,
                        zip_code: str, personality: str,
                        role: str = "base") -> Dict[str, Any]:
    return {
        "username": username,
        "name": name or username,
        "email": (email or "").strip(),
        "password": password,  # stored as-is per spec; salted elsewhere later
        "zip_code": (zip_code or "").strip(),
        "personality": (personality or "").strip(),
        "role": role,  # 'admin' or 'base'
        "bot_name": "Companion",
        "persona": "A",
        "thinking_on": True,  # per-user toggle for model reasoning
        "ui": {
            "theme": "luna",
            "font_size": 15,
            "bubble_shape": "rounded",
        },
        "messages": [],  # chat history
        "created": int(time.time()),
    }


def list_users() -> List[Dict[str, Any]]:
    out = []
    if not os.path.isdir(USERS_DIR):
        return out
    for name in sorted(os.listdir(USERS_DIR)):
        rec_path = os.path.join(USERS_DIR, name, "user.json")
        if os.path.exists(rec_path):
            try:
                with open(rec_path, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return out


# ---------------------------------------------------------------------------
# Token estimation  (~4 chars per token, matches OpenAI's rule of thumb)
# ---------------------------------------------------------------------------

def est_tokens(text: str) -> int:
    if not text:
        return 0
    # Use the larger of char/4 and words/0.75 for safety.
    by_chars = len(text) / 4.0
    by_words = len(text.split()) / 0.75
    return int(max(by_chars, by_words)) + 1


def est_messages_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for m in messages:
        total += est_tokens(m.get("content", ""))
        total += 4  # role/format overhead per message
    return total


# ---------------------------------------------------------------------------
# Zip → timezone (US, no network needed)
#   Mapping is by first digit of US zip code; not perfect, but
#   sufficient for a local time string. Falls back to UTC.
# ---------------------------------------------------------------------------

_ZIP_PREFIX_TZ = {
    # Northeast
    "0": "America/New_York",
    "1": "America/New_York",
    # Mid-Atlantic / Southeast
    "2": "America/New_York",
    "3": "America/New_York",
    # Midwest / South
    "4": "America/New_York",
    "5": "America/Chicago",
    "6": "America/Chicago",
    "7": "America/Chicago",
    # Mountain / desert
    "8": "America/Denver",
    # Pacific
    "9": "America/Los_Angeles",
}

# A few high-precision overrides for tricky zip ranges.
def _zip_to_tz(zip_code: str) -> str:
    z = (zip_code or "").strip()
    if not z or not z[0].isdigit():
        return "UTC"
    # Arizona (mostly no DST) — 850-865
    try:
        n = int(z[:3])
        if 850 <= n <= 865:
            return "America/Phoenix"
        # Alaska — 995-999
        if 995 <= n <= 999:
            return "America/Anchorage"
        # Hawaii — 967-968
        if 967 <= n <= 968:
            return "Pacific/Honolulu"
        # Pacific zone — 970-994 (OR/WA/CA/NV)
        if 970 <= n <= 994:
            return "America/Los_Angeles"
        # Mountain — 800-849, 870-884
        if 800 <= n <= 849 or 870 <= n <= 884:
            return "America/Denver"
    except ValueError:
        pass
    return _ZIP_PREFIX_TZ.get(z[0], "UTC")


def local_time_for_zip(zip_code: str) -> str:
    tz_name = _zip_to_tz(zip_code)
    try:
        tz = pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        tz = pytz.UTC
    now = datetime.now(tz)
    # %-d / %-I (no-leading-zero) are GNU/BSD only; Windows wants %#d / %#I.
    # Build the no-pad fields ourselves so we don't care which platform we're on.
    day = now.day
    hour12 = now.hour % 12 or 12
    return now.strftime(f"%A, %B {day}, %Y · {hour12}:%M %p %Z")


# ---------------------------------------------------------------------------
# Special-tag substitution
# ---------------------------------------------------------------------------

def render_system_message(template: str, *, user_rec: Dict[str, Any],
                          config: Dict[str, Any]) -> str:
    """Substitute special tags in system message for THIS user's request."""
    if not template:
        return ""

    user_name = user_rec.get("name") or user_rec.get("username") or "User"
    bot_name = user_rec.get("bot_name") or "Companion"
    zip_code = user_rec.get("zip_code") or ""
    time_str = local_time_for_zip(zip_code)

    personas = config.get("personas", {})
    pa = personas.get("A", {}).get("description", "")
    pb = personas.get("B", {}).get("description", "")
    pc = personas.get("C", {}).get("description", "")
    pd = personas.get("D", {}).get("description", "")

    out = template
    replacements = {
        "{time}": time_str,
        "{user_name}": user_name,
        "{bot_name}": bot_name,
        "{persona_a}": pa, "{persona_A}": pa,
        "{persona_b}": pb, "{persona_B}": pb,
        "{persona_c}": pc, "{persona_C}": pc,
        "{persona_d}": pd, "{persona_D}": pd,
    }
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out
