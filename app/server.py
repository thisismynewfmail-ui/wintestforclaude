"""
server.py — Flask application.

Routes
------
GET  /                       redirect to /chat or /login
GET  /login                  login + signup page
POST /api/login              authenticate, set session
POST /api/signup             create user, set session
POST /api/logout             clear session
GET  /chat                   main chat UI (auth required)
GET  /admincontrols          admin panel (admin role required)
GET  /api/me                 current user record
POST /api/user/settings      user-side settings (bot name, persona, theme...)
POST /api/user/reset         reset chat history
GET  /api/user/history       fetch chat history
POST /api/chat/stream        SSE stream of an assistant reply

GET  /api/admin/config       full config (admin)
POST /api/admin/config       save full config (admin)
POST /api/admin/test         endpoint reachability test
GET  /api/admin/templates    list of preset template names
GET  /api/admin/users        user roster (admin)
POST /api/admin/user_role    change a user's role (admin)
"""
import json
import os
import secrets
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from flask import (
    Flask, Response, jsonify, redirect, render_template, request, session,
    stream_with_context, url_for, abort
)

from . import utils
from . import llm


_config_cache: Dict[str, Any] = {}
_config_cache_lock = threading.RLock()


def get_config() -> Dict[str, Any]:
    """Return a fresh-enough copy of config. Reads disk if cache empty."""
    with _config_cache_lock:
        if not _config_cache:
            _config_cache.update(utils.load_config())
        return json.loads(json.dumps(_config_cache))  # deep copy


def set_config(cfg: Dict[str, Any]) -> None:
    """Persist + update cache atomically. Applies live to all future requests."""
    with _config_cache_lock:
        utils.save_config(cfg)
        _config_cache.clear()
        _config_cache.update(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_user() -> Optional[Dict[str, Any]]:
    uname = session.get("username")
    if not uname:
        return None
    return utils.load_user(uname)


def _require_admin() -> Optional[Dict[str, Any]]:
    rec = _require_user()
    if rec and rec.get("role") == "admin":
        return rec
    return None


def _json_error(msg: str, status: int = 400):
    return jsonify({"ok": False, "error": msg}), status


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(base_dir: str) -> Flask:
    template_dir = os.path.join(base_dir, "templates")
    static_dir = os.path.join(base_dir, "static")
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

    # Persistent session key so logins survive restarts.
    secret_path = os.path.join(utils.DATA_DIR, ".session_secret")
    if os.path.exists(secret_path):
        with open(secret_path, "rb") as f:
            app.secret_key = f.read()
    else:
        key = secrets.token_bytes(32)
        with open(secret_path, "wb") as f:
            f.write(key)
        app.secret_key = key

    # Warm config cache and bootstrap the first user as admin.
    with _config_cache_lock:
        _config_cache.clear()
        _config_cache.update(utils.load_config())

    # ----------------------- Pages ----------------------------------------

    @app.route("/")
    def root():
        if session.get("username"):
            return redirect(url_for("chat"))
        return redirect(url_for("login"))

    @app.route("/login")
    def login():
        if session.get("username"):
            return redirect(url_for("chat"))
        return render_template("login.html")

    @app.route("/chat")
    def chat():
        rec = _require_user()
        if not rec:
            return redirect(url_for("login"))
        cfg = get_config()
        return render_template(
            "chat.html",
            user=rec,
            personas=cfg.get("personas", {}),
            ui_defaults=cfg.get("ui_defaults", {}),
            thinking_enabled=cfg.get("llm", {}).get("thinking_enabled", True),
            thinking_hidden=cfg.get("llm", {}).get("thinking_hidden", False),
            thinking_compact_default=cfg.get("llm", {}).get("thinking_compact_default", True),
            stream_speed_ms=cfg.get("llm", {}).get("stream_speed_ms", 18),
            fade_in_ms=cfg.get("llm", {}).get("fade_in_ms", 220),
        )

    @app.route("/admincontrols")
    def admincontrols():
        rec = _require_user()
        if not rec:
            return redirect(url_for("login"))
        if rec.get("role") != "admin":
            return render_template("forbidden.html"), 403
        cfg = get_config()
        return render_template(
            "admin.html",
            user=rec,
            config=cfg,
            template_presets=list(utils.CHAT_TEMPLATES.keys()),
        )

    # ----------------------- Auth API -------------------------------------

    @app.post("/api/signup")
    def api_signup():
        d = request.get_json(silent=True) or {}
        username = (d.get("username") or "").strip()
        username2 = (d.get("username2") or "").strip()
        name = (d.get("name") or "").strip()
        email = (d.get("email") or "").strip()
        password = d.get("password") or ""
        password2 = d.get("password2") or ""
        zip_code = (d.get("zip_code") or "").strip()
        personality = (d.get("personality") or "").strip()

        # All fields required.
        if not all([username, username2, name, email, password, password2,
                    zip_code, personality]):
            return _json_error("All fields are required.")
        if username != username2:
            return _json_error("Usernames do not match.")
        if password != password2:
            return _json_error("Passwords do not match.")
        if len(username) < 2 or len(username) > 32:
            return _json_error("Username must be 2–32 characters.")
        if utils.user_exists(username):
            return _json_error("That username is taken.")
        if utils.find_user_by_email(email):
            return _json_error("That email is already registered.")
        if "@" not in email or "." not in email:
            return _json_error("Email looks invalid.")

        # First account becomes admin automatically.
        role = "base"
        if not utils.list_users():
            role = "admin"

        rec = utils.default_user_record(
            username=username, name=name, email=email, password=password,
            zip_code=zip_code, personality=personality, role=role,
        )
        utils.save_user(rec)
        session["username"] = rec["username"]
        return jsonify({"ok": True, "user": _public_user(rec),
                        "first_admin": role == "admin"})

    @app.post("/api/login")
    def api_login():
        d = request.get_json(silent=True) or {}
        ident = (d.get("identifier") or "").strip()
        password = d.get("password") or ""
        if not ident or not password:
            return _json_error("Username/email and password are required.")

        username = ident
        if not utils.user_exists(username):
            # Try email lookup
            found = utils.find_user_by_email(ident)
            if found:
                username = found
            else:
                return _json_error("No account found.")

        rec = utils.load_user(username)
        if not rec:
            return _json_error("No account found.")
        if rec.get("password") != password:
            return _json_error("Incorrect password.")

        session["username"] = rec["username"]
        return jsonify({"ok": True, "user": _public_user(rec)})

    @app.post("/api/logout")
    def api_logout():
        session.clear()
        return jsonify({"ok": True})

    # ----------------------- User API -------------------------------------

    @app.get("/api/me")
    def api_me():
        rec = _require_user()
        if not rec:
            return _json_error("Not logged in.", 401)
        return jsonify({"ok": True, "user": _public_user(rec)})

    @app.post("/api/user/settings")
    def api_user_settings():
        rec = _require_user()
        if not rec:
            return _json_error("Not logged in.", 401)
        d = request.get_json(silent=True) or {}

        allowed_fields = ("bot_name", "persona", "zip_code")
        for f in allowed_fields:
            if f in d:
                val = (d[f] or "").strip() if isinstance(d[f], str) else d[f]
                if f == "persona" and val not in ("A", "B", "C", "D"):
                    continue
                if f == "bot_name" and val and len(val) > 40:
                    val = val[:40]
                rec[f] = val

        ui = d.get("ui") or {}
        if isinstance(ui, dict):
            allowed_ui = ("theme", "font_size", "bubble_shape")
            for k in allowed_ui:
                if k in ui:
                    rec.setdefault("ui", {})[k] = ui[k]

        utils.save_user(rec)
        return jsonify({"ok": True, "user": _public_user(rec)})

    @app.post("/api/user/reset")
    def api_user_reset():
        rec = _require_user()
        if not rec:
            return _json_error("Not logged in.", 401)
        rec["messages"] = []
        utils.save_user(rec)
        return jsonify({"ok": True})

    @app.get("/api/user/history")
    def api_user_history():
        rec = _require_user()
        if not rec:
            return _json_error("Not logged in.", 401)
        return jsonify({"ok": True, "messages": rec.get("messages", [])})

    # ----------------------- Chat stream ----------------------------------

    @app.post("/api/chat/stream")
    def api_chat_stream():
        rec = _require_user()
        if not rec:
            return _json_error("Not logged in.", 401)
        d = request.get_json(silent=True) or {}
        user_msg = (d.get("message") or "").strip()
        if not user_msg:
            return _json_error("Message is empty.")

        cfg = get_config()
        llm_cfg = cfg.get("llm", {})

        # Render the system message NOW, with this specific user's data.
        system_msg = utils.render_system_message(
            llm_cfg.get("system_message", ""),
            user_rec=rec, config=cfg,
        )

        # Append the user message to this user's history immediately.
        send_lock = llm._user_send_lock(rec["username"])
        message_id = uuid.uuid4().hex

        def gen():
            # Per-user serialization (one send at a time per user).
            with send_lock:
                # Refresh record under lock in case another request edited it.
                rec_l = utils.load_user(rec["username"]) or rec
                history = list(rec_l.get("messages", []))
                history.append({"role": "user", "content": user_msg,
                                "ts": int(time.time())})
                rec_l["messages"] = history
                utils.save_user(rec_l)

                yield _sse({"type": "user_saved", "id": message_id,
                            "ts": history[-1]["ts"]})

                accumulated = ""
                meta_sent = False
                try:
                    for ev in llm.run_completion(
                        config=cfg, user_rec=rec_l, history=history,
                        system_msg=system_msg,
                    ):
                        t = ev.get("type")
                        if t == "meta":
                            meta_sent = True
                            yield _sse(ev)
                        elif t == "delta":
                            accumulated += ev.get("text", "")
                            yield _sse({"type": "delta", "text": ev.get("text", "")})
                        elif t == "done":
                            yield _sse(ev)
                            break
                        elif t == "error":
                            yield _sse(ev)
                            break
                except GeneratorExit:
                    # Client disconnected; persist whatever we have.
                    pass

                # Persist assistant reply (even partial) and clean think tags
                # out of the stored copy so they don't compound context.
                if accumulated:
                    rec_l = utils.load_user(rec_l["username"]) or rec_l
                    history = list(rec_l.get("messages", []))
                    history.append({"role": "assistant",
                                    "content": accumulated,
                                    "ts": int(time.time())})
                    rec_l["messages"] = history
                    utils.save_user(rec_l)

                yield _sse({"type": "end"})

        return Response(stream_with_context(gen()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    # ----------------------- Admin API ------------------------------------

    @app.get("/api/admin/config")
    def api_admin_config():
        if not _require_admin():
            return _json_error("Admin only.", 403)
        cfg = get_config()
        # Don't ship the API key plaintext on every page render
        return jsonify({"ok": True, "config": cfg,
                        "template_presets": list(utils.CHAT_TEMPLATES.keys()),
                        "template_bodies": utils.CHAT_TEMPLATES})

    @app.post("/api/admin/config")
    def api_admin_config_save():
        if not _require_admin():
            return _json_error("Admin only.", 403)
        d = request.get_json(silent=True) or {}
        new_cfg = d.get("config") or {}

        # Sanity: clamp numeric values
        try:
            llm_cfg = new_cfg.setdefault("llm", {})
            llm_cfg["context_size"] = max(512, int(llm_cfg.get("context_size", 8196)))
            llm_cfg["compaction_threshold"] = min(0.99, max(0.1, float(
                llm_cfg.get("compaction_threshold", 0.8))))
            llm_cfg["stream_speed_ms"] = max(0, int(llm_cfg.get("stream_speed_ms", 18)))
            llm_cfg["fade_in_ms"] = max(0, int(llm_cfg.get("fade_in_ms", 220)))

            s = new_cfg.setdefault("sampling", {})
            s["temperature"] = float(s.get("temperature", 0.85))
            s["top_p"] = float(s.get("top_p", 0.95))
            s["top_k"] = int(s.get("top_k", 40))
            s["min_p"] = float(s.get("min_p", 0.05))
            s["repeat_penalty"] = float(s.get("repeat_penalty", 1.08))
            s["frequency_penalty"] = float(s.get("frequency_penalty", 0.0))
            s["presence_penalty"] = float(s.get("presence_penalty", 0.0))
            s["max_tokens"] = int(s.get("max_tokens", 1024))

            srv = new_cfg.setdefault("server", {})
            srv["port"] = int(srv.get("port", 5005))
            srv["lan_visible"] = bool(srv.get("lan_visible", True))
        except (ValueError, TypeError) as exc:
            return _json_error(f"Invalid number: {exc}")

        set_config(new_cfg)
        return jsonify({"ok": True, "config": get_config()})

    @app.post("/api/admin/test")
    def api_admin_test():
        if not _require_admin():
            return _json_error("Admin only.", 403)
        d = request.get_json(silent=True) or {}
        endpoint = d.get("endpoint") or get_config().get("llm", {}).get("endpoint", "")
        api_key = d.get("api_key") or get_config().get("llm", {}).get("api_key", "")
        ok, msg, models = llm.test_endpoint(endpoint, api_key)
        return jsonify({"ok": ok, "message": msg, "models": models})

    @app.get("/api/admin/templates")
    def api_admin_templates():
        if not _require_admin():
            return _json_error("Admin only.", 403)
        return jsonify({"ok": True, "presets": utils.CHAT_TEMPLATES})

    @app.get("/api/admin/users")
    def api_admin_users():
        if not _require_admin():
            return _json_error("Admin only.", 403)
        out = []
        for u in utils.list_users():
            out.append({
                "username": u.get("username"),
                "name": u.get("name"),
                "email": u.get("email"),
                "role": u.get("role", "base"),
                "created": u.get("created"),
                "messages": len(u.get("messages", [])),
            })
        return jsonify({"ok": True, "users": out})

    @app.post("/api/admin/user_role")
    def api_admin_user_role():
        admin = _require_admin()
        if not admin:
            return _json_error("Admin only.", 403)
        d = request.get_json(silent=True) or {}
        username = d.get("username")
        role = d.get("role")
        if role not in ("admin", "base"):
            return _json_error("Role must be 'admin' or 'base'.")
        rec = utils.load_user(username)
        if not rec:
            return _json_error("No such user.")
        # Prevent demoting the last admin
        if role == "base":
            admins = [u for u in utils.list_users() if u.get("role") == "admin"]
            if len(admins) <= 1 and rec.get("role") == "admin":
                return _json_error("Cannot demote the only admin.")
        rec["role"] = role
        utils.save_user(rec)
        return jsonify({"ok": True})

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _public_user(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Strip password before sending to client."""
    safe = dict(rec)
    safe.pop("password", None)
    return safe


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
