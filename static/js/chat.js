// chat.js — chat client. Streams assistant replies via SSE, renders
// thinking blocks, manages the settings drawer.
(function () {
    "use strict";

    const APP = window.__APP__ || {};
    const body = document.body;
    const chatBody = document.getElementById("chat-body");
    const composer = document.getElementById("composer");
    const input = document.getElementById("msg-input");
    const sendBtn = document.getElementById("send-btn");
    const ctxFill = document.getElementById("ctx-fill");
    const ctxText = document.getElementById("ctx-text");
    const compactPill = document.getElementById("compact-pill");
    const connPill = document.getElementById("conn-pill");
    const connText = document.getElementById("conn-text");
    const titleLabel = document.getElementById("title-label");
    const toast = document.getElementById("toast");

    // Set fade-in CSS var
    document.documentElement.style.setProperty("--fade-in-ms", APP.fade_in_ms + "ms");

    function showToast(msg, kind) {
        toast.textContent = msg;
        toast.className = "toast show " + (kind || "");
        setTimeout(() => toast.classList.remove("show"), 2400);
    }

    function escapeHTML(s) {
        return (s || "").replace(/[&<>"']/g, (c) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;",
            "\"": "&quot;", "'": "&#39;",
        })[c]);
    }

    function formatTime(ts) {
        if (!ts) return "";
        const d = new Date(ts * 1000);
        const hh = d.getHours();
        const mm = String(d.getMinutes()).padStart(2, "0");
        const ampm = hh >= 12 ? "PM" : "AM";
        const h12 = ((hh + 11) % 12) + 1;
        return `${h12}:${mm} ${ampm}`;
    }

    // ---- Message rendering -------------------------------------------------

    function makeMsg(role, content, ts) {
        const wrap = document.createElement("div");
        wrap.className = "msg " + role;
        const who = role === "user" ? APP.user.name :
            (APP.user.bot_name || "Companion");
        wrap.innerHTML = `
            <div class="avatar"></div>
            <div class="bubble">
                <div class="who">${escapeHTML(who)}</div>
                <div class="content"></div>
                <div class="time">${ts ? formatTime(ts) : ""}</div>
            </div>`;
        const contentEl = wrap.querySelector(".content");
        renderContent(contentEl, content || "");
        return wrap;
    }

    /**
     * Render text into the content element, handling <think>…</think> blocks.
     * Stored history may contain think tags; live streaming arrives as raw text
     * which is parsed incrementally by the streaming code below.
     */
    function renderContent(el, text) {
        el.innerHTML = "";
        const parts = splitThinking(text);
        for (const part of parts) {
            if (part.kind === "think") {
                const thinkBlock = makeThinkBlock(part.text,
                    APP.thinking_compact_default ? false : true);
                el.appendChild(thinkBlock);
            } else {
                el.appendChild(document.createTextNode(part.text));
            }
        }
    }

    function splitThinking(text) {
        // Split into alternating think / non-think segments.
        const out = [];
        const re = /<think>([\s\S]*?)<\/think>/g;
        let last = 0;
        let m;
        while ((m = re.exec(text)) !== null) {
            if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
            out.push({ kind: "think", text: m[1] });
            last = m.index + m[0].length;
        }
        if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
        return out;
    }

    function makeThinkBlock(text, openByDefault) {
        const wrap = document.createElement("div");
        wrap.className = "think" + (openByDefault ? " open" : "");
        const header = document.createElement("div");
        header.className = "think-header";
        header.textContent = "Thinking";
        const body = document.createElement("div");
        body.className = "think-body";
        body.textContent = text;
        wrap.appendChild(header);
        wrap.appendChild(body);
        header.addEventListener("click", () => wrap.classList.toggle("open"));
        return wrap;
    }

    function scrollToBottom(force) {
        const nearBottom = chatBody.scrollHeight - chatBody.scrollTop -
            chatBody.clientHeight < 80;
        if (force || nearBottom) {
            chatBody.scrollTop = chatBody.scrollHeight;
        }
    }

    function fadeAppend(target, text) {
        // Append text as one fade-in span per chunk (preserves whitespace).
        if (!text) return;
        const span = document.createElement("span");
        span.className = "fade-chunk";
        span.textContent = text;
        target.appendChild(span);
    }

    // ---- Context bar -------------------------------------------------------

    let contextSize = 8196;

    function updateContext(used, total, threshold) {
        contextSize = total || contextSize;
        const pct = Math.min(100, Math.round((used / total) * 100));
        ctxFill.style.width = pct + "%";
        ctxText.textContent = `${used} / ${total} tk · ${pct}%`;
        ctxFill.style.filter = pct > (threshold * 100) ? "saturate(1.3)" : "";
    }

    // ---- History load ------------------------------------------------------

    async function loadHistory() {
        try {
            const r = await fetch("/api/user/history", { credentials: "same-origin" });
            const d = await r.json();
            if (d.ok) {
                chatBody.innerHTML = "";
                for (const m of d.messages) {
                    chatBody.appendChild(makeMsg(m.role, m.content, m.ts));
                }
                if (d.messages.length === 0) {
                    welcomeMessage();
                }
                scrollToBottom(true);
            }
        } catch (e) { /* noop */ }
    }

    function welcomeMessage() {
        const greeting = `Hello, ${APP.user.name}. I'm ${APP.user.bot_name || "your companion"}. ` +
            `Settle in — when you're ready, tell me how you're doing.`;
        const m = makeMsg("assistant", greeting, Math.floor(Date.now() / 1000));
        chatBody.appendChild(m);
    }

    // ---- Sending / streaming -----------------------------------------------

    let sending = false;
    let currentAssistantEl = null;
    let currentContentEl = null;
    let liveBuffer = "";   // raw text including <think> tags

    // Incremental streaming parser state:
    //   mode: "text" | "think" | "tag"
    //   tagBuf: partial tag like "<thi"
    //   textTarget: the DOM node currently receiving plain text chunks
    //   thinkTarget: the .think-body node receiving thinking text
    //   thinkWrap: the .think wrapper element
    let streamState = null;

    async function send(text) {
        if (sending || !text.trim()) return;
        sending = true;
        sendBtn.disabled = true;

        // Append user message immediately.
        const userMsg = makeMsg("user", text, Math.floor(Date.now() / 1000));
        chatBody.appendChild(userMsg);
        scrollToBottom(true);

        // Prepare assistant placeholder.
        currentAssistantEl = makeMsg("assistant", "", null);
        const bubble = currentAssistantEl.querySelector(".content");
        bubble.classList.add("streaming");
        currentContentEl = bubble;
        liveBuffer = "";
        streamState = {
            mode: "text",
            tagBuf: "",
            contentEl: bubble,
            thinkWrap: null,
            thinkBody: null,
        };
        chatBody.appendChild(currentAssistantEl);
        scrollToBottom(true);

        try {
            const r = await fetch("/api/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
                body: JSON.stringify({ message: text }),
                credentials: "same-origin",
            });
            if (!r.ok) {
                showToast("Server rejected the request (HTTP " + r.status + ")", "bad");
                bubble.classList.remove("streaming");
                bubble.textContent = "(failed to start)";
                sending = false; sendBtn.disabled = false;
                return;
            }
            await consumeSSE(r);
        } catch (e) {
            showToast("Connection error: " + e.message, "bad");
            bubble.classList.remove("streaming");
        } finally {
            sending = false;
            sendBtn.disabled = false;
            if (currentContentEl) {
                currentContentEl.classList.remove("streaming");
                // Finalize: tag the time
                const timeEl = currentAssistantEl.querySelector(".time");
                if (timeEl) timeEl.textContent = formatTime(Math.floor(Date.now() / 1000));
            }
            currentAssistantEl = null;
            currentContentEl = null;
        }
    }

    async function consumeSSE(response) {
        const reader = response.body.getReader();
        const dec = new TextDecoder("utf-8");
        let buf = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });

            // SSE: events separated by \n\n
            let idx;
            while ((idx = buf.indexOf("\n\n")) !== -1) {
                const ev = buf.slice(0, idx);
                buf = buf.slice(idx + 2);
                handleSSEEvent(ev);
            }
        }
        // flush
        if (buf.trim()) handleSSEEvent(buf);
    }

    function handleSSEEvent(raw) {
        const line = raw.split("\n").find((l) => l.startsWith("data:"));
        if (!line) return;
        let obj;
        try { obj = JSON.parse(line.slice(5).trim()); }
        catch (e) { return; }

        if (obj.type === "meta") {
            updateContext(obj.tokens_used, obj.context_size, obj.threshold || 0.8);
            if (obj.compacted) {
                compactPill.style.display = "";
                showToast("Memory tidied (older turns trimmed)", "good");
                setTimeout(() => { compactPill.style.display = "none"; }, 4000);
            }
        } else if (obj.type === "delta") {
            appendStreamingText(obj.text || "");
            scrollToBottom();
        } else if (obj.type === "done" || obj.type === "end") {
            // stream completed
        } else if (obj.type === "error") {
            showToast("LLM error: " + obj.message, "bad");
            if (currentContentEl) {
                const err = document.createElement("div");
                err.style.color = "#871818";
                err.style.fontStyle = "italic";
                err.style.fontSize = "13px";
                err.textContent = "[error] " + obj.message;
                currentContentEl.appendChild(err);
            }
        } else if (obj.type === "user_saved") {
            // server confirmed our user message was persisted; nothing to do
        }
    }

    function appendStreamingText(chunk) {
        liveBuffer += chunk;
        feedStream(streamState, chunk);
    }

    /**
     * Incremental SSE-text-to-DOM parser.
     * Detects <think>...</think> across chunk boundaries, opens/closes think
     * blocks, and appends new text as a single fade-in span per chunk.
     * Already-rendered DOM is never rewritten, so fade-in only runs once.
     */
    function feedStream(state, chunk) {
        let i = 0;
        let pending = "";   // characters not yet flushed for the current mode

        const flushPending = () => {
            if (!pending) return;
            if (state.mode === "text") {
                appendFadedText(state.contentEl, pending);
            } else if (state.mode === "think") {
                if (APP.thinking_enabled) {
                    if (!state.thinkBody) {
                        const wrap = makeThinkBlock("", !APP.thinking_compact_default);
                        state.thinkWrap = wrap;
                        state.thinkBody = wrap.querySelector(".think-body");
                        state.contentEl.appendChild(wrap);
                    }
                    appendFadedText(state.thinkBody, pending);
                }
                // If thinking is disabled at the LLM/UI level we drop the text.
            }
            pending = "";
        };

        while (i < chunk.length) {
            const ch = chunk[i];

            if (state.mode === "tag") {
                // Accumulating either an opening or closing tag.
                state.tagBuf += ch;
                if (state.tagBuf === "<think>") {
                    state.mode = "think";
                    state.tagBuf = "";
                } else if (state.tagBuf === "</think>") {
                    state.mode = "text";
                    state.tagBuf = "";
                    state.thinkWrap = null;
                    state.thinkBody = null;
                } else if (!"<think>".startsWith(state.tagBuf) &&
                           !"</think>".startsWith(state.tagBuf)) {
                    // Not actually a think tag — treat the accumulated chars as text.
                    pending += state.tagBuf;
                    state.tagBuf = "";
                    state.mode = (state.thinkBody ? "think" : "text");
                }
                i++;
                continue;
            }

            if (ch === "<") {
                flushPending();
                state.mode = "tag";
                state.tagBuf = "<";
                i++;
                continue;
            }

            pending += ch;
            i++;
        }
        flushPending();
        scrollToBottom();
    }

    function appendFadedText(target, text) {
        if (!text) return;
        const span = document.createElement("span");
        span.className = "fade-chunk";
        span.textContent = text;
        target.appendChild(span);
    }

    // Composer submission
    composer.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = input.value;
        if (!text.trim()) return;
        input.value = "";
        autoGrow();
        send(text);
    });

    // Enter to send, Shift+Enter for newline
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            composer.dispatchEvent(new Event("submit", { cancelable: true }));
        }
    });

    function autoGrow() {
        input.style.height = "auto";
        input.style.height = Math.min(200, input.scrollHeight) + "px";
    }
    input.addEventListener("input", autoGrow);

    // ---- Settings drawer ---------------------------------------------------

    const drawer = document.getElementById("drawer");
    const scrim = document.getElementById("drawer-scrim");

    function openDrawer() {
        drawer.classList.add("open");
        scrim.classList.add("open");
        drawer.setAttribute("aria-hidden", "false");
    }
    function closeDrawer() {
        drawer.classList.remove("open");
        scrim.classList.remove("open");
        drawer.setAttribute("aria-hidden", "true");
    }
    document.getElementById("open-settings").addEventListener("click", openDrawer);
    document.getElementById("close-drawer").addEventListener("click", closeDrawer);
    scrim.addEventListener("click", closeDrawer);

    // Persona pick
    document.querySelectorAll(".persona-card").forEach((card) => {
        card.addEventListener("click", async () => {
            document.querySelectorAll(".persona-card").forEach((c) => c.classList.remove("selected"));
            card.classList.add("selected");
            const key = card.dataset.key;
            await saveSettings({ persona: key });
        });
    });

    // Theme pick
    document.querySelectorAll(".theme-chip").forEach((chip) => {
        chip.addEventListener("click", async () => {
            document.querySelectorAll(".theme-chip").forEach((c) => c.classList.remove("selected"));
            chip.classList.add("selected");
            const theme = chip.dataset.theme;
            // Swap body class
            body.className = body.className.replace(/theme-\S+/g, "").trim();
            body.classList.add("theme-" + theme);
            await saveSettings({ ui: { theme } });
        });
    });

    // Bubble shape
    document.getElementById("set-bubble-shape").addEventListener("change", async (e) => {
        const shape = e.target.value;
        body.className = body.className.replace(/bubble-shape-\S+/g, "").trim();
        body.classList.add("bubble-shape-" + shape);
        await saveSettings({ ui: { bubble_shape: shape } });
    });

    // Font size
    const fsRange = document.getElementById("set-font-size");
    const fsVal = document.getElementById("font-size-val");
    fsRange.addEventListener("input", (e) => {
        const v = e.target.value;
        fsVal.textContent = v + "px";
        body.className = body.className.replace(/fs-\d+/g, "").trim();
        body.classList.add("fs-" + v);
    });
    fsRange.addEventListener("change", async (e) => {
        await saveSettings({ ui: { font_size: parseInt(e.target.value, 10) } });
    });

    // Bot name (live update)
    const botInput = document.getElementById("set-bot-name");
    let botSaveTimer = null;
    botInput.addEventListener("input", () => {
        clearTimeout(botSaveTimer);
        botSaveTimer = setTimeout(async () => {
            const v = botInput.value.trim() || "Companion";
            APP.user.bot_name = v;
            titleLabel.textContent = `${v} — chatting with ${APP.user.name}`;
            await saveSettings({ bot_name: v });
        }, 600);
    });

    // Zip
    const zipInput = document.getElementById("set-zip");
    let zipSaveTimer = null;
    zipInput.addEventListener("input", () => {
        clearTimeout(zipSaveTimer);
        zipSaveTimer = setTimeout(async () => {
            const v = zipInput.value.trim();
            await saveSettings({ zip_code: v });
        }, 700);
    });

    // Reset mind
    document.getElementById("reset-mind").addEventListener("click", async () => {
        if (!confirm("Clear all chat history with your companion? This cannot be undone.")) return;
        const r = await fetch("/api/user/reset", { method: "POST", credentials: "same-origin" });
        const d = await r.json();
        if (d.ok) {
            chatBody.innerHTML = "";
            welcomeMessage();
            showToast("Memory cleared.", "good");
            updateContext(0, contextSize, 0.8);
        } else {
            showToast("Could not reset.", "bad");
        }
    });

    // Sign out
    async function signOut() {
        await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
        window.location.href = "/login";
    }
    document.getElementById("logout-btn").addEventListener("click", signOut);
    document.getElementById("signout-link").addEventListener("click", signOut);

    async function saveSettings(patch) {
        try {
            const r = await fetch("/api/user/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(patch),
                credentials: "same-origin",
            });
            const d = await r.json();
            if (d.ok) {
                Object.assign(APP.user, d.user);
            } else {
                showToast("Settings not saved.", "bad");
            }
        } catch (e) {
            showToast("Settings save failed.", "bad");
        }
    }

    // ---- Connection probe (light) -----------------------------------------
    // We don't have a public test endpoint for non-admins, so we just
    // mark "ready" once history loads.
    function setConnected(ok, label) {
        connPill.querySelector(".dot").style.background = ok ? "var(--moss)" : "#c23b3b";
        connText.textContent = label;
    }

    // ---- Boot --------------------------------------------------------------
    loadHistory().then(() => setConnected(true, "ready"));

    // Periodically refresh context bar based on current history length.
    async function pollContext() {
        try {
            const r = await fetch("/api/user/history", { credentials: "same-origin" });
            const d = await r.json();
            if (d.ok) {
                // Estimate tokens client-side as the server would.
                let total = 0;
                for (const m of d.messages) {
                    const t = (m.content || "");
                    total += Math.max(Math.ceil(t.length / 4), Math.ceil(t.split(/\s+/).length / 0.75)) + 4;
                }
                updateContext(total, contextSize, 0.8);
            }
        } catch (e) { /* noop */ }
    }
    setTimeout(pollContext, 1000);
})();
