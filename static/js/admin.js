/* admin.js
 * Wires the admin control panel:
 *   - tabs
 *   - live slider values
 *   - connection test (+ populate model datalist)
 *   - chat-template preset preview
 *   - gather form -> POST /api/admin/config
 *   - users table with role toggle
 *   - sign out
 *
 * The page renders with window.__ADMIN__ = { config, template_presets }.
 * "Apply Settings" gathers a complete config object and POSTs it; the server
 * merges + writes + updates the live cache, so changes propagate to all users.
 */
(function () {
    "use strict";

    const A = window.__ADMIN__ || { config: {}, template_presets: [] };
    let CONFIG = JSON.parse(JSON.stringify(A.config || {}));   // working copy
    let TEMPLATE_BODIES = {};                                  // filled by /api/admin/templates

    // ----- tiny helpers ------------------------------------------------------

    const $  = (sel, root) => (root || document).querySelector(sel);
    const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

    function showToast(msg, kind) {
        const t = $("#toast");
        if (!t) return;
        t.textContent = msg;
        t.className = "toast show " + (kind || "");
        setTimeout(() => t.classList.remove("show"), 2400);
    }

    async function getJSON(url) {
        const r = await fetch(url, { credentials: "same-origin" });
        let d = null;
        try { d = await r.json(); } catch (_) {}
        return { ok: r.ok, status: r.status, data: d || {} };
    }

    async function postJSON(url, body) {
        const r = await fetch(url, {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body || {}),
        });
        let d = null;
        try { d = await r.json(); } catch (_) {}
        return { ok: r.ok, status: r.status, data: d || {} };
    }

    // ----- tab switching -----------------------------------------------------

    function activateTab(name) {
        $$(".admin-tab").forEach(t => t.classList.toggle("active", t.dataset.section === name));
        $$(".admin-section").forEach(s => s.classList.toggle("active", s.dataset.section === name));
    }

    $$(".admin-tab").forEach(t => {
        t.addEventListener("click", () => activateTab(t.dataset.section));
    });

    // ----- live slider value labels -----------------------------------------

    const SLIDER_FMT = {
        temperature:        v => Number(v).toFixed(2),
        top_p:              v => Number(v).toFixed(2),
        top_k:              v => String(parseInt(v, 10)),
        min_p:              v => Number(v).toFixed(2),
        repeat_penalty:     v => Number(v).toFixed(2),
        frequency_penalty:  v => Number(v).toFixed(2),
        presence_penalty:   v => Number(v).toFixed(2),
        compaction_threshold: v => Math.round(Number(v) * 100) + "%",
        fade_in_ms:         v => parseInt(v, 10) + "ms",
        stream_speed_ms:    v => parseInt(v, 10) + "ms/char",
    };

    Object.keys(SLIDER_FMT).forEach(key => {
        const input = document.getElementById(key);
        const out = document.getElementById(key + "_val");
        if (!input || !out) return;
        const update = () => { out.textContent = SLIDER_FMT[key](input.value); };
        input.addEventListener("input", update);
        update();
    });

    // ----- connection test ---------------------------------------------------

    function setConnStatus(state, label, detail) {
        const pill = $("#conn-status");
        if (!pill) return;
        pill.classList.remove("ok", "bad");
        if (state === "ok")  pill.classList.add("ok");
        if (state === "bad") pill.classList.add("bad");
        const labelEl = pill.querySelector("span:last-child");
        if (labelEl) labelEl.textContent = label || "unknown";
        const detailEl = $("#conn-detail");
        if (detailEl) detailEl.textContent = detail || "";
    }

    $("#btn-test")?.addEventListener("click", async () => {
        const endpoint = $("#endpoint").value.trim();
        const api_key  = $("#api_key").value;
        setConnStatus("", "testing…", "");
        const { ok, data } = await postJSON("/api/admin/test", { endpoint, api_key });
        if (ok && data.ok) {
            setConnStatus("ok", "connected", data.message || "");
            // Populate the model datalist
            const dl = $("#model-options");
            if (dl) {
                dl.innerHTML = "";
                (data.models || []).forEach(name => {
                    const opt = document.createElement("option");
                    opt.value = name;
                    dl.appendChild(opt);
                });
            }
            showToast("Endpoint reachable.", "good");
        } else {
            setConnStatus("bad", "unreachable", (data && data.message) || "no response");
            showToast("Couldn't reach endpoint.", "bad");
        }
    });

    // ----- chat template preset preview --------------------------------------

    async function loadTemplateBodies() {
        const { ok, data } = await getJSON("/api/admin/templates");
        if (ok && data.ok) {
            TEMPLATE_BODIES = data.presets || {};
            refreshTemplatePreview();
        }
    }

    function refreshTemplatePreview() {
        const sel = $("#chat_template_preset");
        const ta  = $("#template_preview");
        if (!sel || !ta) return;
        const body = TEMPLATE_BODIES[sel.value] || "";
        ta.value = body;
    }

    $("#chat_template_preset")?.addEventListener("change", refreshTemplatePreview);

    // ----- gather: form -> CONFIG -------------------------------------------
    //
    // We rebuild the entire config object from current form values, preserving
    // anything the form doesn't expose (so we don't drop fields the backend
    // added but the UI doesn't know about).

    function readNumber(id, fallback) {
        const el = document.getElementById(id);
        if (!el) return fallback;
        const v = parseFloat(el.value);
        return Number.isFinite(v) ? v : fallback;
    }
    function readInt(id, fallback) {
        const el = document.getElementById(id);
        if (!el) return fallback;
        const v = parseInt(el.value, 10);
        return Number.isFinite(v) ? v : fallback;
    }
    function readText(id, fallback) {
        const el = document.getElementById(id);
        return el ? el.value : (fallback || "");
    }
    function readBool(id) {
        const el = document.getElementById(id);
        return !!(el && el.checked);
    }

    function gatherConfig() {
        const cfg = JSON.parse(JSON.stringify(CONFIG));   // start from working copy
        cfg.server  = cfg.server  || {};
        cfg.llm     = cfg.llm     || {};
        cfg.sampling = cfg.sampling || {};
        cfg.personas = cfg.personas || {};

        // Connection / Model
        cfg.llm.endpoint   = readText("endpoint", cfg.llm.endpoint);
        cfg.llm.api_key    = readText("api_key", cfg.llm.api_key);
        cfg.llm.model_name = readText("model_name", cfg.llm.model_name);
        cfg.llm.context_size = readInt("context_size", cfg.llm.context_size || 8196);
        cfg.llm.compaction_threshold = readNumber("compaction_threshold", cfg.llm.compaction_threshold || 0.8);

        // Sampling
        cfg.sampling.temperature = readNumber("temperature", cfg.sampling.temperature);
        cfg.sampling.top_p       = readNumber("top_p", cfg.sampling.top_p);
        cfg.sampling.top_k       = readInt("top_k", cfg.sampling.top_k);
        cfg.sampling.min_p       = readNumber("min_p", cfg.sampling.min_p);
        cfg.sampling.repeat_penalty    = readNumber("repeat_penalty", cfg.sampling.repeat_penalty);
        cfg.sampling.frequency_penalty = readNumber("frequency_penalty", cfg.sampling.frequency_penalty);
        cfg.sampling.presence_penalty  = readNumber("presence_penalty", cfg.sampling.presence_penalty);
        cfg.sampling.max_tokens  = readInt("max_tokens", cfg.sampling.max_tokens);

        // System message
        cfg.llm.system_message = readText("system_message", cfg.llm.system_message);

        // Chat template
        cfg.llm.chat_template_preset   = readText("chat_template_preset", cfg.llm.chat_template_preset);
        cfg.llm.chat_template_override = readText("chat_template_override", cfg.llm.chat_template_override);
        cfg.llm.use_template_override  = readBool("use_template_override");

        // Personas (A, B, C, D)
        ["A", "B", "C", "D"].forEach(k => {
            cfg.personas[k] = cfg.personas[k] || {};
            const root = document.querySelector(`.persona-card-admin[data-key="${k}"]`);
            if (!root) return;
            const nameEl = root.querySelector(`[data-field="persona_${k}_name"]`);
            const sumEl  = root.querySelector(`[data-field="persona_${k}_summary"]`);
            const descEl = root.querySelector(`[data-field="persona_${k}_description"]`);
            if (nameEl) cfg.personas[k].name = nameEl.value;
            if (sumEl)  cfg.personas[k].summary = sumEl.value;
            if (descEl) cfg.personas[k].description = descEl.value;
        });

        // UI & Thinking
        cfg.llm.fade_in_ms      = readInt("fade_in_ms", cfg.llm.fade_in_ms);
        cfg.llm.stream_speed_ms = readInt("stream_speed_ms", cfg.llm.stream_speed_ms);
        cfg.llm.thinking_enabled         = readBool("thinking_enabled");
        cfg.llm.thinking_compact_default = readBool("thinking_compact_default");
        cfg.llm.thinking_hidden          = readBool("thinking_hidden");

        // Network
        cfg.server.lan_visible = readBool("lan_visible");
        cfg.server.port        = readInt("port", cfg.server.port || 5005);

        return cfg;
    }

    // ----- Apply / Reload ----------------------------------------------------

    $("#btn-apply")?.addEventListener("click", async () => {
        const cfg = gatherConfig();
        const { ok, data } = await postJSON("/api/admin/config", { config: cfg });
        if (ok && data.ok) {
            CONFIG = data.config || cfg;
            showToast("Settings applied to all sessions.", "good");
        } else {
            showToast((data && data.message) || "Could not save settings.", "bad");
        }
    });

    $("#btn-reload")?.addEventListener("click", async () => {
        const { ok, data } = await getJSON("/api/admin/config");
        if (ok && data.ok) {
            CONFIG = data.config || CONFIG;
            applyConfigToForm(CONFIG);
            await loadTemplateBodies();
            showToast("Reloaded from disk.", "good");
        } else {
            showToast("Could not reload.", "bad");
        }
    });

    function setVal(id, value) {
        const el = document.getElementById(id);
        if (!el) return;
        if (el.type === "checkbox") el.checked = !!value;
        else el.value = (value === undefined || value === null) ? "" : value;
        // Refresh any companion label
        if (SLIDER_FMT[id]) {
            const out = document.getElementById(id + "_val");
            if (out) out.textContent = SLIDER_FMT[id](el.value);
        }
    }

    function applyConfigToForm(cfg) {
        cfg = cfg || {};
        const llm = cfg.llm || {}, s = cfg.sampling || {}, srv = cfg.server || {}, p = cfg.personas || {};

        setVal("endpoint", llm.endpoint);
        setVal("api_key", llm.api_key);
        setVal("model_name", llm.model_name);
        setVal("context_size", llm.context_size);
        setVal("compaction_threshold", llm.compaction_threshold);

        setVal("temperature", s.temperature);
        setVal("top_p", s.top_p);
        setVal("top_k", s.top_k);
        setVal("min_p", s.min_p);
        setVal("repeat_penalty", s.repeat_penalty);
        setVal("frequency_penalty", s.frequency_penalty);
        setVal("presence_penalty", s.presence_penalty);
        setVal("max_tokens", s.max_tokens);

        setVal("system_message", llm.system_message);
        setVal("chat_template_preset", llm.chat_template_preset);
        setVal("chat_template_override", llm.chat_template_override);
        setVal("use_template_override", llm.use_template_override);

        ["A", "B", "C", "D"].forEach(k => {
            const data = p[k] || {};
            const root = document.querySelector(`.persona-card-admin[data-key="${k}"]`);
            if (!root) return;
            const n = root.querySelector(`[data-field="persona_${k}_name"]`);
            const su = root.querySelector(`[data-field="persona_${k}_summary"]`);
            const d = root.querySelector(`[data-field="persona_${k}_description"]`);
            if (n)  n.value  = data.name || "";
            if (su) su.value = data.summary || "";
            if (d)  d.value  = data.description || "";
        });

        setVal("fade_in_ms", llm.fade_in_ms);
        setVal("stream_speed_ms", llm.stream_speed_ms);
        setVal("thinking_enabled", llm.thinking_enabled);
        setVal("thinking_compact_default", llm.thinking_compact_default);
        setVal("thinking_hidden", llm.thinking_hidden);

        setVal("lan_visible", srv.lan_visible);
        setVal("port", srv.port);

        refreshTemplatePreview();
    }

    // ----- Users tab ---------------------------------------------------------

    async function loadUsers() {
        const tbody = $("#user-table tbody");
        if (!tbody) return;
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--ink-faint);">loading…</td></tr>`;
        const { ok, data } = await getJSON("/api/admin/users");
        if (!ok || !data.ok) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:#871818;">could not load</td></tr>`;
            return;
        }
        const users = data.users || [];
        if (!users.length) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--ink-faint);">no users yet</td></tr>`;
            return;
        }
        tbody.innerHTML = "";
        users.forEach(u => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${escapeHTML(u.username || "")}</strong></td>
                <td>${escapeHTML(u.name || "")}</td>
                <td>${escapeHTML(u.email || "")}</td>
                <td><span class="role-pill ${u.role === "admin" ? "admin" : "base"}">${u.role || "base"}</span></td>
                <td>${u.messages || 0}</td>
            `;
            const actionTd = document.createElement("td");
            actionTd.style.textAlign = "right";
            const btn = document.createElement("button");
            btn.className = "btn ghost";
            btn.textContent = u.role === "admin" ? "Demote to base" : "Promote to admin";
            btn.addEventListener("click", async () => {
                btn.disabled = true;
                const newRole = u.role === "admin" ? "base" : "admin";
                const { ok, data } = await postJSON("/api/admin/user_role",
                                                   { username: u.username, role: newRole });
                if (ok && data.ok) {
                    showToast(`${u.username} → ${newRole}`, "good");
                    loadUsers();
                } else {
                    btn.disabled = false;
                    showToast((data && data.message) || "Couldn't change role.", "bad");
                }
            });
            actionTd.appendChild(btn);
            tr.appendChild(actionTd);
            tbody.appendChild(tr);
        });
    }

    function escapeHTML(s) {
        return String(s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    // Reload users when the Users tab is opened.
    $$(".admin-tab").forEach(t => {
        if (t.dataset.section === "users") {
            t.addEventListener("click", loadUsers);
        }
    });

    // ----- sign out ----------------------------------------------------------

    $("#logout-btn")?.addEventListener("click", async (e) => {
        e.preventDefault();
        await postJSON("/api/logout", {});
        window.location.href = "/login";
    });

    // ----- boot --------------------------------------------------------------

    loadTemplateBodies();
    // If users tab is the initial active tab, pre-load it. Otherwise it loads on click.
    if ($(".admin-tab.active")?.dataset.section === "users") loadUsers();
})();
