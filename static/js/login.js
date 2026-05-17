// login.js — handles signin and signup form submission.
(function () {
    "use strict";

    const tabs = document.querySelectorAll(".tab");
    const formSignin = document.getElementById("form-signin");
    const formSignup = document.getElementById("form-signup");
    const errBox = document.getElementById("msg-error");
    const noticeBox = document.getElementById("msg-notice");

    function showError(text) {
        noticeBox.style.display = "none";
        errBox.style.display = "block";
        errBox.textContent = text;
    }
    function showNotice(text) {
        errBox.style.display = "none";
        noticeBox.style.display = "block";
        noticeBox.textContent = text;
    }
    function clearMsgs() {
        errBox.style.display = "none";
        noticeBox.style.display = "none";
    }

    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            tabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            clearMsgs();
            const which = tab.dataset.tab;
            formSignin.style.display = which === "signin" ? "" : "none";
            formSignup.style.display = which === "signup" ? "" : "none";
        });
    });

    async function postJSON(url, body) {
        const r = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            credentials: "same-origin",
        });
        let data;
        try { data = await r.json(); } catch (e) { data = { ok: false, error: "Bad response." }; }
        return { status: r.status, data };
    }

    formSignin.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearMsgs();
        const body = {
            identifier: document.getElementById("signin-id").value.trim(),
            password: document.getElementById("signin-pw").value,
        };
        const { data } = await postJSON("/api/login", body);
        if (data.ok) {
            window.location.href = "/chat";
        } else {
            showError(data.error || "Sign in failed.");
        }
    });

    formSignup.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearMsgs();
        const body = {
            username: document.getElementById("su-username").value.trim(),
            username2: document.getElementById("su-username2").value.trim(),
            name: document.getElementById("su-name").value.trim(),
            email: document.getElementById("su-email").value.trim(),
            password: document.getElementById("su-pw").value,
            password2: document.getElementById("su-pw2").value,
            zip_code: document.getElementById("su-zip").value.trim(),
            personality: document.getElementById("su-personality").value.trim(),
        };
        const { data } = await postJSON("/api/signup", body);
        if (data.ok) {
            if (data.first_admin) {
                showNotice("Account created — you're the first user, so you've been made admin. Redirecting…");
            } else {
                showNotice("Account created. Redirecting…");
            }
            setTimeout(() => { window.location.href = "/chat"; }, 800);
        } else {
            showError(data.error || "Could not create account.");
        }
    });
})();
