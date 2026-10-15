// Monitoring page. Every value from the server is put in as plain text (textContent), never as HTML. The operator's
// token is kept in this tab only (sessionStorage) and sent with each call; the backend is what checks it.
"use strict";

const $ = (id) => document.getElementById(id);
const STORE = "aem-monitor-token";
const REFRESH_MS = 30000;

let token = null;
let timer = null;

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
}

function show(id, visible) { $(id).hidden = !visible; }

function loadToken() {
    try { return sessionStorage.getItem(STORE); } catch (error) { return null; }
}

function saveToken(value) {
    try {
        if (value) { sessionStorage.setItem(STORE, value); } else { sessionStorage.removeItem(STORE); }
    } catch (error) { /* private mode: the page still works until it is reloaded */ }
}

const pct = (x) => (x === null || x === undefined ? "–" : (x * 100).toFixed(1) + "%");
const num = (x, digits) => (x === null || x === undefined ? "–" : Number(x).toFixed(digits === undefined ? 0 : digits));
const usd = (x) => (x === null || x === undefined ? "–" : "$" + Number(x).toFixed(4));

function seconds(x) {
    if (x === null || x === undefined) { return "–"; }
    return x >= 120 ? Math.round(x / 60) + " min" : Number(x).toFixed(1) + " s";
}

function card(label, value, note) {
    const box = el("div", "card");
    box.appendChild(el("div", "label", label));
    box.appendChild(el("div", "value", value));
    if (note) { box.appendChild(el("div", "hint", note)); }
    return box;
}

function fill(list, rows) {
    list.textContent = "";
    for (const [name, value] of rows) {
        const row = el("tr");
        row.appendChild(el("td", "", name));
        row.appendChild(el("td", "right", String(value)));
        list.appendChild(row);
    }
    if (!rows.length) { const row = el("tr"); row.appendChild(el("td", "muted", "Nothing yet.")); list.appendChild(row); }
}

// The overall state, worked out only from what the backend reported (no invented "healthy").
function overall(data) {
    const alerts = data.alerts || [];
    const audit = data.audit_integrity;
    if (audit && audit.intact === false) { return ["critical", "Audit trail broken"]; }
    if (alerts.some((a) => a.severity === "critical")) { return ["critical", "Critical alert"]; }
    if (alerts.length) { return ["warning", alerts.length + (alerts.length === 1 ? " warning" : " warnings")]; }
    return ["ok", "No alerts, audit intact"];
}

function render(data) {
    const summary = data.summary || {};
    const [level, text] = overall(data);
    const health = $("health");
    health.hidden = false;
    health.className = "badge " + level;
    health.textContent = text;

    $("meta").textContent = "Numbers come from backend pod " + (data.instance || "unknown")
        + " only: each pod keeps its own logs. Updated " + new Date().toLocaleTimeString() + ".";

    const cards = $("cards");
    cards.textContent = "";
    if (!summary.runs) {
        cards.appendChild(el("p", "muted", "No requests have been recorded on this pod yet."));
    } else {
        cards.appendChild(card("Requests", String(summary.runs), "in the window"));
        cards.appendChild(card("Blocked by the input guard", pct(summary.blocked_rate), "prompt-injection attempts and test traffic"));
        cards.appendChild(card("Escalated to a person", pct(summary.escalation_rate)));
        cards.appendChild(card("Could not be answered", pct(summary.unavailable_rate), "model or index unavailable"));
        cards.appendChild(card("Median response time", seconds(summary.p50_latency_s), "blocked requests excluded"));
        cards.appendChild(card("95th percentile", seconds(summary.p95_latency_s)));
        cards.appendChild(card("Average confidence", num(summary.avg_confidence, 2)));
        cards.appendChild(card("Input tokens per request", num(summary.avg_input_tokens), "average, requests that used the model"));
        cards.appendChild(card("Model calls per request", num(summary.avg_model_calls, 1)));
        cards.appendChild(card("Estimated cost", usd(summary.estimated_cost_usd), "trend at gpt-4o-mini prices"));
        cards.appendChild(card("Rated helpful", pct(summary.helpful_rate), (summary.feedback_count || 0) + " ratings"));
    }

    const alerts = $("alerts");
    alerts.textContent = "";
    for (const alert of data.alerts || []) {
        const li = el("li", "alert " + alert.severity);
        li.appendChild(el("strong", "", alert.severity.toUpperCase() + ": "));
        li.appendChild(el("span", "", alert.message + " (" + (alert.value * 100).toFixed(1) + "%, limit "
            + (alert.threshold * 100).toFixed(0) + "%)"));
        alerts.appendChild(li);
    }
    if (!(data.alerts || []).length) {
        alerts.appendChild(el("li", "muted", "None. Alerts need at least 10 recorded requests."));
    }

    const spend = data.spend || {};
    const cap = spend.cap_usd;
    const meter = $("spend-fill");
    const used = cap ? Math.min(100, (spend.estimated_usd / cap) * 100) : 0;
    meter.style.width = used + "%";
    meter.className = "fill" + (spend.cap_reached ? " bad" : used >= 80 ? " warn" : "");
    $("spend-text").textContent = "Estimated spend " + usd(spend.estimated_usd) + " over " + (spend.calls || 0) + " calls; "
        + (cap ? "cap $" + Number(cap).toFixed(2) + (spend.cap_reached ? " (reached: no further model calls)" : "") : "no cap set") + ".";

    const review = data.review || {};
    const list = $("review");
    list.textContent = "";
    const rows = [
        ["Waiting", review.pending],
        ["Oldest waiting", review.oldest_pending_age_s === null || review.oldest_pending_age_s === undefined ? "–" : seconds(review.oldest_pending_age_s)],
        ["Decided", review.decided],
        ["Confirmed / corrected / rejected", [review.confirmed, review.corrected, review.rejected].map((x) => x || 0).join(" / ")],
        ["Assistant overridden", pct(review.override_rate)],
        ["Median time to decide", seconds(review.median_seconds_to_decide)]
    ];
    for (const [name, value] of rows) {
        list.appendChild(el("dt", "", name));
        list.appendChild(el("dd", "", value === undefined || value === null ? "–" : String(value)));
    }

    const audit = data.audit_integrity;
    $("audit").textContent = audit
        ? (audit.intact ? "Intact: " + (audit.checked || 0) + " records checked, the hash chain is unbroken."
            : "BROKEN at record " + audit.broken_at + (audit.broken_file ? " of " + audit.broken_file : "") + ".")
        : "Not checked.";

    fill($("statuses"), Object.entries(summary.status_counts || {}));
    fill($("equipment"), Object.entries(summary.by_equipment || {}));
}

function problem(message) {
    const box = $("problem");
    box.hidden = !message;
    box.textContent = message || "";
}

async function refresh() {
    if (!token) { return; }
    let res;
    try {
        res = await fetch("/api/monitor/data?limit=" + encodeURIComponent($("limit").value) + "&verify=true", {
            headers: { "X-Monitor-Token": token }
        });
    } catch (error) {
        problem("Could not reach the server. The numbers below are the last ones received.");
        return;
    }
    let body = null;
    try { body = await res.json(); } catch (error) { body = null; }
    if (res.status === 401) { signOut("The token was not accepted. Sign in again."); return; }
    if (res.status === 503) {
        problem("Monitoring is not configured on the server (no monitoring token is set).");
        return;
    }
    if (!res.ok || !body) {
        problem("Monitoring is unavailable (HTTP " + res.status + "). The numbers below are the last ones received.");
        return;
    }
    problem("");
    render(body);
}

function schedule() {
    if (timer) { clearInterval(timer); timer = null; }
    if ($("auto").checked && token) { timer = setInterval(refresh, REFRESH_MS); }
}

function signedIn() {
    show("login", false);
    show("board", true);
    refresh();
    schedule();
}

function signOut(message) {
    token = null;
    saveToken(null);
    schedule();
    show("board", false);
    show("login", true);
    $("health").hidden = true;
    $("login-error").textContent = typeof message === "string" ? message : "";
}

document.addEventListener("DOMContentLoaded", () => {
    $("sign-in").addEventListener("click", () => {
        const value = $("monitor-token").value;
        if (!value) { $("login-error").textContent = "Enter the monitoring token."; return; }
        $("login-error").textContent = "";
        token = value;
        saveToken(value);
        $("monitor-token").value = "";
        signedIn();
    });
    $("sign-out").addEventListener("click", () => signOut());
    $("refresh").addEventListener("click", refresh);
    $("limit").addEventListener("change", refresh);
    $("auto").addEventListener("change", schedule);
    const saved = loadToken();
    if (saved) { token = saved; signedIn(); }
});
