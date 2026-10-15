// Expert review page. All text from the server is put in as plain text (textContent), never as HTML, so a question
// or an answer can never run as markup. The reviewer token is kept in this tab only (sessionStorage) and sent with each call;
// the backend is what checks it.
"use strict";

const $ = (id) => document.getElementById(id);
const STORE = "aem-reviewer";

let reviewer = null;
let currentId = null;

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
}

function show(id, visible) { $(id).hidden = !visible; }

function loadSession() {
    try { return JSON.parse(sessionStorage.getItem(STORE) || "null"); } catch (error) { return null; }
}

function saveSession(value) {
    try {
        if (value) { sessionStorage.setItem(STORE, JSON.stringify(value)); } else { sessionStorage.removeItem(STORE); }
    } catch (error) { /* private mode: the page still works until it is reloaded */ }
}

async function api(path, options = {}) {
    const headers = Object.assign({
        "Content-Type": "application/json",
        "X-Reviewer-Token": reviewer.token,
        "X-Reviewer-Id": reviewer.id
    }, options.headers || {});
    const res = await fetch(path, Object.assign({}, options, { headers }));
    let body = null;
    try { body = await res.json(); } catch (error) { body = null; }
    return { status: res.status, ok: res.ok, body };
}

function whyText(item) {
    return {
        halted: "stopped", escalated: "escalated", escalate_only: "escalate-only hazard",
        low_confidence: "low confidence", sample: "quality sample"
    }[item.trigger] || item.trigger || "";
}

function when(iso) { return iso ? iso.replace("T", " ").slice(0, 16) : ""; }

// What each hazard identifier means, for the reviewer (shown beside the hazards of a case and in the help panel).
// Kept in step with the safety rules by a test: every category the rules can find must be listed here.
const HAZARDS = {
    safety_control_defeat: "A request to bypass, defeat or disable a safety control such as an interlock. Always stopped by a rule, before the manual is searched.",
    high_voltage: "High voltage or live electrical work. Escalate-only: a question that names it gets no procedure, only a referral.",
    stored_energy: "Energy that can release unexpectedly (capacitors, discharge, pressurised parts). Escalate-only, like high voltage.",
    lockout_tagout: "Lock-out / tag-out: isolating and locking energy sources before work. Adds warnings, PPE and a review flag.",
    thermal: "Hot surfaces, heater elements or high temperatures (burn risk). Adds warnings, PPE and a review flag.",
    chemical: "Glycol, isopropyl alcohol, solvents or other hazardous substances. Adds warnings, PPE and a review flag.",
    pressure: "Pneumatic, hydraulic or pressurised lines and supplies. Adds warnings, PPE and a review flag.",
    mechanical: "Moving parts, pinch points or crushing risk. Adds warnings, PPE and a review flag.",
    cryogenic: "Cryogenic or very cold materials. Adds warnings, PPE and a review flag.",
    esd: "Electrostatic discharge when handling static-sensitive parts. Adds warnings, PPE and a review flag."
};

function hazardLine(id) { return HAZARDS[id] ? id + ": " + HAZARDS[id] : id; }

function buildHazardGlossary() {
    const list = $("hazard-glossary");
    list.textContent = "";
    for (const id of Object.keys(HAZARDS)) {
        list.appendChild(el("dt", "", id));
        list.appendChild(el("dd", "", HAZARDS[id]));
    }
}

function signedIn() {
    show("login", false);
    show("queue", true);
    $("who").textContent = "Signed in as " + reviewer.id;
    loadList();
}

function signOut() {
    reviewer = null;
    currentId = null;
    saveSession(null);
    show("queue", false);
    show("detail", false);
    show("login", true);
    $("who").textContent = "";
}

function authFailed(result) {
    if (result.status === 401 || result.status === 400 || result.status === 503) {
        const detail = result.body && result.body.detail;
        signOut();
        $("login-error").textContent = typeof detail === "string" ? detail : "Sign-in failed.";
        return true;
    }
    return false;
}

async function loadList() {
    const result = await api("/api/review/items?status=" + encodeURIComponent($("status-filter").value));
    if (authFailed(result)) { return; }
    if (!result.ok) {
        $("counts").textContent = "Could not load the cases (HTTP " + result.status + ").";
        return;
    }
    const body = result.body;
    const c = body.counts || {};
    $("counts").textContent = "Waiting: " + (c.pending || 0) + "   Confirmed: " + (c.confirmed || 0)
        + "   Corrected: " + (c.corrected || 0) + "   Rejected: " + (c.rejected || 0);
    $("instance").textContent = body.queue_instance ? "Queue on: " + body.queue_instance : "";
    const rows = $("rows");
    rows.textContent = "";
    for (const item of body.items) {
        const tr = el("tr");
        tr.appendChild(el("td", "", when(item.created_at)));
        tr.appendChild(el("td", "", whyText(item)));
        tr.appendChild(el("td", "", item.equipment_model || ""));
        tr.appendChild(el("td", "", item.question));
        const status = el("td");
        status.appendChild(el("span", "badge " + item.status, item.status));
        tr.appendChild(status);
        tr.addEventListener("click", () => openCase(item.review_id));
        rows.appendChild(tr);
    }
    show("empty", body.items.length === 0);
}

function fact(list, name, value) {
    if (value === undefined || value === null || value === "") { return; }
    list.appendChild(el("dt", "", name));
    list.appendChild(el("dd", "", String(value)));
}

function bullets(parent, title, values) {
    if (!values || !values.length) { return; }
    parent.appendChild(el("h3", "", title));
    const ul = el("ul");
    for (const value of values) { ul.appendChild(el("li", "", value)); }
    parent.appendChild(ul);
}

async function openCase(id) {
    const result = await api("/api/review/items/" + encodeURIComponent(id));
    if (authFailed(result)) { return; }
    if (!result.ok) { return; }
    const item = result.body;
    currentId = id;
    $("case-id").textContent = id.slice(0, 8);
    const facts = $("facts");
    facts.textContent = "";
    fact(facts, "Why it is here", whyText(item));
    fact(facts, "Equipment", item.equipment_model);
    fact(facts, "Handled by", item.specialist);
    fact(facts, "Assistant's decision", item.verdict);
    fact(facts, "Confidence", item.confidence_level
        ? item.confidence_level + (item.confidence !== null && item.confidence !== undefined ? " (" + item.confidence + ")" : "") : "");
    fact(facts, "Opened", when(item.created_at));
    $("question").textContent = item.question || "";
    $("answer").textContent = item.answer || "";
    const lists = $("lists");
    lists.textContent = "";
    bullets(lists, "Hazards", (item.hazards || []).map(hazardLine));
    bullets(lists, "Protective equipment", item.ppe_required);
    bullets(lists, "Sources", item.sources);
    const citations = $("citations");
    citations.textContent = "";
    for (const c of item.citations || []) {
        const li = el("li");
        li.appendChild(el("strong", "", [c.source_file, c.page ? "p." + c.page : "", c.section].filter(Boolean).join(", ")));
        li.appendChild(el("div", "muted", c.quote || ""));
        citations.appendChild(li);
    }
    if (!(item.citations || []).length) {
        citations.appendChild(el("li", "muted", "None recorded. A safety rule can stop a request before the manual is searched, "
            + "so there may be no passage to show; the rule that applied is the hazard listed above."));
    }
    renderDecision(item);
    show("detail", true);
    $("detail").scrollIntoView({ behavior: "smooth" });
}

function renderDecision(item) {
    const pending = item.status === "pending";
    show("decision-form", pending);
    const decided = $("decided");
    decided.textContent = "";
    show("decided", !pending);
    $("decision-errors").textContent = "";
    if (pending) {
        $("decision-form").reset();
        show("correction-fields", false);
        return;
    }
    const box = el("div", "decided-box");
    box.appendChild(el("span", "badge " + item.status, item.status));
    box.appendChild(el("div", "muted", "By " + (item.reviewer || "") + ", " + when(item.decided_at)));
    if (item.reason) { box.appendChild(el("div", "", "Reason: " + item.reason)); }
    if (item.corrected_text) {
        box.appendChild(el("h3", "", "Corrected guidance given to the engineer"));
        box.appendChild(el("pre", "", item.corrected_text));
        if ((item.evidence || []).length) { box.appendChild(el("div", "muted", "Sources: " + item.evidence.join("; "))); }
    }
    decided.appendChild(box);
}

async function submitDecision(event) {
    event.preventDefault();
    const errors = $("decision-errors");
    errors.textContent = "";
    const choice = document.querySelector("input[name=decision]:checked");
    if (!choice) { errors.appendChild(el("li", "", "Choose Confirm, Correct or Reject.")); return; }
    const payload = { decision: choice.value, reason: $("reason").value.trim() || null };
    if (choice.value === "correct") {
        payload.corrected_text = $("corrected-text").value.trim();
        payload.evidence = $("evidence").value.split("\n").map((s) => s.trim()).filter(Boolean);
    }
    $("submit-decision").disabled = true;
    try {
        const result = await api("/api/review/items/" + encodeURIComponent(currentId) + "/decision",
            { method: "POST", body: JSON.stringify(payload) });
        if (authFailed(result)) { return; }
        if (result.ok) {
            renderDecision(result.body);
            loadList();
            return;
        }
        const detail = result.body && result.body.detail;
        if (result.status === 409) {
            errors.appendChild(el("li", "", "This case has already been decided."));
            const fresh = await api("/api/review/items/" + encodeURIComponent(currentId));
            if (fresh.ok) { renderDecision(fresh.body); }
            loadList();
        } else if (detail && detail.problems) {
            for (const problem of detail.problems) { errors.appendChild(el("li", "", problem)); }
        } else if (Array.isArray(detail)) {
            errors.appendChild(el("li", "", "Check the form: a field is too long or missing."));
        } else {
            errors.appendChild(el("li", "", typeof detail === "string" ? detail : "The decision could not be recorded."));
        }
    } finally {
        $("submit-decision").disabled = false;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    $("sign-in").addEventListener("click", () => {
        const id = $("reviewer-id").value.trim();
        const token = $("reviewer-token").value;
        if (!id || !token) { $("login-error").textContent = "Enter your name and the reviewer token."; return; }
        $("login-error").textContent = "";
        reviewer = { id, token };
        saveSession(reviewer);
        $("reviewer-token").value = "";
        signedIn();
    });
    $("sign-out").addEventListener("click", signOut);
    $("refresh").addEventListener("click", loadList);
    $("status-filter").addEventListener("change", loadList);
    $("close-detail").addEventListener("click", () => show("detail", false));
    buildHazardGlossary();
    $("help-toggle").addEventListener("click", () => {
        const open = $("help").hidden;
        show("help", open);
        $("help-toggle").setAttribute("aria-expanded", String(open));
        $("help-toggle").textContent = open ? "Hide help" : "Reviewer help";
    });
    $("decision-form").addEventListener("submit", submitDecision);
    for (const radio of document.querySelectorAll("input[name=decision]")) {
        radio.addEventListener("change", () => show("correction-fields", radio.value === "correct" && radio.checked));
    }
    const saved = loadSession();
    if (saved && saved.id && saved.token) { reviewer = saved; signedIn(); }
});
