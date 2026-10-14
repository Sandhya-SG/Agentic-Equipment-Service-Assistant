const STORAGE_KEY = "aem-assistant-conversations-v1";

const conversation = document.getElementById(
    "conversation"
);

const conversationInner = document.getElementById(
    "conversation-inner"
);

const welcome = document.getElementById(
    "welcome"
);

const select = document.getElementById(
    "equipment"
);

const sendButton = document.getElementById(
    "send"
);

const messageBox = document.getElementById(
    "message"
);

const newChatButton = document.getElementById(
    "new-chat"
);

const historyBox = document.getElementById(
    "chat-history"
);

const historyEmpty = document.getElementById(
    "history-empty"
);

const clearHistoryButton = document.getElementById(
    "clear-history"
);

const DEFAULT_PLACEHOLDER = (
    messageBox.placeholder
);


let conversations = loadConversations();

let activeChatId = null;

let conversationId = null;


function createId() {
    return (
        window.crypto
        && window.crypto.randomUUID
    )
        ? window.crypto.randomUUID()
        : (
            Date.now().toString(36)
            + "-"
            + Math.random()
                .toString(36)
                .slice(2)
        );
}


function loadConversations() {
    try {
        const raw = localStorage.getItem(
            STORAGE_KEY
        );

        if (!raw) {
            return [];
        }

        const parsed = JSON.parse(raw);

        return Array.isArray(parsed)
            ? parsed
            : [];
    } catch (error) {
        return [];
    }
}


function saveConversations() {
    try {
        localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify(conversations)
        );
    } catch (error) {
        console.warn(
            "Could not save chat history.",
            error
        );
    }
}


function currentConversation() {
    return conversations.find(
        function (item) {
            return item.id === activeChatId;
        }
    );
}


function createConversation() {
    const chat = {
        id: createId(),
        conversation_id: createId(),
        title: "New equipment conversation",
        equipment_model: select.value || "",
        created_at: Date.now(),
        updated_at: Date.now(),
        messages: []
    };

    conversations.unshift(chat);

    activeChatId = chat.id;
    conversationId = chat.conversation_id;

    saveConversations();

    renderHistory();

    return chat;
}


function ensureConversation() {
    let chat = currentConversation();

    if (!chat) {
        chat = createConversation();
    }

    return chat;
}


function setClarifying(on) {
    messageBox.placeholder = on
        ? "Answer the clarification question above..."
        : DEFAULT_PLACEHOLDER;
}


function el(
    tag,
    className,
    text
) {
    const node = document.createElement(tag);

    if (className) {
        node.className = className;
    }

    if (text !== undefined) {
        node.textContent = text;
    }

    return node;
}


function addList(
    parent,
    title,
    items
) {
    if (!items || !items.length) {
        return;
    }

    const box = el(
        "div",
        "meta"
    );

    box.appendChild(
        el(
            "strong",
            "",
            title
        )
    );

    const list = el("ul");

    items.forEach(
        function (item) {
            list.appendChild(
                el(
                    "li",
                    "",
                    item
                )
            );
        }
    );

    box.appendChild(list);

    parent.appendChild(box);
}


function addEvidenceDetails(
    parent,
    data
) {
    const safety = data.safety || {};

    const hasPpe = (
        safety.ppe_required
        && safety.ppe_required.length
    );

    const hasSources = (
        data.sources
        && data.sources.length
    );

    const hasEscalationReason = (
        Boolean(data.escalation_reason)
    );

    const hasRunId = (
        Boolean(data.run_id)
    );

    if (
        !hasPpe
        && !hasSources
        && !hasEscalationReason
        && !hasRunId
    ) {
        return;
    }

    const details = el(
        "details",
        "evidence-details"
    );

    const summary = el(
        "summary",
        "",
        "Evidence & details"
    );

    const content = el(
        "div",
        "evidence-content"
    );

    addList(
        content,
        "Required PPE",
        safety.ppe_required
    );

    addList(
        content,
        "Sources",
        data.sources
    );

    if (data.escalation_reason) {
        content.appendChild(
            el(
                "div",
                "meta",
                (
                    "Escalation reason: "
                    + data.escalation_reason
                )
            )
        );
    }

    if (data.run_id) {
        content.appendChild(
            el(
                "div",
                "meta",
                "Run ID: " + data.run_id
            )
        );
    }

    details.appendChild(summary);
    details.appendChild(content);

    parent.appendChild(details);
}


function hideWelcome() {
    welcome.style.display = "none";
}


function showWelcome() {
    welcome.style.display = "";
}


function scrollToBottom() {
    window.requestAnimationFrame(
        function () {
            conversation.scrollTop = (
                conversation.scrollHeight
            );
        }
    );
}


function makeTitle(text) {
    const clean = text
        .replace(/\s+/g, " ")
        .trim();

    if (!clean) {
        return "New equipment conversation";
    }

    return clean.length > 38
        ? clean.slice(0, 38) + "…"
        : clean;
}


function renderHistory() {
    historyBox.replaceChildren();

    const sorted = [...conversations].sort(
        function (a, b) {
            return (
                (b.updated_at || 0)
                - (a.updated_at || 0)
            );
        }
    );

    historyEmpty.style.display = (
        sorted.length
            ? "none"
            : ""
    );

    clearHistoryButton.disabled = (
        sorted.length === 0
    );

    sorted.forEach(
        function (chat) {
            const row = el(
                "div",
                "chat-history-row"
            );

            const button = el(
                "button",
                (
                    "chat-history-item"
                    + (
                        chat.id === activeChatId
                            ? " active"
                            : ""
                    )
                ),
                chat.title
            );

            button.type = "button";

            button.title = chat.title;

            button.addEventListener(
                "click",
                function () {
                    switchConversation(
                        chat.id
                    );
                }
            );

            const deleteButton = el(
                "button",
                "delete-chat-button",
                "×"
            );

            deleteButton.type = "button";

            deleteButton.title = (
                "Delete conversation"
            );

            deleteButton.setAttribute(
                "aria-label",
                (
                    "Delete conversation: "
                    + chat.title
                )
            );

            deleteButton.addEventListener(
                "click",
                function (event) {
                    event.stopPropagation();

                    deleteConversation(
                        chat.id
                    );
                }
            );

            row.appendChild(button);
            row.appendChild(deleteButton);

            historyBox.appendChild(row);
        }
    );
}


function deleteConversation(chatId) {
    const chat = conversations.find(
        function (item) {
            return item.id === chatId;
        }
    );

    if (!chat) {
        return;
    }

    const confirmed = window.confirm(
        (
            'Delete "'
            + chat.title
            + '"?'
        )
    );

    if (!confirmed) {
        return;
    }

    const deletingActive = (
        chat.id === activeChatId
    );

    conversations = conversations.filter(
        function (item) {
            return item.id !== chatId;
        }
    );

    saveConversations();

    if (deletingActive) {
        startNewConversation();
        return;
    }

    renderHistory();
}


function clearTimeline() {
    Array.from(
        conversationInner.children
    ).forEach(
        function (child) {
            if (child !== welcome) {
                child.remove();
            }
        }
    );
}


function appendUserMessage(
    text,
    shouldScroll = true
) {
    hideWelcome();

    const row = el(
        "div",
        "message-row user"
    );

    const message = el(
        "div",
        "message"
    );

    message.appendChild(
        el(
            "div",
            "message-label",
            "You"
        )
    );

    message.appendChild(
        el(
            "div",
            "message-bubble",
            text
        )
    );

    row.appendChild(message);

    conversationInner.appendChild(row);

    if (shouldScroll) {
        scrollToBottom();
    }
}


function appendLoadingMessage() {
    const existing = document.getElementById(
        "loading-message"
    );

    if (existing) {
        existing.remove();
    }

    const row = el(
        "div",
        "message-row assistant"
    );

    row.id = "loading-message";

    const message = el(
        "div",
        "message"
    );

    message.appendChild(
        el(
            "div",
            "message-label",
            "Assistant"
        )
    );

    const bubble = el(
        "div",
        "message-bubble"
    );

    const loading = el(
        "div",
        "loading"
    );

    loading.appendChild(
        el(
            "span",
            "loading-dot"
        )
    );

    loading.appendChild(
        el(
            "span",
            "",
            "Searching the manual..."
        )
    );

    bubble.appendChild(loading);

    message.appendChild(bubble);

    row.appendChild(message);

    conversationInner.appendChild(row);

    scrollToBottom();
}


function removeLoadingMessage() {
    const loading = document.getElementById(
        "loading-message"
    );

    if (loading) {
        loading.remove();
    }
}


// Human review: when the assistant sent the answer to a qualified reviewer, the engineer sees that it is pending and
// then the outcome. The wording is to be approved by the subject-matter expert.
const REVIEW_TEXT = {
    pending: "This answer is pending review by a qualified engineer. Do not rely on it until it is confirmed.",
    confirmed: "Reviewed and confirmed by a qualified engineer.",
    corrected: "Reviewed by a qualified engineer, who corrected this guidance:",
    rejected: "A qualified engineer reviewed this answer and withdrew it. Please escalate to your supervisor."
};

function showReviewOutcome(banner, view) {
    banner.className = "review-banner " + view.status;
    banner.textContent = view.message || REVIEW_TEXT[view.status] || "";

    if (view.status === "corrected" && view.corrected_guidance) {
        banner.appendChild(el("div", "review-guidance", view.corrected_guidance));

        if (view.evidence && view.evidence.length) {
            banner.appendChild(el("div", "review-sources", "Sources: " + view.evidence.join("; ")));
        }
    } else if (view.status === "rejected" && view.reason) {
        banner.appendChild(el("div", "review-sources", "Reason: " + view.reason));
    }
}

function addReviewBanner(card, data) {
    const review = data.review;

    if (!review || !review.review_id) {
        return;
    }

    const banner = el("div", "review-banner pending", REVIEW_TEXT.pending);

    card.appendChild(banner);

    let attempts = 0;

    const poll = async () => {
        attempts += 1;

        try {
            const res = await fetch("/api/review/status/" + encodeURIComponent(review.review_id));

            if (res.ok) {
                const view = await res.json();

                if (view.status && view.status !== "pending") {
                    showReviewOutcome(banner, view);
                    return;
                }
            }
        } catch (error) {
            // try again on the next round
        }

        if (attempts < 120) {
            setTimeout(poll, 10000);
        }
    };

    setTimeout(poll, 3000);
}


function addResponseIndicators(
    card,
    data,
    status
) {
    const indicators = el(
        "div",
        "response-indicators"
    );

    indicators.appendChild(
        el(
            "span",
            "badge",
            status
        )
    );

    const explanation = (
        data.explanation || {}
    );

    const confidenceLevel = (
        typeof explanation.confidence_level
        === "string"
    )
        ? explanation.confidence_level
            .trim()
            .toLowerCase()
        : "";

    const allowedLevels = [
        "high",
        "medium",
        "low"
    ];

    if (
        allowedLevels.includes(
            confidenceLevel
        )
    ) {
        indicators.appendChild(
            el(
                "span",
                (
                    "confidence-badge "
                    + confidenceLevel
                ),
                (
                    confidenceLevel
                    + " confidence"
                )
            )
        );
    }

    card.appendChild(indicators);
}


// Model-generated content continues to use textContent.
// Never render model output with innerHTML.
function appendAssistantMessage(
    data,
    shouldScroll = true
) {
    removeLoadingMessage();

    hideWelcome();

    const status = data.status || "error";

    const row = el(
        "div",
        "message-row assistant"
    );

    const message = el(
        "div",
        "message"
    );

    message.appendChild(
        el(
            "div",
            "message-label",
            "Assistant"
        )
    );

    const bubble = el(
        "div",
        "message-bubble"
    );

    const card = el(
        "div",
        "assistant-card " + status
    );

    addResponseIndicators(
        card,
        data,
        status
    );

    card.appendChild(
        el(
            "div",
            "answer",
            (
                data.response
                || data.error
                || JSON.stringify(
                    data,
                    null,
                    2
                )
            )
        )
    );

    const safety = data.safety || {};

    if (safety.warning) {
        card.appendChild(
            el(
                "div",
                "warn",
                safety.warning
            )
        );
    }

    addEvidenceDetails(
        card,
        data
    );

    addReviewBanner(
        card,
        data
    );

    bubble.appendChild(card);

    message.appendChild(bubble);

    row.appendChild(message);

    conversationInner.appendChild(row);

    if (shouldScroll) {
        scrollToBottom();
    }
}


function renderConversation(chat) {
    clearTimeline();

    setClarifying(false);

    if (!chat || !chat.messages.length) {
        showWelcome();
        return;
    }

    hideWelcome();

    chat.messages.forEach(
        function (item) {
            if (item.role === "user") {
                appendUserMessage(
                    item.text,
                    false
                );
            }

            if (item.role === "assistant") {
                appendAssistantMessage(
                    item.data,
                    false
                );

                setClarifying(
                    item.data.status
                    === "clarification"
                );
            }
        }
    );

    scrollToBottom();
}


function switchConversation(chatId) {
    const chat = conversations.find(
        function (item) {
            return item.id === chatId;
        }
    );

    if (!chat) {
        return;
    }

    activeChatId = chat.id;

    conversationId = chat.conversation_id;

    if (chat.equipment_model) {
        select.value = chat.equipment_model;
    }

    renderHistory();

    renderConversation(chat);

    messageBox.value = "";

    resizeComposer();

    messageBox.focus();
}


function startNewConversation() {
    activeChatId = null;

    conversationId = null;

    setClarifying(false);

    clearTimeline();

    showWelcome();

    messageBox.value = "";

    resizeComposer();

    renderHistory();

    messageBox.focus();
}


function resizeComposer() {
    messageBox.style.height = "auto";

    messageBox.style.height = (
        Math.min(
            messageBox.scrollHeight,
            150
        )
        + "px"
    );
}


async function loadEquipment() {
    try {
        const res = await fetch(
            "/api/equipment"
        );

        if (!res.ok) {
            throw new Error(
                "Equipment request failed"
            );
        }

        const items = await res.json();

        select.replaceChildren(
            new Option(
                "Select equipment...",
                ""
            )
        );

        items.forEach(
            function (item) {
                select.appendChild(
                    new Option(
                        item.label,
                        item.id
                    )
                );
            }
        );

        const chat = currentConversation();

        if (
            chat
            && chat.equipment_model
        ) {
            select.value = (
                chat.equipment_model
            );
        }
    } catch (error) {
        select.replaceChildren(
            new Option(
                "Could not load equipment",
                ""
            )
        );
    }
}


select.addEventListener(
    "change",
    function () {
        const chat = currentConversation();

        if (!chat) {
            return;
        }

        chat.equipment_model = select.value;

        chat.updated_at = Date.now();

        saveConversations();
    }
);


messageBox.addEventListener(
    "input",
    resizeComposer
);


messageBox.addEventListener(
    "keydown",
    function (event) {
        if (
            event.key === "Enter"
            && !event.shiftKey
        ) {
            event.preventDefault();

            document
                .getElementById(
                    "chat-form"
                )
                .requestSubmit();
        }
    }
);


newChatButton.addEventListener(
    "click",
    startNewConversation
);


clearHistoryButton.addEventListener(
    "click",
    function () {
        if (!conversations.length) {
            return;
        }

        const confirmed = window.confirm(
            (
                "Clear all chat history from "
                + "this browser?"
            )
        );

        if (!confirmed) {
            return;
        }

        conversations = [];

        localStorage.removeItem(
            STORAGE_KEY
        );

        activeChatId = null;
        conversationId = null;

        setClarifying(false);

        clearTimeline();

        showWelcome();

        messageBox.value = "";

        resizeComposer();

        renderHistory();

        messageBox.focus();
    }
);


document
    .getElementById("chat-form")
    .addEventListener(
        "submit",
        async function (event) {
            event.preventDefault();

            const text = (
                messageBox.value.trim()
            );

            if (
                !text
                || !select.value
                || sendButton.disabled
            ) {
                return;
            }

            const chat = ensureConversation();

            if (!chat.messages.length) {
                chat.title = makeTitle(text);
            }

            chat.equipment_model = select.value;

            chat.updated_at = Date.now();

            chat.messages.push(
                {
                    role: "user",
                    text: text
                }
            );

            saveConversations();

            renderHistory();

            appendUserMessage(text);

            messageBox.value = "";

            resizeComposer();

            sendButton.disabled = true;

            appendLoadingMessage();

            try {
                const res = await fetch(
                    "/api/ask",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify(
                            {
                                message: text,

                                equipment_model:
                                    select.value
                                    || null,

                                conversation_id:
                                    conversationId,

                                clarification_state:
                                    chat.clarification_state
                                    || null
                            }
                        )
                    }
                );

                const data = await res.json();

                if (data.conversation_id) {
                    conversationId = (
                        data.conversation_id
                    );

                    chat.conversation_id = (
                        data.conversation_id
                    );
                }

                // The signed clarification state travels with the conversation, so any backend pod can
                // continue it. A clarification sets it; an answer, a halt or an escalation ends it; a blocked
                // or failed request leaves it as it was (the backend keeps the clarification pending then too).
                if (data.status === "clarification") {
                    chat.clarification_state = (
                        data.clarification_state || null
                    );
                } else if (
                    ["ok", "halted", "escalated"].includes(
                        data.status
                    )
                ) {
                    chat.clarification_state = null;
                }

                chat.messages.push(
                    {
                        role: "assistant",
                        data: data
                    }
                );

                chat.updated_at = Date.now();

                saveConversations();

                renderHistory();

                appendAssistantMessage(data);

                setClarifying(
                    data.status
                    === "clarification"
                );
            } catch (error) {
                const errorData = {
                    status: "error",

                    error:
                        "Error: "
                        + error.message
                };

                chat.messages.push(
                    {
                        role: "assistant",
                        data: errorData
                    }
                );

                chat.updated_at = Date.now();

                saveConversations();

                appendAssistantMessage(
                    errorData
                );

                setClarifying(false);
            } finally {
                sendButton.disabled = false;

                messageBox.focus();
            }
        }
    );


async function initialize() {
    renderHistory();

    await loadEquipment();

    if (conversations.length) {
        const latest = [...conversations].sort(
            function (a, b) {
                return (
                    (b.updated_at || 0)
                    - (a.updated_at || 0)
                );
            }
        )[0];

        switchConversation(
            latest.id
        );
    } else {
        startNewConversation();
    }

    resizeComposer();
}


initialize();