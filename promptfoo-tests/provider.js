// Custom Promptfoo provider for the AEM backend.
//
// Why a custom provider and not the built-in https one: the tests assert on the HTTP status (a 422 is the
// expected answer to bad input), measure latency, and must send a fresh conversation_id on every call. The
// backend keeps a pending clarification per conversation, so reusing an id (or omitting it) lets one test's
// clarification leak into the next.
//
// Test vars understood here:
//   message, equipment_model, conversation_id   the normal chat body (conversation_id defaults to a fresh one)
//   message_length                              send a message of this many characters (size-limit tests)
//   raw_body                                    JSON string sent as the body unchanged (malformed-input tests)
//   method, path                                for non-chat endpoints (default POST /api/chat)
// The output is a JSON string: { http_status, latency_ms, body }.

const crypto = require('crypto');

class AemChatProvider {
  constructor(options) {
    this.config = (options && options.config) || {};
  }

  id() {
    return 'aem-chat';
  }

  async callApi(prompt, context) {
    const vars = (context && context.vars) || {};
    const base = (process.env.BACKEND_URL || this.config.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
    const method = (vars.method || 'POST').toUpperCase();
    const path = vars.path || '/api/chat';

    let body;
    if (method !== 'GET') {
      if (vars.raw_body !== undefined) {
        body = typeof vars.raw_body === 'string' ? vars.raw_body : JSON.stringify(vars.raw_body);
      } else {
        const payload = {};
        if (vars.message_length) payload.message = 'a'.repeat(Number(vars.message_length));
        else if (vars.message !== undefined) payload.message = vars.message;
        if (vars.equipment_model) payload.equipment_model = vars.equipment_model;
        payload.conversation_id = vars.conversation_id || `pf-${crypto.randomBytes(6).toString('hex')}`;
        body = JSON.stringify(payload);
      }
    }

    const started = Date.now();
    try {
      const response = await fetch(base + path, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: AbortSignal.timeout(Number(this.config.timeoutMs || 120000)),
      });
      const text = await response.text();
      let parsed;
      try {
        parsed = JSON.parse(text);
      } catch (_) {
        parsed = { raw: text };
      }
      return { output: JSON.stringify({ http_status: response.status, latency_ms: Date.now() - started, body: parsed }) };
    } catch (error) {
      return { error: `request failed: ${error.message}` };
    }
  }
}

module.exports = AemChatProvider;
