import html
import json
import os
import sys
import uuid
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # picks up any endpoint credentials from a local .env file, if present

try:
    # Package-relative import: works when this file lives inside a package,
    # e.g. examples/helpdesk/agent.py imported as
    # examples.helpdesk.agent (as when a Flask app does
    # `from examples.helpdesk.agent import ...`).
    from .helpdesk_env import (
        HelpdeskTools, ground_truth, ground_truth_for_resource, KB_ARTICLES, CATEGORIES,
        world_and_ticket_from_row, PolicyComplianceScorer,
    )
except ImportError:
    # Flat/script import: works when this file and helpdesk_env.py
    # sit side by side and this file is run directly, e.g.
    # `python agent.py ...`.
    from helpdesk_env import (
        HelpdeskTools, ground_truth, ground_truth_for_resource, KB_ARTICLES, CATEGORIES,
        world_and_ticket_from_row, PolicyComplianceScorer,
    )

DEFAULT_BASE_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT = """You are an IT helpdesk triage agent. For each ticket you must:
1. Gather the facts you need using the available tools (account info, ticket
   history, the escalation policy, and the knowledge base).
2. Either resolve the ticket yourself with the right KB article, or escalate
   it to the right team at the right priority.
Finish by calling exactly one of: resolve_ticket, create_escalation.
Do not guess - use the tools to check account and history details, and follow
the escalation policy rather than the tone of the message.

Occasionally a ticket is missing a detail that ONLY the requester can supply
(for example, which specific resource, drive, or folder an access request is
for) - something no amount of lookup_account/check_ticket_history/search_kb
will reveal. In that case, call request_more_info to ask a single, specific
question and stop; you'll be given the requester's reply as a new message
and can continue from there.

Use request_more_info sparingly - most tickets already contain everything
you need. Before asking, check the ticket text carefully: if the requester
already named the resource, system, or software they mean, don't ask again.
Only ask when the missing detail would genuinely change what you do next
and there is truly no other way to get it - never just because a ticket
sounds vague at first glance, and never for something you could instead
find out from your tools."""

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_escalation_policy",
        "description": "Fetch the current IT escalation policy document.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "lookup_account",
        "description": "Look up account tier, device OS, and license seats for a user.",
        "parameters": {"type": "object", "properties": {
            "account_id": {"type": "string"}}, "required": ["account_id"]}}},
    {"type": "function", "function": {
        "name": "check_ticket_history",
        "description": "Check how many tickets this account filed in this category in the last 30 days.",
        "parameters": {"type": "object", "properties": {
            "account_id": {"type": "string"},
            "category": {"type": "string", "enum": CATEGORIES}},
            "required": ["account_id", "category"]}}},
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "Search the knowledge base for a relevant article.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "resolve_ticket",
        "description": "Resolve the ticket directly using a KB article. Terminal action.",
        "parameters": {"type": "object", "properties": {
            "kb_article_id": {"type": "string"},
            "response_text": {"type": "string"}},
            "required": ["kb_article_id", "response_text"]}}},
    {"type": "function", "function": {
        "name": "create_escalation",
        "description": "Escalate the ticket to a specialist team. Terminal action.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string", "enum": ["EUS", "NetOps", "SecOps", "Hardware", "License"]},
            "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
            "notes": {"type": "string"}},
            "required": ["team", "priority", "notes"]}}},
    {"type": "function", "function": {
        "name": "request_more_info",
        "description": (
            "Ask the requester a clarifying question when the ticket is missing information that "
            "ONLY they can supply (e.g. which specific resource/system they mean) - never for "
            "something lookup_account, check_ticket_history, or search_kb could answer instead. "
            "Use sparingly: only when the ticket doesn't already say and the detail truly changes "
            "what you do next - most tickets already have everything you need. Ends this turn and "
            "waits for their reply."
        ),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}}, "required": ["question"]}}},
]

TERMINAL_TOOLS = {"resolve_ticket", "create_escalation"}

# The other tool that ends a turn, but with a "pending clarification"
# outcome rather than a final action - see run_episode()'s status handling
# and _render_customer_response()'s pending-question branch below.
PENDING_TOOLS = {"request_more_info"}

# Categories where request_more_info is NEVER useful, so it's left OUT of
# the tools offered to the model at all (see _tools_schema_for_category
# below) rather than just relying on it following an instruction not to ask
# - a hard guarantee, not a hope. hardware_failure is the one category
# where this holds unconditionally: per helpdesk_env.py's
# POLICY_TEXT, a hardware failure ALWAYS escalates to Hardware regardless of
# what's actually wrong with the machine (including anything about the
# laptop screen/monitor) - priority only depends on account tier, already
# available via lookup_account - so no detail the requester could add would
# ever change the correct action.
#
# Every OTHER category's policy is ALSO fully decidable from
# lookup_account/check_ticket_history/search_kb alone for every ticket in
# this project's own synthetic templates (see helpdesk_env.py's
# TICKET_TEMPLATES and ground_truth) - "access_request" is the one
# EXCEPTION engineered to genuinely need it (see Ticket.hidden_detail) - but
# a REAL, free-form ticket in any of those other categories could
# occasionally be vague enough that asking is genuinely warranted (e.g.
# "I need a license for the design software" without saying which one), so
# the tool stays available there rather than blocked. The model is expected
# to use judgment (see SYSTEM_PROMPT below) about when that's actually true
# rather than asking routinely - and helpdesk_env.py's
# score_trajectory still penalizes asking on any of THIS project's
# synthetic tickets outside access_request, since none of them actually
# withhold a needed detail, so over-asking still shows up as a training/
# eval signal even though the tool itself isn't blocked.
NO_CLARIFICATION_CATEGORIES = {"hardware_failure"}


def _tools_schema_for_category(category):
    """The function-calling tool list to actually offer the model for this
    ticket's category - see NO_CLARIFICATION_CATEGORIES above."""
    if category in NO_CLARIFICATION_CATEGORIES:
        return [t for t in TOOLS_SCHEMA if t["function"]["name"] != "request_more_info"]
    return TOOLS_SCHEMA

# Safety bound on how many "ask -> reply" round trips a single ticket can go
# through end to end (CLI, harness, and app/wb.py all respect this) - a well
# behaved agent needs at most one, so this is just a guard against a model
# that keeps asking.
MAX_CLARIFICATION_ROUNDS = 3

# Keyword hints used ONLY to guess a ticket's category from free-form text
# when a caller doesn't supply category= explicitly (see ask_agent() and
# NaiveDummyClient below). Deliberately NOT "split each category name on
# '_' and check if any piece appears in the text" (the original approach):
# "vpn_access" and "access_request" both contain the generic token
# "access", so a real access_request ticket ("...access to a shared
# drive...") always misguessed as vpn_access under that scheme - silently
# routing it down the wrong policy branch before request_more_info ever got
# a chance to fire. These are curated to actually distinguish the
# categories from each other. Still just a best-effort fallback for ad hoc/
# free-form use (e.g. the CLI with no --category) - every synthetic-data
# caller in this project always passes category explicitly instead of
# relying on this.
CATEGORY_GUESS_KEYWORDS = {
    "password_reset": ["password"],
    "account_lockout": ["locked", "lockout", "lock out"],
    "vpn_access": ["vpn"],
    "wifi": ["wifi", "wi-fi", "wireless"],
    "phishing_report": ["phishing", "phish"],
    "malware_alert": ["malware", "virus", "antivirus"],
    "hardware_failure": ["screen", "keyboard", "hardware", "flicker"],
    "software_license": ["license", "licence"],
    "slow_performance": ["slow", "performance", "forever to load"],
    "access_request": ["shared drive", "folder", "permission", "access to a", "need access"],
}


def _guess_category(text: str) -> str:
    """Best-effort category guess from free-form ticket text - see
    CATEGORY_GUESS_KEYWORDS above for why this isn't simple token-splitting.
    Falls back to "slow_performance" (as before) when nothing matches."""
    text = text.lower()
    for c in CATEGORIES:
        if any(kw in text for kw in CATEGORY_GUESS_KEYWORDS.get(c, [])):
            return c
    return "slow_performance"


def run_episode(ticket, world, chat_fn, max_turns=8,
                 agent_name="helpdesk-triage-agent", model_name="unknown", provider_name="unknown",
                 prior_messages=None, prior_tool_calls=None, follow_up_message=None):
    """
    prior_messages/prior_tool_calls/follow_up_message: set together to RESUME
    a ticket that previously ended on request_more_info (see PENDING_TOOLS
    above and ask_agent()'s "resume" parameter). prior_messages is the full
    chat-message history from the earlier call (system/user/assistant/tool
    messages, unmodified), prior_tool_calls is that earlier call's full
    HelpdeskTools.calls audit log (so scoring - see helpdesk_env.py's
    score_trajectory - sees the whole trajectory, not just this round), and
    follow_up_message is the requester's reply to the pending question,
    appended as a new user message.

    When all three are None (the normal, single-round case), this behaves
    exactly as before: a fresh message history seeded from the ticket text.
    """
    tools = HelpdeskTools(world)
    tools.calls = list(prior_tool_calls) if prior_tool_calls else []

    if prior_messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"New ticket from account {ticket.account_id}:\n{ticket.text}"},
        ]
    else:
        messages = list(prior_messages)
        messages.append({"role": "user", "content": follow_up_message})

    final_action = None
    pending_question = None  # set if this round ends on request_more_info - see PENDING_TOOLS above

    tools_schema = _tools_schema_for_category(ticket.category)

    for _ in range(max_turns):
        response = chat_fn(messages, tools_schema)
        messages.append({
            "role": "assistant",
            "content": response.get("content"),
            "tool_calls": response.get("tool_calls"),
        })

        tool_calls = response.get("tool_calls") or []

        if not tool_calls:
            # Model answered without acting - episode ends here, no final
            # action. Nothing was resolved/escalated/asked.
            break

        stop = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            method = getattr(tools, name, None)
            if method is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = method(**args)
                except Exception as e:
                    # The model called this tool with missing/malformed
                    # arguments (e.g. resolve_ticket without kb_article_id).
                    # Feed the error back as a tool result so the model can
                    # retry next turn, rather than crashing the whole episode
                    # (and, with it, the whole eval run) over one bad call.
                    result = {"error": f"invalid arguments for {name}: {e}"}

            messages.append({
                "role": "tool", "tool_call_id": tc.get("id", name),
                "name": name, "content": json.dumps(result),
            })
            if name in TERMINAL_TOOLS and "error" not in result:
                final_action = tools.calls[-1][1]
                stop = True
            elif name in PENDING_TOOLS and "error" not in result:
                pending_question = args.get("question")
                stop = True
            # Stop at the FIRST terminal/pending tool call in this response,
            # rather than continuing to execute any further ones. Without
            # this, a response that happened to include more than one such
            # call (e.g. the model repeating request_more_info twice in the
            # same completion) would run EACH one even though only the
            # first should ever count.
            if stop:
                break

        if stop:
            break

    final_response_text = _render_customer_response_text(
        ticket.ticket_id, ticket.text,
        {
            "tool_calls": _tool_calls_to_jsonable(tools.calls),
            "status": "pending" if pending_question is not None else "done",
        },
    )

    # Print it to the console, labeled by the agent that produced it
    # (agent_name - "helpdesk-triage-agent" for the real/remote model,
    # "helpdesk-triage-naive-baseline"/"helpdesk-triage-oracle-ceiling" for
    # the sanity-check clients), so it's visible live in whatever's running
    # this - the CLI, a batch harness's output, or a Flask app's server log.
    # One line per round: for a ticket that needed clarification, this
    # prints once when it asks and again once it resolves, matching the two
    # run_episode() calls.
    print(f"[{agent_name}] {final_response_text}")

    return {
        "tool_calls": tools.calls,
        "final_action": final_action,
        "messages": messages,
        "turns": len(messages),
        "turn_trace_id": None,
        "status": "pending" if pending_question is not None else "done",
        "pending_question": pending_question,
    }


def _tool_calls_to_jsonable(tool_calls):
    return [{"name": name, "args": args} for name, args in tool_calls]


class HelpdeskAgentModel:
    """Wraps one OpenAI-compatible tool-calling endpoint. Point model_name/
    base_url at a base model, or at a Serverless RL checkpoint (e.g.
    "<inference_name>:step30"), to compare two configurations."""

    def __init__(self, model_name: str, base_url: str, api_key: str, max_turns: int = 8):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_turns = max_turns

    def predict(self, ticket_id: str, account_id: str, category: str, text: str,
                account_tier: str, account_device_os: str,
                account_seats_remaining: int, history_count_30d: int,
                hidden_detail: Optional[str] = None,
                prior_messages: Optional[list] = None, prior_tool_calls: Optional[list] = None,
                follow_up_message: Optional[str] = None) -> dict:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        def chat_fn(messages, tools_schema):
            resp = client.chat.completions.create(model=self.model_name, messages=messages, tools=tools_schema)
            msg = resp.choices[0].message
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            return {"content": msg.content, "tool_calls": tool_calls}

        world, ticket = world_and_ticket_from_row({
            "ticket_id": ticket_id, "account_id": account_id, "category": category, "text": text,
            "account_tier": account_tier, "account_device_os": account_device_os,
            "account_seats_remaining": account_seats_remaining, "history_count_30d": history_count_30d,
            "hidden_detail": hidden_detail,
        })
        ep = run_episode(
            ticket, world, chat_fn, max_turns=self.max_turns,
            agent_name="helpdesk-triage-agent", model_name=self.model_name, provider_name="wandb-serverless",
            prior_messages=prior_messages, prior_tool_calls=prior_tool_calls, follow_up_message=follow_up_message,
        )
        return {
            "final_action": ep["final_action"],
            "tool_calls": _tool_calls_to_jsonable(ep["tool_calls"]),
            "turns": ep["turns"],
            "turn_trace_id": ep["turn_trace_id"],
            "status": ep["status"],
            "pending_question": ep["pending_question"],
            "messages": ep["messages"],
        }


# ---------------------------------------------------------------------------
# Credential-free sanity-check clients/functions. These exist purely to
# validate the harness (helpdesk_env.py + agent.py) end to
# end without any API key or GPU:
#
#   naive_predict  - mimics a plausible *untrained* agent: guesses the
#                    category from ticket text and resolves immediately,
#                    never checking account/history/policy. Gets "easy"
#                    tickets right by luck, fails every trap ticket.
#   oracle_predict - "cheats" by calling ground_truth directly, after
#                    properly calling the required tools first. Proves the
#                    scoring harness has a reachable ~100% ceiling.
# ---------------------------------------------------------------------------

class NaiveDummyClient:
    def __call__(self, messages, tools_schema):
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        text = user_msg.lower()
        guessed_cat = _guess_category(text)
        kb_id = KB_ARTICLES.get(guessed_cat, KB_ARTICLES["slow_performance"])[0]
        return {
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "resolve_ticket",
                "arguments": json.dumps({"kb_article_id": kb_id, "response_text": "Here is a fix, let us know if it persists."}),
            }}],
        }


class OraclePolicyClient:
    """One-shot: construct a fresh instance per ticket. Already knows the
    ticket's true category/account (it's a test harness, not a real agent).

    For "access_request" tickets - the one category where the correct first
    move is to ask - this "cheats" the same way it does everywhere else: it
    already knows hidden_detail (the true resource), but still calls
    request_more_info on the FIRST round (already_clarified=False) rather
    than skip straight to the answer, so it demonstrates the same
    ask-then-resume trajectory a real trained agent should produce. On the
    resumed round (already_clarified=True, passed by oracle_predict() once
    it detects a resume is underway), it answers directly using
    ground_truth_for_resource - no need to re-derive anything from the
    (simulated) requester's reply text, since this client already has the
    ground truth."""

    def __init__(self, world, account_id: str, category: str, hidden_detail: Optional[str] = None,
                 already_clarified: bool = False):
        self.world = world
        self.account_id = account_id
        self.category = category
        self.hidden_detail = hidden_detail
        self.already_clarified = already_clarified
        self._step = 0

    def __call__(self, messages, tools_schema):
        step, self._step = self._step, self._step + 1

        if self.category == "access_request":
            if not self.already_clarified:
                return {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "request_more_info",
                    "arguments": json.dumps({"question": "Which specific resource, drive, or folder do you need access to?"})}}]}
            resolved = ground_truth_for_resource(self.hidden_detail)
            if resolved["action"] == "resolve":
                return {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "resolve_ticket",
                    "arguments": json.dumps({"kb_article_id": "N/A",
                                              "response_text": f"Access to {self.hidden_detail} granted."})}}]}
            return {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "create_escalation",
                "arguments": json.dumps({"team": resolved["team"], "priority": resolved["priority"],
                                          "notes": f"Access request for restricted resource {self.hidden_detail}."})}}]}

        if step == 0:
            return {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "lookup_account", "arguments": json.dumps({"account_id": self.account_id})}}]}

        if step == 1:
            return {"content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {
                "name": "check_ticket_history",
                "arguments": json.dumps({"account_id": self.account_id, "category": self.category})}}]}

        truth = ground_truth(self.account_id, self.category, self.world)
        if truth["action"] == "resolve":
            return {"content": None, "tool_calls": [{"id": "c3", "type": "function", "function": {
                "name": "resolve_ticket",
                "arguments": json.dumps({"kb_article_id": truth["kb_id"], "response_text": "Resolved per KB."})}}]}
        return {"content": None, "tool_calls": [{"id": "c3", "type": "function", "function": {
            "name": "create_escalation",
            "arguments": json.dumps({"team": truth["team"], "priority": truth["priority"], "notes": "Escalated per policy."})}}]}


def naive_predict(ticket_id: str, account_id: str, category: str, text: str,
                   account_tier: str, account_device_os: str,
                   account_seats_remaining: int, history_count_30d: int,
                   hidden_detail: Optional[str] = None,
                   prior_messages: Optional[list] = None, prior_tool_calls: Optional[list] = None,
                   follow_up_message: Optional[str] = None) -> dict:
    world, ticket = world_and_ticket_from_row({
        "ticket_id": ticket_id, "account_id": account_id, "category": category, "text": text,
        "account_tier": account_tier, "account_device_os": account_device_os,
        "account_seats_remaining": account_seats_remaining, "history_count_30d": history_count_30d,
        "hidden_detail": hidden_detail,
    })
    # NaiveDummyClient never calls request_more_info - it's the deliberately
    # "dumb" baseline, so an access_request ticket is a guaranteed miss for
    # it (see helpdesk_env.py's score_trajectory: no
    # request_more_info call -> score 0), same story as every other trap
    # this baseline exists to fail.
    ep = run_episode(
        ticket, world, NaiveDummyClient(),
        agent_name="helpdesk-triage-naive-baseline", model_name="naive-dummy", provider_name="test-harness",
        prior_messages=prior_messages, prior_tool_calls=prior_tool_calls, follow_up_message=follow_up_message,
    )
    return {
        "final_action": ep["final_action"],
        "tool_calls": _tool_calls_to_jsonable(ep["tool_calls"]),
        "turns": ep["turns"],
        "turn_trace_id": ep["turn_trace_id"],
        "status": ep["status"],
        "pending_question": ep["pending_question"],
        "messages": ep["messages"],
    }


def oracle_predict(ticket_id: str, account_id: str, category: str, text: str,
                    account_tier: str, account_device_os: str,
                    account_seats_remaining: int, history_count_30d: int,
                    hidden_detail: Optional[str] = None,
                    prior_messages: Optional[list] = None, prior_tool_calls: Optional[list] = None,
                    follow_up_message: Optional[str] = None) -> dict:
    world, ticket = world_and_ticket_from_row({
        "ticket_id": ticket_id, "account_id": account_id, "category": category, "text": text,
        "account_tier": account_tier, "account_device_os": account_device_os,
        "account_seats_remaining": account_seats_remaining, "history_count_30d": history_count_30d,
        "hidden_detail": hidden_detail,
    })
    # A resume is underway iff the caller handed back prior context - that's
    # this client's cue to answer (already_clarified=True) instead of asking
    # again. See OraclePolicyClient's docstring.
    already_clarified = prior_messages is not None
    ep = run_episode(
        ticket, world, OraclePolicyClient(world, account_id, category, hidden_detail=hidden_detail,
                                           already_clarified=already_clarified),
        agent_name="helpdesk-triage-oracle-ceiling", model_name="oracle-dummy", provider_name="test-harness",
        prior_messages=prior_messages, prior_tool_calls=prior_tool_calls, follow_up_message=follow_up_message,
    )
    return {
        "final_action": ep["final_action"],
        "tool_calls": _tool_calls_to_jsonable(ep["tool_calls"]),
        "turns": ep["turns"],
        "turn_trace_id": ep["turn_trace_id"],
        "status": ep["status"],
        "pending_question": ep["pending_question"],
        "messages": ep["messages"],
    }


def _build_row(ticket_id, question, category, account_tier, account_device_os,
               account_seats_remaining, history_count_30d, hidden_detail=None):
    return {
        "ticket_id": ticket_id,
        "account_id": "cli-account",
        "category": category,
        "text": question,
        "account_tier": account_tier,
        "account_device_os": account_device_os,
        "account_seats_remaining": account_seats_remaining,
        "history_count_30d": history_count_30d,
        "hidden_detail": hidden_detail,
    }


# category -> {kb_id: description}, flattened for the customer-facing summary
# below (KB_ARTICLES itself is keyed by category, one article each).
_KB_BY_ID = {kb_id: desc for kb_id, desc in KB_ARTICLES.values()}


def _e(value):
    """Shorthand for html.escape() on an arbitrary (possibly non-string)
    value - every piece of ticket/tool-call data interpolated into the HTML
    response below goes through this, since it may contain user-supplied
    text (the ticket text itself) or model-supplied text (tool-call
    arguments), neither of which should be trusted to already be safe HTML.

    Also strips stray backslash-escaping artifacts (\\" / \\') before
    escaping. Model-generated free text (e.g. create_escalation's "notes")
    sometimes over-escapes an embedded quote when the model emits its own
    tool-call arguments as JSON - a common LLM quirk. json.loads() faithfully
    preserves whatever escaping the model produced, so an over-escaped quote
    survives parsing as a literal backslash immediately followed by a quote
    character. html.escape() doesn't touch backslashes (they're not special
    in HTML), so that stray "\" was passing straight through into the
    rendered page in front of every such quote. A literal backslash-quote
    pair is never legitimate content in these already-parsed strings, so
    it's always this artifact - safe to strip unconditionally."""
    text = str(value).replace('\\"', '"').replace("\\'", "'")
    return html.escape(text)


def _describe_tool_call(name, args):
    """Plain-language, HTML-safe description of one non-terminal tool call,
    for the customer-facing summary. resolve_ticket/create_escalation (the
    terminal actions) are described separately as the resolution itself."""
    if name == "get_escalation_policy":
        return "Reviewed the current IT escalation policy."
    if name == "lookup_account":
        return f"Looked up account details for {_e(args.get('account_id'))}."
    if name == "check_ticket_history":
        return (f"Checked ticket history for account {_e(args.get('account_id'))} "
                f"in category &lsquo;{_e(args.get('category'))}&rsquo;.")
    if name == "search_kb":
        return f"Searched the knowledge base for &ldquo;{_e(args.get('query'))}&rdquo;."
    return f"Called {_e(name)}({_e(args)})."


def _response_parts(output):
    """Extracts (resolution_html, steps_html, pending_question) from an
    episode's tool_calls/final_action - the shared, un-rendered facts
    behind both the HTML customer-facing answer (_render_customer_response,
    below) and the plain-text mirror (_render_customer_response_text,
    below). resolution and each step are already HTML-escaped (via _e())
    since the HTML renderer embeds them directly - the plain-text renderer
    unescapes them back via html.unescape() for a plain reader.
    pending_question is returned RAW (not escaped) since callers decide
    separately how to render it."""
    resolution = None
    pending_question = None
    steps = []
    for tc in output["tool_calls"]:
        name, args = tc["name"], tc["args"]
        if name == "resolve_ticket":
            kb_id = args.get("kb_article_id")
            kb_desc = _KB_BY_ID.get(kb_id)
            resolution = (f"Resolved directly using {_e(kb_id)} ({_e(kb_desc)})." if kb_desc
                           else f"Resolved directly using {_e(kb_id)}.")
            continue
        if name == "create_escalation":
            team, priority = args.get("team"), args.get("priority")
            notes = args.get("notes", "")
            resolution = f"Escalated to the {_e(team)} team at {_e(priority)} priority. {_e(notes)}".strip()
            continue
        if name == "request_more_info":
            pending_question = args.get("question")
            steps.append(f"Asked for more information: &ldquo;{_e(pending_question)}&rdquo;")
            continue
        steps.append(_describe_tool_call(name, args))
    return resolution, steps, pending_question


def _render_customer_response(ticket_id, question, category, output):
    """Builds the agent's final response to the requester as clean, minimal
    HTML - meant to be dropped straight into an HTML chat window rather than
    a plain-text console. Built deterministically from the structured
    tool_calls/final_action the episode already produced, rather than a
    separate free-form LLM call, so it's always coherent and available even
    for the naive/oracle sanity-check clients (neither of which is a real
    model). Every interpolated value is HTML-escaped via _e() since it may
    be user- or model-supplied text.

    Markup is intentionally minimal (no inline styles) - a <div
    class="agent-response"> wrapping a summary line, an <ol> of steps
    (omitted if there were none), and a resolution paragraph - so the host
    chat window's own CSS controls how it looks.

    When the episode is still waiting on the requester (output["status"] ==
    "pending" - see run_episode()/PENDING_TOOLS), the resolution paragraph
    is replaced with the clarifying question instead, so callers (CLI,
    harness, Flask app) can render one consistent HTML blob either way
    rather than branching on status themselves."""
    resolution, steps, pending_question = _response_parts(output)

    parts = [
        '<div class="agent-response">',
        f'  <p class="agent-response-summary">Ticket <strong>{_e(ticket_id)}</strong> received: '
        f'&ldquo;{_e(question)}&rdquo;</p>',
    ]
    if steps:
        parts.append('  <p class="agent-response-steps-label"><strong>Steps taken:</strong></p>')
        parts.append('  <ol class="agent-response-steps">')
        parts.extend(f'    <li>{step}</li>' for step in steps)
        parts.append('  </ol>')
    if resolution:
        parts.append(f'  <p class="agent-response-resolution"><strong>Resolution:</strong> {resolution}</p>')
    elif output.get("status") == "pending" and pending_question:
        parts.append(
            f'  <p class="agent-response-pending"><strong>We need a bit more information:</strong> '
            f'{_e(pending_question)}</p>'
        )
    else:
        parts.append(
            '  <p class="agent-response-resolution"><strong>Resolution:</strong> No final resolution was '
            'reached within the allotted turns - this ticket may need manual follow-up.</p>'
        )
    parts.append('</div>')
    return "\n".join(parts)


def _render_customer_response_text(ticket_id, question, output):
    """Plain-text mirror of _render_customer_response() - same summary /
    steps-taken / resolution (or pending-question) content, no HTML markup.
    Used to print the agent's full, final response to the console (see
    run_episode() above)."""
    resolution, steps, pending_question = _response_parts(output)

    lines = [f"Ticket {ticket_id} received: “{question}”"]
    if steps:
        lines.append("")
        lines.append("Steps taken:")
        lines.append("")
        lines.extend(f"{i}. {html.unescape(step)}" for i, step in enumerate(steps, 1))
    lines.append("")
    if resolution:
        lines.append(f"Resolution: {html.unescape(resolution)}")
    elif output.get("status") == "pending" and pending_question:
        lines.append(f"We need a bit more information: {pending_question}")
    else:
        lines.append(
            "Resolution: No final resolution was reached within the allotted turns - "
            "this ticket may need manual follow-up."
        )
    return "\n".join(lines)


def ask_agent(question, *, ticket_id=None, category=None,
              account_tier=None, account_device_os=None,
              account_seats_remaining=None, history_count_30d=None, hidden_detail=None,
              client="remote", model=None, base_url=DEFAULT_BASE_URL, api_key=None,
              score=False, resume=None):
    """Programmatic entry point: ask the helpdesk agent one question and get
    back the structured result. This is the single place that owns "build a
    row, run one client (naive/oracle/remote), render a customer-facing
    answer" - both main() below (the CLI) and a Flask route can call this,
    so they can't drift out of sync.

    api_key defaults to the WANDB_API_KEY environment variable (populated
    from a local .env file via python-dotenv - see load_dotenv() at the top
    of this module), so callers only need to pass it explicitly if they want
    to override that.

    resume: pass the "resume" dict from a PRIOR ask_agent() call whose
    result had status == "pending" (i.e. the agent called request_more_info
    and is waiting on the requester - see run_episode()'s PENDING_TOOLS
    handling), together with `question` now holding the requester's REPLY
    text rather than a new ticket. This continues the SAME ticket_id as a
    second round, with the full prior message and tool-call history threaded
    through so scoring (see helpdesk_env.py's score_trajectory)
    sees the whole trajectory. ticket_id/category/hidden_detail/the account
    fields should simply be OMITTED on a resume call - the resume dict
    already carries all of them forward from the original call (so e.g.
    category isn't re-guessed from the reply text, which usually doesn't
    contain the same keywords the original ticket text did); passing one
    explicitly overrides the carried-forward value.

    Returns: {"ticket_id", "category", "output", "answer", "turn_trace_id",
    "status", "pending_question", "resume" (only when status=="pending"),
    "score" (if score=True)}. "output" has "tool_calls"/"final_action"/
    "turns"/"turn_trace_id"/"status"/"pending_question"/"messages" - "answer"
    is HTML (see _render_customer_response()), meant to be dropped straight
    into an HTML chat window; printed as-is (raw tags and all) by the CLI
    below. "status" is "done" once a terminal action (resolve_ticket/
    create_escalation) has been taken, or "pending" if the agent is instead
    waiting on a reply to request_more_info - in which case "pending_question"
    holds that question and "resume" holds everything a follow-up ask_agent()
    call needs (see the `resume` parameter above). "turn_trace_id" is always
    None in this de-instrumented module (kept only for shape-compatibility
    with callers written against agent.py's contract).

    Raises ValueError if client="remote" and no api_key/model is available.
    """
    api_key = api_key or os.environ.get("WANDB_API_KEY")
    if client == "remote" and not model:
        raise ValueError("client='remote' requires a model name")
    if client == "remote" and not api_key:
        raise ValueError("client='remote' requires a WANDB_API_KEY - set it in .env or pass api_key explicitly")

    prior_messages = prior_tool_calls = follow_up_message = None
    if resume is not None:
        # Carry the original ticket's identity/context forward by default -
        # an explicit kwarg still wins, but the whole point of `resume` is
        # that callers shouldn't need to repeat any of this (and, crucially,
        # category/hidden_detail must NOT be re-derived from the requester's
        # reply text below - that text usually doesn't contain the original
        # ticket's category keywords, e.g. "I need access to team-drive"
        # guessing "vpn_access" from "access").
        ticket_id = ticket_id or resume["ticket_id"]
        category = category or resume.get("category")
        hidden_detail = hidden_detail if hidden_detail is not None else resume.get("hidden_detail")
        account_tier = account_tier or resume.get("account_tier")
        account_device_os = account_device_os or resume.get("account_device_os")
        account_seats_remaining = (account_seats_remaining if account_seats_remaining is not None
                                    else resume.get("account_seats_remaining"))
        history_count_30d = (history_count_30d if history_count_30d is not None
                              else resume.get("history_count_30d"))
        prior_messages = resume["messages"]
        prior_tool_calls = [(tc["name"], tc["args"]) for tc in resume["tool_calls"]]
        follow_up_message = question

    # Defaults applied here (rather than as parameter defaults) so a resume
    # call's carried-forward values above take precedence over them.
    account_tier = account_tier or "standard"
    account_device_os = account_device_os or "standard"
    account_seats_remaining = account_seats_remaining if account_seats_remaining is not None else 10
    history_count_30d = history_count_30d if history_count_30d is not None else 0

    ticket_id = ticket_id or uuid.uuid4().hex[:8]
    category = category or _guess_category(question)
    row = _build_row(ticket_id, question, category, account_tier, account_device_os,
                      account_seats_remaining, history_count_30d, hidden_detail=hidden_detail)
    resume_kwargs = {
        "prior_messages": prior_messages, "prior_tool_calls": prior_tool_calls,
        "follow_up_message": follow_up_message,
    }

    if client == "naive":
        output = naive_predict(**row, **resume_kwargs)
    elif client == "oracle":
        output = oracle_predict(**row, **resume_kwargs)
    else:
        output = HelpdeskAgentModel(model_name=model, base_url=base_url, api_key=api_key).predict(**row, **resume_kwargs)

    result = {
        "ticket_id": ticket_id,
        "category": category,
        "output": output,
        "answer": _render_customer_response(ticket_id, question, category, output),
        "turn_trace_id": output.get("turn_trace_id"),
        "status": output.get("status", "done"),
        "pending_question": output.get("pending_question"),
    }
    if result["status"] == "pending":
        result["resume"] = {
            "ticket_id": ticket_id,
            "category": category,
            "hidden_detail": hidden_detail,
            "account_tier": account_tier,
            "account_device_os": account_device_os,
            "account_seats_remaining": account_seats_remaining,
            "history_count_30d": history_count_30d,
            "messages": output.get("messages"),
            "tool_calls": output.get("tool_calls"),
        }
    if score:
        result["score"] = PolicyComplianceScorer().score(output=output, **row)
    return result


def main():
    """CLI: ask the agent one ad hoc question and print how it responds.

    Thin wrapper around ask_agent() (see its docstring).

    Examples:
      python agent.py "My VPN keeps disconnecting"
      python agent.py "My VPN keeps disconnecting" --client oracle --score
      python agent.py "Need admin access to the billing console" \\
          --client remote --model meta-llama/Llama-3.1-8B-Instruct
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ask the helpdesk triage agent one question and see how it responds.")
    parser.add_argument("question", help="The ticket text, e.g. 'My VPN keeps disconnecting'")
    parser.add_argument("--category", choices=CATEGORIES, default=None,
                         help="Ticket category (determines the hidden account/history facts the agent must look "
                              "up). If omitted, guessed from the question text.")
    parser.add_argument("--account-tier", choices=["standard", "executive"], default="standard")
    parser.add_argument("--account-device-os", choices=["standard", "unsupported_legacy"], default="standard")
    parser.add_argument("--account-seats-remaining", type=int, default=10)
    parser.add_argument("--history-count-30d", type=int, default=0,
                         help="How many tickets in this category this account has filed in the last 30 days")
    parser.add_argument("--client", choices=["naive", "oracle", "remote"], default="remote")
    parser.add_argument("--model", default=None, help="Model name (required for --client remote)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--score", action="store_true", help="Also grade the response against ground truth")
    args = parser.parse_args()

    if args.client == "remote" and not args.model:
        sys.exit("--client remote requires --model")

    api_key = os.environ.get("WANDB_API_KEY")
    if args.client == "remote" and not api_key:
        sys.exit("Set WANDB_API_KEY in .env (or export it) to use --client remote")

    ticket_id = f"CLI-{uuid.uuid4().hex[:8]}"
    result = ask_agent(
        args.question, ticket_id=ticket_id, category=args.category,
        account_tier=args.account_tier, account_device_os=args.account_device_os,
        account_seats_remaining=args.account_seats_remaining, history_count_30d=args.history_count_30d,
        client=args.client, model=args.model, base_url=args.base_url, score=args.score,
    )
    _print_ask_agent_result(result, args)

    # If the agent asked a clarifying question (request_more_info - see
    # run_episode()'s PENDING_TOOLS), keep prompting the person at the
    # terminal for a reply and resuming the SAME ticket_id until it reaches
    # a real resolution, or MAX_CLARIFICATION_ROUNDS is hit.
    rounds = 0
    while result["status"] == "pending" and rounds < MAX_CLARIFICATION_ROUNDS:
        rounds += 1
        try:
            reply = input(f"\n> {result['pending_question']}\nYour reply: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(no reply given - leaving this ticket pending)")
            break
        if not reply:
            print("(empty reply - leaving this ticket pending)")
            break
        result = ask_agent(
            reply, client=args.client, model=args.model, base_url=args.base_url, score=args.score,
            resume=result["resume"],
        )
        _print_ask_agent_result(result, args)


def _print_ask_agent_result(result, args):
    """Shared by main()'s first call and every clarification-round resume
    below, so the CLI prints identically either way."""
    output = result["output"]
    print(f"\n(category={result['category']}, tier={args.account_tier}, device_os={args.account_device_os}, "
          f"seats_remaining={args.account_seats_remaining}, history_count_30d={args.history_count_30d})")
    print(f"Turns: {output['turns']}\n")
    print(result["answer"])

    if args.score:
        print(f"\nScore (vs. ground truth for category={result['category']}): {result['score']['score']}")
        print(f"Breakdown: {result['score']}")


if __name__ == "__main__":
    main()
