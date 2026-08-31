"""
agent.py

The tool-calling agent, wrapped as a real W&B Weave Model so every call is
versioned and traced.

`run_episode` is the shared core loop: it drives one ticket through a
chat-completion-shaped callable (`chat_fn`) against the mock helpdesk tools
in helpdesk_env.py.

chat_fn(messages, tools_schema) -> {"content": str | None, "tool_calls": [...]}

is intentionally provider-agnostic - it's what lets HelpdeskAgentModel (a real
OpenAI-compatible endpoint), NaiveDummyClient, and OraclePolicyClient all
share one code path.

HelpdeskAgentModel is a weave.Model: a small typed class (model_name,
base_url, api_key) whose @weave.op() predict() method is what
weave.Evaluation calls per dataset row (see eval.py / train_rl.py). Because
weave.Model versions automatically whenever its fields change, pointing
model_name at the base model vs. a Serverless-RL checkpoint
(e.g. "<inference_name>:step30") naturally produces two distinct, comparable
Weave Model versions - the "before" and "after" of this demo.

naive_predict / oracle_predict are plain @weave.op() functions (no config
worth tracking) for the credential-free sanity-check path - Weave supports
evaluating either a Model or a bare op-decorated function.

W&B Weave Agent Tracing: in addition to the standard @weave.op() tracing
above (which powers the Evals/Traces tabs), run_episode() is also
instrumented with Weave's agent-tracing API
(weave.start_conversation/start_turn/start_llm/start_tool - see
docs.wandb.ai/weave/guides/tracking/trace-agents), which populates the
**Agents** tab with a proper conversation/turn/LLM-call/tool-call hierarchy:
one ticket = one conversation with exactly one turn (the ticket is a single
user message; everything the agent does to resolve it - however many LLM
calls and tool calls that takes - nests under that one turn). This is
separate from and complementary to the @weave.op() tracing; both run at
once. It's a no-op if weave.init() hasn't been called, so it's safe to leave
in place even for the credential-free naive/oracle sanity check.
"""

import html
import json
import os
import sys
import uuid

import weave
from dotenv import load_dotenv
from openai import OpenAI
from weave.conversation import Message, Usage

load_dotenv()  # picks up WANDB_API_KEY (and anything else) from a local .env file, if present

try:
    # Package-relative import: works when this file lives inside a package,
    # e.g. examples/helpdesk/agent.py imported as examples.helpdesk.agent
    # (as when a Flask app does `from examples.helpdesk.agent import ...`).
    from .helpdesk_env import (
        HelpdeskTools, ground_truth, KB_ARTICLES, CATEGORIES, world_and_ticket_from_row,
        PolicyComplianceScorer,
    )
except ImportError:
    # Flat/script import: works when agent.py and helpdesk_env.py sit
    # side by side and this file is run directly, e.g. `python agent.py ...`.
    from helpdesk_env import (
        HelpdeskTools, ground_truth, KB_ARTICLES, CATEGORIES, world_and_ticket_from_row,
        PolicyComplianceScorer,
    )

DEFAULT_BASE_URL = "https://api.training.wandb.ai/v1"  # W&B Serverless Training's OpenAI-compatible endpoint

PROJECT = "HelpDeskAgent"  # same project eval.py / train_rl.py log to

SYSTEM_PROMPT = """You are an IT helpdesk triage agent. For each ticket you must:
1. Gather the facts you need using the available tools (account info, ticket
   history, the escalation policy, and the knowledge base).
2. Either resolve the ticket yourself with the right KB article, or escalate
   it to the right team at the right priority.
Always finish by calling exactly one of: resolve_ticket, create_escalation.
Do not guess - use the tools to check account and history details, and follow
the escalation policy rather than the tone of the message."""

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
]

TERMINAL_TOOLS = {"resolve_ticket", "create_escalation"}


@weave.op()
def run_episode(ticket, world, chat_fn, max_turns=8,
                 agent_name="helpdesk-triage-agent", model_name="unknown", provider_name="unknown"):
    tools = HelpdeskTools(world)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"New ticket from account {ticket.account_id}:\n{ticket.text}"},
    ]
    final_action = None

    # W&B Weave Agent Tracing: one ticket = one conversation with exactly one
    # turn (the ticket is a single user message). Every LLM call inside the
    # loop below, and every tool call it triggers, nests under that turn -
    # this is what populates the Agents tab. conversation_id is stable per
    # ticket so re-running the same ticket_id groups into the same
    # conversation; each call still gets its own turn/trace.
    with weave.start_conversation(
        agent_name=agent_name,
        conversation_id=ticket.ticket_id,
        conversation_name=f"{ticket.category} / {ticket.ticket_id}",
        model=model_name,
    ) as conversation:
        with conversation.start_turn(
            user_message=f"New ticket from account {ticket.account_id}:\n{ticket.text}",
            model=model_name,
        ):
            for _ in range(max_turns):
                with weave.start_llm(model=model_name, provider_name=provider_name,
                                      system_instructions=[SYSTEM_PROMPT]) as llm:
                    llm.input_messages = [
                        Message(role=m["role"], content=m.get("content") or "") for m in messages
                    ]

                    response = chat_fn(messages, TOOLS_SCHEMA)
                    messages.append({
                        "role": "assistant",
                        "content": response.get("content"),
                        "tool_calls": response.get("tool_calls"),
                    })

                    tool_calls = response.get("tool_calls") or []
                    llm.output(response.get("content") or f"[requested {len(tool_calls)} tool call(s)]")
                    if response.get("usage"):
                        llm.usage = Usage(**response["usage"])

                    if not tool_calls:
                        break  # model answered without acting - episode ends, no final action; ends llm + falls out of the for loop

                    stop = False
                    for tc in tool_calls:
                        name = tc["function"]["name"]
                        raw_args = tc["function"].get("arguments") or "{}"
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = {}
                        with weave.start_tool(name=name, arguments=raw_args, tool_call_id=tc.get("id", name)) as tool_span:
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
                            tool_span.result = result

                        messages.append({
                            "role": "tool", "tool_call_id": tc.get("id", name),
                            "name": name, "content": json.dumps(result),
                        })
                        if name in TERMINAL_TOOLS and "error" not in result:
                            final_action = tools.calls[-1][1]
                            stop = True
                if stop:
                    break

    return {
        "tool_calls": tools.calls,
        "final_action": final_action,
        "messages": messages,
        "turns": len(messages),
    }


def _tool_calls_to_jsonable(tool_calls):
    return [{"name": name, "args": args} for name, args in tool_calls]


class HelpdeskAgentModel(weave.Model):
    """A Weave Model wrapping one OpenAI-compatible tool-calling endpoint.
    Point model_name/base_url at a base model for the 'before' eval, or at a
    Serverless RL checkpoint (e.g. "<inference_name>:step30") for 'after'.
    Changing those fields creates a new tracked Weave Model version
    automatically - exactly the before/after story this demo needs."""

    model_name: str
    base_url: str
    api_key: str
    max_turns: int = 8

    @weave.op()
    def predict(self, ticket_id: str, account_id: str, category: str, text: str,
                account_tier: str, account_device_os: str,
                account_seats_remaining: int, history_count_30d: int) -> dict:
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
            usage = None
            if getattr(resp, "usage", None):
                usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
            return {"content": msg.content, "tool_calls": tool_calls, "usage": usage}

        world, ticket = world_and_ticket_from_row({
            "ticket_id": ticket_id, "account_id": account_id, "category": category, "text": text,
            "account_tier": account_tier, "account_device_os": account_device_os,
            "account_seats_remaining": account_seats_remaining, "history_count_30d": history_count_30d,
        })
        ep = run_episode(
            ticket, world, chat_fn, max_turns=self.max_turns,
            agent_name="helpdesk-triage-agent", model_name=self.model_name, provider_name="wandb-serverless",
        )
        return {
            "final_action": ep["final_action"],
            "tool_calls": _tool_calls_to_jsonable(ep["tool_calls"]),
            "turns": ep["turns"],
        }


# ---------------------------------------------------------------------------
# Credential-free sanity-check clients/ops. These exist purely to validate
# the harness (helpdesk_env.py + agent.py + eval.py) end to end without any
# API key or GPU:
#
#   naive_predict  - mimics a plausible *untrained* agent: guesses the
#                    category from ticket text and resolves immediately,
#                    never checking account/history/policy. Gets "easy"
#                    tickets right by luck, fails every trap ticket.
#   oracle_predict - "cheats" by calling ground_truth directly, after
#                    properly calling the required tools first. Proves the
#                    scoring harness has a reachable ~100% ceiling.
#
# Run `python eval.py --client naive ...` vs `--client oracle ...` to see the
# floor and the ceiling before spending a single real model call or GPU-hour.
# ---------------------------------------------------------------------------

class NaiveDummyClient:
    def __call__(self, messages, tools_schema):
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        text = user_msg.lower()
        guessed_cat = next((c for c in CATEGORIES if any(tok in text for tok in c.split("_"))), "slow_performance")
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
    ticket's true category/account (it's a test harness, not a real agent)."""

    def __init__(self, world, account_id: str, category: str):
        self.world = world
        self.account_id = account_id
        self.category = category
        self._step = 0

    def __call__(self, messages, tools_schema):
        step, self._step = self._step, self._step + 1

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


@weave.op()
def naive_predict(ticket_id: str, account_id: str, category: str, text: str,
                   account_tier: str, account_device_os: str,
                   account_seats_remaining: int, history_count_30d: int) -> dict:
    world, ticket = world_and_ticket_from_row({
        "ticket_id": ticket_id, "account_id": account_id, "category": category, "text": text,
        "account_tier": account_tier, "account_device_os": account_device_os,
        "account_seats_remaining": account_seats_remaining, "history_count_30d": history_count_30d,
    })
    ep = run_episode(
        ticket, world, NaiveDummyClient(),
        agent_name="helpdesk-triage-naive-baseline", model_name="naive-dummy", provider_name="test-harness",
    )
    return {
        "final_action": ep["final_action"],
        "tool_calls": _tool_calls_to_jsonable(ep["tool_calls"]),
        "turns": ep["turns"],
    }


@weave.op()
def oracle_predict(ticket_id: str, account_id: str, category: str, text: str,
                    account_tier: str, account_device_os: str,
                    account_seats_remaining: int, history_count_30d: int) -> dict:
    world, ticket = world_and_ticket_from_row({
        "ticket_id": ticket_id, "account_id": account_id, "category": category, "text": text,
        "account_tier": account_tier, "account_device_os": account_device_os,
        "account_seats_remaining": account_seats_remaining, "history_count_30d": history_count_30d,
    })
    ep = run_episode(
        ticket, world, OraclePolicyClient(world, account_id, category),
        agent_name="helpdesk-triage-oracle-ceiling", model_name="oracle-dummy", provider_name="test-harness",
    )
    return {
        "final_action": ep["final_action"],
        "tool_calls": _tool_calls_to_jsonable(ep["tool_calls"]),
        "turns": ep["turns"],
    }


def _build_row(ticket_id, question, category, account_tier, account_device_os,
               account_seats_remaining, history_count_30d):
    return {
        "ticket_id": ticket_id,
        "account_id": "cli-account",
        "category": category,
        "text": question,
        "account_tier": account_tier,
        "account_device_os": account_device_os,
        "account_seats_remaining": account_seats_remaining,
        "history_count_30d": history_count_30d,
    }


# category -> {kb_id: description}, flattened for the customer-facing summary
# below (KB_ARTICLES itself is keyed by category, one article each).
_KB_BY_ID = {kb_id: desc for kb_id, desc in KB_ARTICLES.values()}


def _e(value):
    """Shorthand for html.escape() on an arbitrary (possibly non-string)
    value - every piece of ticket/tool-call data interpolated into the HTML
    response below goes through this, since it may contain user-supplied
    text (the ticket text itself) or model-supplied text (tool-call
    arguments), neither of which should be trusted to already be safe HTML."""
    return html.escape(str(value))


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


def _render_customer_response(ticket_id, question, category, output):
    """Builds the agent's final response to the requester as clean, minimal
    HTML - meant to be dropped straight into an HTML chat window (see
    app.py) rather than a plain-text console. Built deterministically from
    the structured tool_calls/final_action the episode already produced,
    rather than a separate free-form LLM call, so it's always coherent and
    available even for the naive/oracle sanity-check clients (neither of
    which is a real model). Every interpolated value is HTML-escaped via
    _e() since it may be user- or model-supplied text.

    Markup is intentionally minimal (no inline styles) - a <div
    class="agent-response"> wrapping a summary line, an <ol> of steps
    (omitted if there were none), and a resolution paragraph - so the host
    chat window's own CSS controls how it looks."""
    resolution = None
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
        steps.append(_describe_tool_call(name, args))

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
    else:
        parts.append(
            '  <p class="agent-response-resolution"><strong>Resolution:</strong> No final resolution was '
            'reached within the allotted turns - this ticket may need manual follow-up.</p>'
        )
    parts.append('</div>')
    return "\n".join(parts)


def ask_agent(question, *, ticket_id=None, category=None,
              account_tier="standard", account_device_os="standard",
              account_seats_remaining=10, history_count_30d=0,
              client="remote", model=None, base_url=DEFAULT_BASE_URL, api_key=None,
              score=False):
    """Programmatic entry point: ask the helpdesk agent one question and get
    back the structured result. This is the single place that owns "build a
    row, run one client (naive/oracle/remote), render a customer-facing
    answer" - both main() below (the CLI) and examples.helpdesk.app's Flask
    route call this, so they can't drift out of sync.

    Runs through the exact same run_episode() path (and therefore the exact
    same W&B Weave agent tracing - start_conversation/start_turn/start_llm/
    start_tool, populating the Agents tab) as every eval/training rollout.
    conversation_id is the ticket_id, so if you reuse the same ticket_id
    across calls (e.g. a multi-message support session), they group into one
    Weave conversation with one turn per call.

    Does NOT call weave.init()/flush() itself - the caller controls the
    Weave client lifecycle. If weave.init() hasn't been called, the tracing
    calls are safely no-ops (see run_episode's docstring note); if it has,
    this traces normally. A long-lived process (e.g. a Flask app) should
    call weave.init() once at startup or once per request per its own
    convention - not from inside this function.

    api_key defaults to the WANDB_API_KEY environment variable (populated
    from a local .env file via python-dotenv - see load_dotenv() at the top
    of this module), so callers only need to pass it explicitly if they want
    to override that.

    Returns: {"ticket_id", "category", "output", "answer", "score" (if
    score=True)}. "output" has "tool_calls"/"final_action"/"turns" -
    "answer" is HTML (see _render_customer_response()), meant to be dropped
    straight into an HTML chat window; printed as-is (raw tags and all) by
    the CLI below.

    Raises ValueError if client="remote" and no api_key/model is available.
    """
    api_key = api_key or os.environ.get("WANDB_API_KEY")
    if client == "remote" and not model:
        raise ValueError("client='remote' requires a model name")
    if client == "remote" and not api_key:
        raise ValueError("client='remote' requires a WANDB_API_KEY - set it in .env or pass api_key explicitly")

    ticket_id = ticket_id or uuid.uuid4().hex[:8]
    category = category or next(
        (c for c in CATEGORIES if any(tok in question.lower() for tok in c.split("_"))), "slow_performance"
    )
    row = _build_row(ticket_id, question, category, account_tier, account_device_os,
                      account_seats_remaining, history_count_30d)

    if client == "naive":
        output = naive_predict(**row)
    elif client == "oracle":
        output = oracle_predict(**row)
    else:
        output = HelpdeskAgentModel(model_name=model, base_url=base_url, api_key=api_key).predict(**row)

    result = {
        "ticket_id": ticket_id,
        "category": category,
        "output": output,
        "answer": _render_customer_response(ticket_id, question, category, output),
    }
    if score:
        result["score"] = PolicyComplianceScorer().score(output=output, **row)
    return result


def main():
    """CLI: ask the agent one ad hoc question and print how it responds.

    Thin wrapper around ask_agent() (see its docstring) that also owns the
    Weave client lifecycle for a one-shot CLI process: init before the call,
    flush before exit.

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
    parser.add_argument("--no-trace", action="store_true",
                         help="Skip weave.init() - run locally without logging to W&B Weave")
    args = parser.parse_args()

    if args.client == "remote" and not args.model:
        sys.exit("--client remote requires --model")

    # WANDB_API_KEY comes from .env (via load_dotenv() at the top of this
    # module) or the environment - needed both to call the real endpoint
    # (--client remote) and to authenticate weave.init() so this call's
    # agent tracing actually lands in your W&B Weave project, same as
    # eval.py.
    api_key = os.environ.get("WANDB_API_KEY")
    if not args.no_trace and not api_key:
        sys.exit("Set WANDB_API_KEY in .env (or export it), or pass --no-trace to skip Weave logging")
    if args.client == "remote" and not api_key:
        sys.exit("Set WANDB_API_KEY in .env (or export it) to use --client remote")

    weave_client = None
    if not args.no_trace:
        # client_parallelism bumped as in eval.py/train_rl.py - agent tracing
        # produces several spans per call and the default can lag behind.
        weave_client = weave.init(PROJECT, settings={"client_parallelism": 50})

    result = ask_agent(
        args.question, ticket_id=f"CLI-{uuid.uuid4().hex[:8]}", category=args.category,
        account_tier=args.account_tier, account_device_os=args.account_device_os,
        account_seats_remaining=args.account_seats_remaining, history_count_30d=args.history_count_30d,
        client=args.client, model=args.model, base_url=args.base_url, score=args.score,
    )
    output = result["output"]

    print(f"\n(category={result['category']}, tier={args.account_tier}, device_os={args.account_device_os}, "
          f"seats_remaining={args.account_seats_remaining}, history_count_30d={args.history_count_30d})")
    print(f"Turns: {output['turns']}\n")
    print(result["answer"])

    if args.score:
        print(f"\nScore (vs. ground truth for category={result['category']}): {result['score']['score']}")
        print(f"Breakdown: {result['score']}")

    if weave_client is not None:
        print("\nflushing pending Weave trace uploads (avoids a long hang on exit)...")
        weave_client.flush()
        print(f"Logged to W&B Weave project '{PROJECT}' - see the Traces and Agents tabs.")


if __name__ == "__main__":
    main()
