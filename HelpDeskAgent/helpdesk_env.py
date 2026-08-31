"""
helpdesk_env.py

Self-contained mock IT helpdesk environment for the Serverless RL demo.

Everything the agent can do lives behind a small set of "tool" functions
(HelpdeskTools below): get_escalation_policy, lookup_account,
check_ticket_history, search_kb, resolve_ticket, create_escalation.

Ground truth for each generated ticket is deterministic and NOT something a
generic instruction-tuned model can guess: several categories require
combining two different tool results (e.g. account tier + ticket history)
against an arbitrary, company-specific policy. That's what makes this a good
RL demo - a capable base model still won't reliably get these right, and the
policy is simple/stable enough to be learnable in a short training run.

score_trajectory() grades a finished episode against that hidden ground
truth, and is deterministic/rule-based on purpose: no LLM judge involved, so
the before/after numbers in the demo are fully auditable.

W&B Weave instrumentation:
  - Every tool call and score_trajectory() are @weave.op(), so a Weave trace
    shows exactly which tools an episode called, with what arguments, and how
    it was graded.
  - ticket_to_row()/build_weave_dataset() flatten tickets (+ their hidden
    account/history fields) into self-contained dicts suitable for
    weave.Evaluation(dataset=...) - see agent.py's HelpdeskAgentModel and this
    file's PolicyComplianceScorer for the other half.
  - world_and_ticket_from_row() is the inverse: it rebuilds a single-ticket
    World from one of those rows, so predict()/score() never need to share a
    mutable World object - each row is fully self-describing, which is the
    idiomatic Weave dataset pattern.

This module requires `weave` to be installed (`pip install weave`).
"""

import random
from dataclasses import dataclass
from typing import Optional

import weave

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

TEAMS = ["EUS", "NetOps", "SecOps", "Hardware", "License"]
PRIORITIES = ["P0", "P1", "P2", "P3"]

CATEGORIES = [
    "password_reset",
    "account_lockout",
    "vpn_access",
    "wifi",
    "phishing_report",
    "malware_alert",
    "hardware_failure",
    "software_license",
    "slow_performance",
]

KB_ARTICLES = {
    "password_reset": ("KB-101", "Self-service password reset: visit reset.corp.internal, verify with MFA, set a new password."),
    "account_lockout": ("KB-102", "Account lockout: wait 15 minutes for auto-unlock, or verify identity to unlock immediately."),
    "vpn_access": ("KB-201", "VPN setup: install the GlobalProtect client, connect to vpn.corp.internal with SSO credentials."),
    "wifi": ("KB-202", "Corporate wifi: forget the network, rejoin 'CorpSecure', authenticate with domain credentials."),
    "hardware_failure": ("KB-301", "Hardware issue triage steps: power cycle, check cables, run built-in diagnostics."),
    "software_license": ("KB-401", "License activation: open the license manager, enter the key from portal.corp.internal/licenses."),
    "slow_performance": ("KB-501", "Performance checklist: clear temp files, check startup apps, restart, run disk cleanup."),
}

POLICY_TEXT = """
IT HELPDESK ESCALATION POLICY (v3)

1. Security: Any phishing report or malware alert must ALWAYS be escalated to
   SecOps, never resolved directly - regardless of how minor it sounds.
   Priority is P0 if the account is 'executive' tier OR the account has any
   prior security ticket in the last 30 days; otherwise P1.

2. Credential issues (password_reset, account_lockout): resolve directly with
   the matching KB article UNLESS the account has 3 or more tickets in the
   SAME category in the last 30 days, which may indicate a compromised
   account - in that case escalate to SecOps at P2.

3. Connectivity (vpn_access, wifi): resolve directly with the matching KB
   article UNLESS the account's device_os is 'unsupported_legacy', in which
   case the standard KB fix does not apply - escalate to NetOps at P2.

4. Hardware failures: always escalate to the Hardware team. Priority is P1
   for executive-tier accounts, otherwise P2.

5. Software licenses: resolve directly with the KB article UNLESS the
   account has 0 seats_remaining, in which case escalate to License at P3
   (requires procurement).

6. Performance complaints: resolve directly with the KB article UNLESS the
   account has 2 or more performance tickets in the last 30 days, which may
   indicate degrading hardware - escalate to Hardware at P3.

Always check the account record and ticket history before deciding. Ticket
tone/urgency language is not a reliable signal of true priority - follow this
policy, not the wording.
"""

TICKET_TEMPLATES = {
    "password_reset": [
        "I can't log into my laptop, it says my password is wrong. Can you help?",
        "URGENT!! Locked out of everything, need my password reset RIGHT NOW.",
    ],
    "account_lockout": [
        "My account got locked after too many login attempts.",
        "System says my account is locked, I need access back today.",
    ],
    "vpn_access": [
        "VPN client won't connect, keeps timing out.",
        "I'm working from home and can't reach the VPN at all.",
    ],
    "wifi": [
        "Can't connect to the office wifi this morning.",
        "Wifi keeps dropping every few minutes on the corp network.",
    ],
    "phishing_report": [
        "Got a weird email asking me to confirm my password, probably nothing but flagging it.",
        "This email looks off, might be phishing, not sure if it's worth reporting.",
    ],
    "malware_alert": [
        "My antivirus popped up a warning, laptop's running kind of slow too.",
        "Got a scary popup about a virus, not sure if I should worry.",
    ],
    "hardware_failure": [
        "My laptop screen has a flickering black bar and won't stop.",
        "Keyboard on my machine has half the keys not responding.",
    ],
    "software_license": [
        "Need a license for the design software, mine expired.",
        "Getting a license error when I open the analytics tool.",
    ],
    "slow_performance": [
        "My computer has been really slow the last few days.",
        "Everything takes forever to load on my laptop lately.",
    ],
}


@dataclass
class Ticket:
    ticket_id: str
    account_id: str
    category: str
    text: str


@dataclass
class Account:
    account_id: str
    tier: str  # "standard" | "executive"
    device_os: str  # "standard" | "unsupported_legacy"
    seats_remaining: int


@dataclass
class World:
    """Holds the hidden account + history data a given ticket batch depends on."""
    accounts: dict
    history: dict  # (account_id, category) -> ticket count in the last 30 days


def generate_world_and_tickets(n: int, seed: int, id_prefix: str = "EV"):
    """Deterministically generate n tickets + the hidden account/history data
    behind them. Use different id_prefix values (e.g. 'TR' for training,
    'EV' for eval - the default) so train and eval sets never look like the
    same tickets even if seeds/n happen to overlap."""
    rng = random.Random(seed)
    accounts = {}
    history = {}
    tickets = []

    for i in range(n):
        account_id = f"{id_prefix}-U{i:04d}"
        category = rng.choice(CATEGORIES)

        tier = "executive" if rng.random() < 0.15 else "standard"
        device_os = "unsupported_legacy" if rng.random() < 0.30 else "standard"
        seats_remaining = 0 if rng.random() < 0.25 else rng.randint(1, 20)
        accounts[account_id] = Account(account_id, tier, device_os, seats_remaining)

        # "trap" conditions: seed history so the naive/obvious answer is wrong
        repeat_count = 0
        if category in ("password_reset", "account_lockout") and rng.random() < 0.35:
            repeat_count = rng.randint(3, 5)
        elif category == "slow_performance" and rng.random() < 0.35:
            repeat_count = rng.randint(2, 4)
        elif category in ("phishing_report", "malware_alert") and rng.random() < 0.3:
            repeat_count = 1
        history[(account_id, category)] = repeat_count

        text = rng.choice(TICKET_TEMPLATES[category])
        tickets.append(Ticket(ticket_id=f"{id_prefix}-T{i:05d}", account_id=account_id, category=category, text=text))

    return World(accounts=accounts, history=history), tickets


# ---------------------------------------------------------------------------
# Tool implementations (bound to a World instance for one episode)
# ---------------------------------------------------------------------------

class HelpdeskTools:
    def __init__(self, world: World):
        self.world = world
        self.calls = []  # audit log of (tool_name, args_or_action) this episode

    @weave.op()
    def get_escalation_policy(self):
        self.calls.append(("get_escalation_policy", {}))
        return POLICY_TEXT

    @weave.op()
    def lookup_account(self, account_id: str):
        self.calls.append(("lookup_account", {"account_id": account_id}))
        acc = self.world.accounts.get(account_id)
        if not acc:
            return {"error": "unknown account"}
        return {"tier": acc.tier, "device_os": acc.device_os, "seats_remaining": acc.seats_remaining}

    @weave.op()
    def check_ticket_history(self, account_id: str, category: str):
        self.calls.append(("check_ticket_history", {"account_id": account_id, "category": category}))
        return {"tickets_last_30_days": self.world.history.get((account_id, category), 0)}

    @weave.op()
    def search_kb(self, query: str):
        self.calls.append(("search_kb", {"query": query}))
        q = query.lower()
        matches = [
            {"kb_id": kb_id, "category": cat, "body": body}
            for cat, (kb_id, body) in KB_ARTICLES.items()
            if any(tok in q for tok in cat.split("_"))
        ]
        if not matches:
            matches = [{"kb_id": kb_id, "category": cat, "body": body} for cat, (kb_id, body) in KB_ARTICLES.items()]
        return matches[:3]

    @weave.op()
    def resolve_ticket(self, kb_article_id: str, response_text: str):
        action = {"type": "resolve", "kb_article_id": kb_article_id, "response_text": response_text}
        self.calls.append(("resolve_ticket", action))
        return {"status": "resolved"}

    @weave.op()
    def create_escalation(self, team: str, priority: str, notes: str):
        action = {"type": "escalate", "team": team, "priority": priority, "notes": notes}
        self.calls.append(("create_escalation", action))
        return {"status": "escalated"}


# ---------------------------------------------------------------------------
# Ground truth + scoring
# ---------------------------------------------------------------------------

def ground_truth(account_id: str, category: str, world: World):
    acc = world.accounts[account_id]
    repeat = world.history.get((account_id, category), 0)

    if category in ("phishing_report", "malware_alert"):
        priority = "P0" if (acc.tier == "executive" or repeat >= 1) else "P1"
        return {"action": "escalate", "team": "SecOps", "priority": priority, "requires": ["lookup_account", "check_ticket_history"]}

    if category in ("password_reset", "account_lockout"):
        if repeat >= 3:
            return {"action": "escalate", "team": "SecOps", "priority": "P2", "requires": ["check_ticket_history"]}
        return {"action": "resolve", "kb_id": KB_ARTICLES[category][0], "requires": ["check_ticket_history"]}

    if category in ("vpn_access", "wifi"):
        if acc.device_os == "unsupported_legacy":
            return {"action": "escalate", "team": "NetOps", "priority": "P2", "requires": ["lookup_account"]}
        return {"action": "resolve", "kb_id": KB_ARTICLES[category][0], "requires": ["lookup_account"]}

    if category == "hardware_failure":
        priority = "P1" if acc.tier == "executive" else "P2"
        return {"action": "escalate", "team": "Hardware", "priority": priority, "requires": ["lookup_account"]}

    if category == "software_license":
        if acc.seats_remaining == 0:
            return {"action": "escalate", "team": "License", "priority": "P3", "requires": ["lookup_account"]}
        return {"action": "resolve", "kb_id": KB_ARTICLES[category][0], "requires": ["lookup_account"]}

    if category == "slow_performance":
        if repeat >= 2:
            return {"action": "escalate", "team": "Hardware", "priority": "P3", "requires": ["check_ticket_history"]}
        return {"action": "resolve", "kb_id": KB_ARTICLES[category][0], "requires": ["check_ticket_history"]}

    raise ValueError(f"unhandled category {category}")


@weave.op()
def score_trajectory(ticket: Ticket, world: World, tool_calls: list, final_action: Optional[dict]):
    """
    tool_calls: HelpdeskTools.calls for this episode - list of (name, args) tuples.
    final_action: the dict passed to resolve_ticket / create_escalation, or None
                  if the episode ended without a terminal action.
    Returns (score in [0, 1], breakdown dict) - used both as the printed eval
    metric and directly as the RL reward.
    """
    truth = ground_truth(ticket.account_id, ticket.category, world)
    called_names = {name for name, _ in tool_calls}

    if final_action is None:
        return 0.0, {"reason": "no_final_action", "required_tools_called": False, "action_type_ok": False}

    breakdown = {}
    required_ok = all(req in called_names for req in truth["requires"])
    breakdown["required_tools_called"] = required_ok

    action_type_ok = final_action["type"] == truth["action"]
    breakdown["action_type_ok"] = action_type_ok

    score = 0.0
    score += 0.35 if action_type_ok else 0.0
    score += 0.25 if required_ok else 0.0

    if truth["action"] == "escalate" and final_action["type"] == "escalate":
        team_ok = final_action.get("team") == truth["team"]
        priority_ok = final_action.get("priority") == truth["priority"]
        breakdown["team_ok"], breakdown["priority_ok"] = team_ok, priority_ok
        score += 0.2 if team_ok else 0.0
        score += 0.2 if priority_ok else 0.0
    elif truth["action"] == "resolve" and final_action["type"] == "resolve":
        kb_ok = final_action.get("kb_article_id") == truth["kb_id"]
        breakdown["kb_ok"] = kb_ok
        score += 0.4 if kb_ok else 0.0
    else:
        breakdown["team_ok"] = breakdown["priority_ok"] = breakdown["kb_ok"] = False

    return round(min(score, 1.0), 3), breakdown


# ---------------------------------------------------------------------------
# Weave dataset <-> World/Ticket conversion
#
# A weave.Evaluation dataset should be a list of self-contained dicts (see
# weave-docs.wandb.ai/guides/core-types/evaluations). Rather than pass a
# shared, mutable World object around, we flatten each ticket plus its hidden
# account/history fields into one row, and rebuild a single-account World
# from that row wherever it's needed (agent.py's HelpdeskAgentModel.predict,
# and PolicyComplianceScorer.score below). This keeps every row fully
# self-describing - you can look at one row in the Weave UI and see exactly
# what determined the correct answer.
# ---------------------------------------------------------------------------

def ticket_to_row(ticket: Ticket, world: World) -> dict:
    acc = world.accounts[ticket.account_id]
    return {
        "ticket_id": ticket.ticket_id,
        "account_id": ticket.account_id,
        "category": ticket.category,
        "text": ticket.text,
        "account_tier": acc.tier,
        "account_device_os": acc.device_os,
        "account_seats_remaining": acc.seats_remaining,
        "history_count_30d": world.history.get((ticket.account_id, ticket.category), 0),
    }


def build_weave_dataset(world: World, tickets: list) -> list:
    """The dataset= argument for weave.Evaluation - one self-contained row
    per ticket."""
    return [ticket_to_row(t, world) for t in tickets]


def world_and_ticket_from_row(row: dict):
    """Inverse of ticket_to_row(): rebuilds a single-account World + Ticket
    from a flattened dataset row so predict()/score() can call the exact same
    HelpdeskTools / ground_truth / score_trajectory code paths used
    everywhere else in this project."""
    account = Account(
        account_id=row["account_id"], tier=row["account_tier"],
        device_os=row["account_device_os"], seats_remaining=row["account_seats_remaining"],
    )
    world = World(
        accounts={row["account_id"]: account},
        history={(row["account_id"], row["category"]): row["history_count_30d"]},
    )
    ticket = Ticket(
        ticket_id=row["ticket_id"], account_id=row["account_id"],
        category=row["category"], text=row["text"],
    )
    return world, ticket


class PolicyComplianceScorer(weave.Scorer):
    """weave.Evaluation scorer: grades one predict() output against the
    ticket's hidden ground truth. Deterministic and rule-based on purpose -
    no LLM judge involved, so the before/after numbers this produces in Weave
    are fully auditable. Numeric/boolean fields returned here are averaged
    automatically by Weave's auto_summarize (means for numbers, counts/
    fractions for booleans), which is what populates the Evals comparison
    view in the UI."""

    @weave.op()
    def score(self, output: dict, ticket_id: str, account_id: str, category: str,
              text: str, account_tier: str, account_device_os: str,
              account_seats_remaining: int, history_count_30d: int) -> dict:
        world, ticket = world_and_ticket_from_row({
            "ticket_id": ticket_id, "account_id": account_id, "category": category, "text": text,
            "account_tier": account_tier, "account_device_os": account_device_os,
            "account_seats_remaining": account_seats_remaining, "history_count_30d": history_count_30d,
        })
        tool_calls = [(tc["name"], tc["args"]) for tc in output.get("tool_calls", [])]
        score, breakdown = score_trajectory(ticket, world, tool_calls, output.get("final_action"))

        result = {
            "score": score,
            "fully_correct": score >= 0.999,
            "action_type_ok": breakdown.get("action_type_ok", False),
            "required_tools_called": breakdown.get("required_tools_called", False),
        }
        for key in ("team_ok", "priority_ok", "kb_ok"):
            if key in breakdown:
                result[key] = breakdown[key]

        if category in ("phishing_report", "malware_alert"):
            final_action = output.get("final_action")
            result["security_escalated_correctly"] = bool(final_action and final_action.get("type") == "escalate")

        return result
