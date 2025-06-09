from pydantic import BaseModel

from agents import Agent

# Agent to sanity‑check a synthesized report for consistency and recall.
# This can be used to flag potential gaps or obvious mistakes.
# VERIFIER_PROMPT = (
#     "You are a meticulous auditor. You have been handed a financial analysis report. "
#     "Your job is to verify the report is internally consistent, clearly sourced, and makes "
#     "no unsupported claims. Point out any issues or uncertainties."
# )
VERIFIER_PROMPT = (
    "You are a meticulous auditor. You have been handed a financial analysis report. "
    "Your job is to verify the report is internally consistent, clearly sourced, and makes "
    "no unsupported claims. Return false if the report contains verifiable falsehoods. "
    "If the report contains unsupported claims or unsupported material, mark it true, "
    "but point out all issues or uncertainties."
)


class VerificationResult(BaseModel):
    verified: bool
    """Whether the report seems coherent and plausible."""

    issues: str
    """If not verified, describe the main issues or concerns."""


verifier_agent = Agent(
    name="VerificationAgent",
    instructions=VERIFIER_PROMPT,
    # model="gpt-4o-mini",
    model="gpt-4.1-mini",
    output_type=VerificationResult,
)
