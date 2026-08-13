from dataclasses import dataclass, field
from typing import Any

from app.utils.llm_client import LLMClient


@dataclass(frozen=True)
class OpportunityEngineInput:
    market: str
    city: str = ""
    product: str = "VUNARO"
    target_segment: str = "restaurants"
    market_score: float = 0.0
    demand_score: float = 0.0
    competition_score: float = 0.0
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpportunityEngineOutput:
    market: str
    city: str
    opportunity_score: float
    priority: str
    opportunity_type: str
    reasons: list[str]
    risks: list[str]
    recommended_action: str


class OpportunityEngine:
    """Agent 01 / Module 3: VUNARO Opportunity Engine."""

    agent_id = "agent_01"
    module_id = "opportunity_engine"
    name = "OPPORTUNITY_ENGINE"
    version = "1.0.0"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    def analyze(
        self, request: OpportunityEngineInput
    ) -> OpportunityEngineOutput:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are VUNARO's Opportunity Engine, Agent 01. "
                    "Your job is to evaluate commercial opportunities "
                    "for VUNARO's restaurant SaaS expansion. "
                    "Think commercially, conservatively, and evidence-first. "
                    "Do not invent market statistics. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evaluate this opportunity:\n"
                    f"Market: {request.market}\n"
                    f"City: {request.city}\n"
                    f"Product: {request.product}\n"
                    f"Target segment: {request.target_segment}\n"
                    f"Market score: {request.market_score}\n"
                    f"Demand score: {request.demand_score}\n"
                    f"Competition score: {request.competition_score}\n"
                    f"Constraints: {request.constraints}\n\n"
                    "Return exactly these JSON fields:\n"
                    "{\n"
                    '  "market": "string",\n'
                    '  "city": "string",\n'
                    '  "opportunity_score": 0.0,\n'
                    '  "priority": "HIGH|MEDIUM|LOW",\n'
                    '  "opportunity_type": "string",\n'
                    '  "reasons": ["string"],\n'
                    '  "risks": ["string"],\n'
                    '  "recommended_action": "string"\n'
                    "}\n"
                    "opportunity_score must be between 0 and 100."
                ),
            },
        ]

        result = self.llm.chat_json(
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
            max_attempts=2,
        )

        return self._parse_result(result)

    @staticmethod
    def _parse_result(
        result: dict[str, Any],
    ) -> OpportunityEngineOutput:
        required = (
            "market",
            "city",
            "opportunity_score",
            "priority",
            "opportunity_type",
            "reasons",
            "risks",
            "recommended_action",
        )

        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(
                f"Opportunity Engine response missing fields: "
                f"{', '.join(missing)}"
            )

        opportunity_score = float(result["opportunity_score"])

        if not 0 <= opportunity_score <= 100:
            raise ValueError(
                "opportunity_score must be between 0 and 100"
            )

        priority = str(result["priority"]).upper()
        if priority not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError(
                "priority must be HIGH, MEDIUM, or LOW"
            )

        reasons = result["reasons"]
        risks = result["risks"]

        if not isinstance(reasons, list) or not all(
            isinstance(item, str) for item in reasons
        ):
            raise ValueError("reasons must be a list of strings")

        if not isinstance(risks, list) or not all(
            isinstance(item, str) for item in risks
        ):
            raise ValueError("risks must be a list of strings")

        return OpportunityEngineOutput(
            market=str(result["market"]),
            city=str(result["city"]),
            opportunity_score=opportunity_score,
            priority=priority,
            opportunity_type=str(result["opportunity_type"]),
            reasons=reasons,
            risks=risks,
            recommended_action=str(result["recommended_action"]),
        )
