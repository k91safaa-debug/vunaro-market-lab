from dataclasses import dataclass, field
from typing import Any

from app.utils.llm_client import LLMClient


@dataclass(frozen=True)
class ICPEngineInput:
    restaurant_name: str
    market: str
    city: str = ""
    restaurant_type: str = ""
    operational_complexity: str = ""
    communication_dependency: str = ""
    booking_guest_volume: str = ""
    automation_opportunity: str = ""
    ability_to_pay: str = ""
    growth_signals: str = ""
    product_fit: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ICPEngineOutput:
    restaurant_name: str
    market: str
    city: str
    icp_score: float
    tier: str
    priority: str
    component_scores: dict[str, float]
    reasons: list[str]
    disqualifiers: list[str]
    recommended_action: str


class ICPEngine:
    """Agent 01 / Module 4: VUNARO ICP Engine."""

    agent_id = "agent_01"
    module_id = "icp_engine"
    name = "ICP_ENGINE"
    version = "1.0.0"

    WEIGHTS = {
        "restaurant_type_fit": 15,
        "operational_complexity": 15,
        "communication_dependency": 15,
        "booking_guest_volume": 15,
        "automation_opportunity": 15,
        "ability_to_pay": 10,
        "growth_signals": 10,
        "product_fit": 5,
    }

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    def analyze(self, request: ICPEngineInput) -> ICPEngineOutput:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are VUNARO's ICP Engine, Agent 01. "
                    "Your job is to evaluate whether a restaurant is an ideal "
                    "customer for VUNARO's restaurant SaaS. "
                    "Think commercially, conservatively, and evidence-first. "
                    "Do not invent facts. "
                    "Score each factor from 0 to 100. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evaluate this restaurant for VUNARO ICP fit.\n\n"
                    f"Restaurant: {request.restaurant_name}\n"
                    f"Market: {request.market}\n"
                    f"City: {request.city}\n"
                    f"Restaurant type: {request.restaurant_type}\n"
                    f"Operational complexity: {request.operational_complexity}\n"
                    f"Communication dependency: {request.communication_dependency}\n"
                    f"Booking / guest volume: {request.booking_guest_volume}\n"
                    f"Automation opportunity: {request.automation_opportunity}\n"
                    f"Ability to pay: {request.ability_to_pay}\n"
                    f"Growth signals: {request.growth_signals}\n"
                    f"VUNARO product fit: {request.product_fit}\n"
                    f"Constraints: {request.constraints}\n\n"
                    "Return exactly these JSON fields:\n"
                    "{\n"
                    '  "restaurant_name": "string",\n'
                    '  "market": "string",\n'
                    '  "city": "string",\n'
                    '  "restaurant_type_fit": 0.0,\n'
                    '  "operational_complexity": 0.0,\n'
                    '  "communication_dependency": 0.0,\n'
                    '  "booking_guest_volume": 0.0,\n'
                    '  "automation_opportunity": 0.0,\n'
                    '  "ability_to_pay": 0.0,\n'
                    '  "growth_signals": 0.0,\n'
                    '  "product_fit": 0.0,\n'
                    '  "reasons": ["string"],\n'
                    '  "disqualifiers": ["string"],\n'
                    '  "recommended_action": "string"\n'
                    "}\n"
                ),
            },
        ]

        result = self.llm.chat_json(
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
            max_attempts=2,
        )

        return self._parse_result(result, request)

    @classmethod
    def _parse_result(
        cls,
        result: dict[str, Any],
        request: ICPEngineInput,
    ) -> ICPEngineOutput:
        required = (
            "restaurant_name",
            "market",
            "city",
            "restaurant_type_fit",
            "operational_complexity",
            "communication_dependency",
            "booking_guest_volume",
            "automation_opportunity",
            "ability_to_pay",
            "growth_signals",
            "product_fit",
            "reasons",
            "disqualifiers",
            "recommended_action",
        )

        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(
                "ICP Engine response missing fields: "
                + ", ".join(missing)
            )

        component_scores = {
            key: float(result[key])
            for key in cls.WEIGHTS
        }

        for key, score in component_scores.items():
            if not 0 <= score <= 100:
                raise ValueError(
                    f"{key} score must be between 0 and 100"
                )

        reasons = result["reasons"]
        disqualifiers = result["disqualifiers"]

        if not isinstance(reasons, list) or not all(
            isinstance(item, str) for item in reasons
        ):
            raise ValueError("reasons must be a list of strings")

        if not isinstance(disqualifiers, list) or not all(
            isinstance(item, str) for item in disqualifiers
        ):
            raise ValueError("disqualifiers must be a list of strings")

        icp_score = sum(
            component_scores[key] * weight / 100
            for key, weight in cls.WEIGHTS.items()
        )

        if disqualifiers:
            tier = "REJECT"
            priority = "LOW"
        elif icp_score >= 80:
            tier = "A"
            priority = "VERY_HIGH"
        elif icp_score >= 65:
            tier = "B"
            priority = "HIGH"
        elif icp_score >= 50:
            tier = "C"
            priority = "LOW"
        else:
            tier = "REJECT"
            priority = "LOW"

        return ICPEngineOutput(
            restaurant_name=str(result["restaurant_name"]),
            market=str(result["market"]),
            city=str(result["city"]),
            icp_score=round(icp_score, 2),
            tier=tier,
            priority=priority,
            component_scores=component_scores,
            reasons=reasons,
            disqualifiers=disqualifiers,
            recommended_action=str(result["recommended_action"]),
        )
