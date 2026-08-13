from dataclasses import dataclass, field
from typing import Any

from app.utils.llm_client import LLMClient


@dataclass(frozen=True)
class MarketStrategistInput:
    market: str
    product: str = "VUNARO"
    target_segment: str = "restaurants"
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketStrategistOutput:
    recommended_market: str
    recommended_city: str
    beachhead: str
    icp: str
    opportunity_score: float
    competitive_pressure: float
    rationale: list[str]
    expansion_path: list[str]


class MarketStrategist:
    """Agent 01: VUNARO Market Strategist."""

    agent_id = "agent_01"
    name = "MARKET_STRATEGIST"
    version = "1.0.0"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    def analyze(self, request: MarketStrategistInput) -> MarketStrategistOutput:
        constraints = request.constraints or {}

        messages = [
            {
                "role": "system",
                "content": (
                    "You are VUNARO's Market Strategist, Agent 01. "
                    "Your job is to identify the strongest market and city "
                    "for VUNARO's initial restaurant SaaS expansion. "
                    "Think commercially, conservatively, and evidence-first. "
                    "Do not invent precise market statistics. "
                    "If evidence is missing, explicitly state uncertainty. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze this market-entry request:\n"
                    f"Product: {request.product}\n"
                    f"Target segment: {request.target_segment}\n"
                    f"Market under consideration: {request.market}\n"
                    f"Constraints: {constraints}\n\n"
                    "Return exactly these JSON fields:\n"
                    "{\n"
                    '  "recommended_market": "string",\n'
                    '  "recommended_city": "string",\n'
                    '  "beachhead": "string",\n'
                    '  "icp": "string",\n'
                    '  "opportunity_score": 0.0,\n'
                    '  "competitive_pressure": 0.0,\n'
                    '  "rationale": ["string"],\n'
                    '  "expansion_path": ["string"]\n'
                    "}\n\n"
                    "Scores must be between 0 and 100. "
                    "The expansion path should describe a practical "
                    "10 -> 20 -> 50 -> 100 -> 500 restaurant progression."
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
    def _parse_result(result: dict[str, Any]) -> MarketStrategistOutput:
        required = (
            "recommended_market",
            "recommended_city",
            "beachhead",
            "icp",
            "opportunity_score",
            "competitive_pressure",
            "rationale",
            "expansion_path",
        )

        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(
                f"Market Strategist response missing fields: {', '.join(missing)}"
            )

        opportunity_score = float(result["opportunity_score"])
        competitive_pressure = float(result["competitive_pressure"])

        if not 0 <= opportunity_score <= 100:
            raise ValueError("opportunity_score must be between 0 and 100")

        if not 0 <= competitive_pressure <= 100:
            raise ValueError("competitive_pressure must be between 0 and 100")

        rationale = result["rationale"]
        expansion_path = result["expansion_path"]

        if not isinstance(rationale, list) or not all(
            isinstance(item, str) for item in rationale
        ):
            raise ValueError("rationale must be a list of strings")

        if not isinstance(expansion_path, list) or not all(
            isinstance(item, str) for item in expansion_path
        ):
            raise ValueError("expansion_path must be a list of strings")

        return MarketStrategistOutput(
            recommended_market=str(result["recommended_market"]),
            recommended_city=str(result["recommended_city"]),
            beachhead=str(result["beachhead"]),
            icp=str(result["icp"]),
            opportunity_score=opportunity_score,
            competitive_pressure=competitive_pressure,
            rationale=rationale,
            expansion_path=expansion_path,
        )
