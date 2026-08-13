from dataclasses import dataclass, field
from typing import Any

from app.utils.llm_client import LLMClient


@dataclass(frozen=True)
class MarketIntelligenceInput:
    market: str
    city: str = ""
    product: str = "VUNARO"
    target_segment: str = "restaurants"
    evidence: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketIntelligenceOutput:
    market: str
    city: str
    demand_signals: list[str]
    competition_signals: list[str]
    buying_signals: list[str]
    risks: list[str]
    evidence_quality: float
    confidence: float
    rationale: list[str]
    next_step: str


class MarketIntelligence:
    """Agent 01 / Module 2: VUNARO Market Intelligence."""

    agent_id = "agent_01_market_intelligence"
    name = "MARKET_INTELLIGENCE"
    version = "1.0.0"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    def analyze(
        self, request: MarketIntelligenceInput
    ) -> MarketIntelligenceOutput:
        evidence = request.evidence or ["No external evidence supplied."]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are VUNARO's Market Intelligence Agent, Module 2. "
                    "Analyze evidence for restaurant SaaS expansion. "
                    "Do not invent statistics, businesses, sources, or facts. "
                    "Separate evidence from inference. "
                    "If evidence is weak, say so explicitly. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Market: {request.market}\n"
                    f"City: {request.city}\n"
                    f"Product: {request.product}\n"
                    f"Target segment: {request.target_segment}\n"
                    f"Constraints: {request.constraints}\n"
                    f"Evidence:\n{evidence}\n\n"
                    "Return exactly these JSON fields:\n"
                    "{\n"
                    '  "market": "string",\n'
                    '  "city": "string",\n'
                    '  "demand_signals": ["string"],\n'
                    '  "competition_signals": ["string"],\n'
                    '  "buying_signals": ["string"],\n'
                    '  "risks": ["string"],\n'
                    '  "evidence_quality": 0.0,\n'
                    '  "confidence": 0.0,\n'
                    '  "rationale": ["string"],\n'
                    '  "next_step": "string"\n'
                    "}\n"
                    "Scores must be between 0 and 100."
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
    def _parse_result(result: dict[str, Any]) -> MarketIntelligenceOutput:
        required = (
            "market",
            "city",
            "demand_signals",
            "competition_signals",
            "buying_signals",
            "risks",
            "evidence_quality",
            "confidence",
            "rationale",
            "next_step",
        )

        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(
                f"Market Intelligence response missing fields: "
                f"{', '.join(missing)}"
            )

        evidence_quality = float(result["evidence_quality"])
        confidence = float(result["confidence"])

        if not 0 <= evidence_quality <= 100:
            raise ValueError("evidence_quality must be between 0 and 100")

        if not 0 <= confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")

        for field_name in (
            "demand_signals",
            "competition_signals",
            "buying_signals",
            "risks",
            "rationale",
        ):
            value = result[field_name]
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"{field_name} must be a list of strings")

        return MarketIntelligenceOutput(
            market=str(result["market"]),
            city=str(result["city"]),
            demand_signals=result["demand_signals"],
            competition_signals=result["competition_signals"],
            buying_signals=result["buying_signals"],
            risks=result["risks"],
            evidence_quality=evidence_quality,
            confidence=confidence,
            rationale=result["rationale"],
            next_step=str(result["next_step"]),
        )
