import pytest

from app.agents.market_intelligence import (
    MarketIntelligence,
    MarketIntelligenceInput,
)


def valid_result():
    return {
        "market": "Canada",
        "city": "Toronto",
        "demand_signals": ["Strong restaurant density"],
        "competition_signals": ["Existing restaurant SaaS providers"],
        "buying_signals": ["Multi-location operators"],
        "risks": ["Competitive market"],
        "evidence_quality": 85,
        "confidence": 80,
        "rationale": ["Evidence supports further investigation"],
        "next_step": "Build restaurant shortlist",
    }


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    def chat_json(self, **kwargs):
        return self.response


def test_market_intelligence_parses_valid_response():
    agent = MarketIntelligence(
        llm_client=FakeLLMClient(valid_result())
    )

    result = agent.analyze(
        MarketIntelligenceInput(
            market="Canada",
            city="Toronto",
        )
    )

    assert result.market == "Canada"
    assert result.city == "Toronto"
    assert result.evidence_quality == 85
    assert result.confidence == 80
    assert result.next_step == "Build restaurant shortlist"


def test_missing_required_field_raises():
    response = valid_result()
    response.pop("confidence")

    agent = MarketIntelligence(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(ValueError, match="missing fields"):
        agent.analyze(
            MarketIntelligenceInput(
                market="Canada",
                city="Toronto",
            )
        )


def test_invalid_score_raises():
    response = valid_result()
    response["confidence"] = 101

    agent = MarketIntelligence(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(
        ValueError, match="confidence must be between 0 and 100"
    ):
        agent.analyze(
            MarketIntelligenceInput(
                market="Canada",
                city="Toronto",
            )
        )


def test_invalid_signal_type_raises():
    response = valid_result()
    response["demand_signals"] = "not-a-list"

    agent = MarketIntelligence(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(
        ValueError, match="demand_signals must be a list of strings"
    ):
        agent.analyze(
            MarketIntelligenceInput(
                market="Canada",
                city="Toronto",
            )
        )


def test_all_scores_are_bounded():
    response = valid_result()
    response["evidence_quality"] = 0
    response["confidence"] = 100

    agent = MarketIntelligence(
        llm_client=FakeLLMClient(response)
    )

    result = agent.analyze(
        MarketIntelligenceInput(
            market="Canada",
            city="Toronto",
        )
    )

    assert 0 <= result.evidence_quality <= 100
    assert 0 <= result.confidence <= 100
