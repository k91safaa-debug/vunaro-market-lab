import pytest

from app.agents.opportunity_engine import (
    OpportunityEngine,
    OpportunityEngineInput,
)


def valid_result():
    return {
        "market": "Canada",
        "city": "Toronto",
        "opportunity_score": 82,
        "priority": "high",
        "opportunity_type": "market_entry",
        "reasons": ["Strong demand and viable market conditions"],
        "risks": ["Competitive market"],
        "recommended_action": "Build restaurant shortlist",
    }


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    def chat_json(self, messages, temperature=0.2, max_tokens=2048, max_attempts=2):
        return self.response


def test_opportunity_engine_valid_result():
    agent = OpportunityEngine(
        llm_client=FakeLLMClient(valid_result())
    )

    result = agent.analyze(
        OpportunityEngineInput(
            market="Canada",
            city="Toronto",
        )
    )

    assert result.market == "Canada"
    assert result.city == "Toronto"
    assert 0 <= result.opportunity_score <= 100
    assert result.recommended_action


def test_opportunity_engine_rejects_invalid_score():
    bad = valid_result()
    bad["opportunity_score"] = 150

    agent = OpportunityEngine(
        llm_client=FakeLLMClient(bad)
    )

    with pytest.raises(ValueError):
        agent.analyze(
            OpportunityEngineInput(
                market="Canada",
                city="Toronto",
            )
        )


def test_opportunity_engine_requires_required_fields():
    bad = valid_result()
    del bad["recommended_action"]

    agent = OpportunityEngine(
        llm_client=FakeLLMClient(bad)
    )

    with pytest.raises(ValueError):
        agent.analyze(
            OpportunityEngineInput(
                market="Canada",
                city="Toronto",
            )
        )
