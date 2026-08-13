from app.agents.market_strategist import (
    MarketStrategist,
    MarketStrategistInput,
)


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def valid_response():
    return {
        "recommended_market": "Canada",
        "recommended_city": "Toronto",
        "beachhead": "Independent and small multi-location restaurants",
        "icp": "Owner-led restaurants with high inbound booking volume",
        "opportunity_score": 82,
        "competitive_pressure": 61,
        "rationale": [
            "Strong restaurant density",
            "Clear need for multilingual guest communication",
        ],
        "expansion_path": [
            "10 restaurants",
            "20 restaurants",
            "50 restaurants",
            "100 restaurants",
            "500 restaurants",
        ],
    }


def test_market_strategist_returns_structured_output():
    fake_llm = FakeLLMClient(valid_response())
    agent = MarketStrategist(llm_client=fake_llm)

    result = agent.analyze(
        MarketStrategistInput(
            market="Canada",
            constraints={"initial_target": 10},
        )
    )

    assert result.recommended_market == "Canada"
    assert result.recommended_city == "Toronto"
    assert result.opportunity_score == 82
    assert result.competitive_pressure == 61
    assert len(result.rationale) == 2
    assert len(result.expansion_path) == 5
    assert len(fake_llm.calls) == 1


def test_market_strategist_rejects_missing_fields():
    fake_llm = FakeLLMClient(
        {
            "recommended_market": "Canada",
        }
    )
    agent = MarketStrategist(llm_client=fake_llm)

    try:
        agent.analyze(MarketStrategistInput(market="Canada"))
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_market_strategist_rejects_invalid_score():
    response = valid_response()
    response["opportunity_score"] = 101

    fake_llm = FakeLLMClient(response)
    agent = MarketStrategist(llm_client=fake_llm)

    try:
        agent.analyze(MarketStrategistInput(market="Canada"))
    except ValueError as exc:
        assert "opportunity_score" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
