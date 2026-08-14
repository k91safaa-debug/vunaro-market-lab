import pytest

from app.agents.icp_engine import (
    ICPEngine,
    ICPEngineInput,
)


def valid_result():
    return {
        "restaurant_name": "Example Restaurant",
        "market": "Canada",
        "city": "Toronto",
        "restaurant_type_fit": 90,
        "operational_complexity": 85,
        "communication_dependency": 90,
        "booking_guest_volume": 80,
        "automation_opportunity": 95,
        "ability_to_pay": 85,
        "growth_signals": 80,
        "product_fit": 95,
        "reasons": [
            "High guest communication volume",
            "Strong automation opportunity",
        ],
        "disqualifiers": [],
        "recommended_action": "Prioritize for outreach",
    }


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    def chat_json(
        self,
        messages,
        temperature=0.2,
        max_tokens=2048,
        max_attempts=2,
    ):
        return self.response


def test_icp_engine_valid_result():
    agent = ICPEngine(
        llm_client=FakeLLMClient(valid_result())
    )

    result = agent.analyze(
        ICPEngineInput(
            restaurant_name="Example Restaurant",
            market="Canada",
            city="Toronto",
            restaurant_type="Upscale restaurant",
            operational_complexity="High",
            communication_dependency="High",
            booking_guest_volume="High",
            automation_opportunity="High",
            ability_to_pay="Strong",
            growth_signals="Strong",
            product_fit="Excellent",
        )
    )

    assert result.restaurant_name == "Example Restaurant"
    assert result.market == "Canada"
    assert result.city == "Toronto"
    assert 0 <= result.icp_score <= 100
    assert result.tier == "A"
    assert result.priority == "VERY_HIGH"
    assert result.component_scores["product_fit"] == 95
    assert result.recommended_action == "Prioritize for outreach"


def test_icp_engine_calculates_weighted_score():
    result = valid_result()

    agent = ICPEngine(
        llm_client=FakeLLMClient(result)
    )

    output = agent.analyze(
        ICPEngineInput(
            restaurant_name="Example Restaurant",
            market="Canada",
            city="Toronto",
        )
    )

    expected = (
        90 * 15 / 100
        + 85 * 15 / 100
        + 90 * 15 / 100
        + 80 * 15 / 100
        + 95 * 15 / 100
        + 85 * 10 / 100
        + 80 * 10 / 100
        + 95 * 5 / 100
    )

    assert output.icp_score == round(expected, 2)


def test_icp_engine_rejects_invalid_component_score():
    bad = valid_result()
    bad["automation_opportunity"] = 150

    agent = ICPEngine(
        llm_client=FakeLLMClient(bad)
    )

    with pytest.raises(ValueError):
        agent.analyze(
            ICPEngineInput(
                restaurant_name="Example Restaurant",
                market="Canada",
                city="Toronto",
            )
        )


def test_icp_engine_requires_required_fields():
    bad = valid_result()
    del bad["recommended_action"]

    agent = ICPEngine(
        llm_client=FakeLLMClient(bad)
    )

    with pytest.raises(ValueError):
        agent.analyze(
            ICPEngineInput(
                restaurant_name="Example Restaurant",
                market="Canada",
                city="Toronto",
            )
        )


def test_icp_engine_rejects_invalid_reasons():
    bad = valid_result()
    bad["reasons"] = "not-a-list"

    agent = ICPEngine(
        llm_client=FakeLLMClient(bad)
    )

    with pytest.raises(ValueError):
        agent.analyze(
            ICPEngineInput(
                restaurant_name="Example Restaurant",
                market="Canada",
                city="Toronto",
            )
        )


def test_icp_engine_rejects_invalid_disqualifiers():
    bad = valid_result()
    bad["disqualifiers"] = "not-a-list"

    agent = ICPEngine(
        llm_client=FakeLLMClient(bad)
    )

    with pytest.raises(ValueError):
        agent.analyze(
            ICPEngineInput(
                restaurant_name="Example Restaurant",
                market="Canada",
                city="Toronto",
            )
        )


def test_icp_engine_rejects_disqualified_restaurant():
    result = valid_result()
    result["disqualifiers"] = [
        "No meaningful guest communication"
    ]

    agent = ICPEngine(
        llm_client=FakeLLMClient(result)
    )

    output = agent.analyze(
        ICPEngineInput(
            restaurant_name="Example Restaurant",
            market="Canada",
            city="Toronto",
        )
    )

    assert output.tier == "REJECT"
    assert output.priority == "LOW"
