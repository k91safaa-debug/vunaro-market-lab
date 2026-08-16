import inspect

from app.agents.market_strategist import MarketStrategist, MarketStrategistInput
from app.agents.market_intelligence import MarketIntelligence, MarketIntelligenceInput
from app.agents.restaurant_discovery import (
    CandidateDiscoveryInput,
    RestaurantCandidate,
    discover_candidates,
)


def test_agent_01_interfaces_are_integratable():
    # Core Agent 01 contracts must remain callable.
    assert hasattr(MarketStrategist, "analyze")
    assert hasattr(MarketIntelligence, "analyze")
    assert callable(discover_candidates)

    strategist_signature = inspect.signature(MarketStrategist.analyze)
    intelligence_signature = inspect.signature(MarketIntelligence.analyze)
    discovery_signature = inspect.signature(discover_candidates)

    assert len(strategist_signature.parameters) >= 1
    assert len(intelligence_signature.parameters) >= 1
    assert len(discovery_signature.parameters) >= 2

    # The three pipeline contracts must be constructible.
    strategist_fields = inspect.signature(MarketStrategistInput).parameters
    intelligence_fields = inspect.signature(MarketIntelligenceInput).parameters
    discovery_fields = inspect.signature(CandidateDiscoveryInput).parameters

    assert strategist_fields
    assert intelligence_fields
    assert {"city", "country", "cuisine", "limit"}.issubset(discovery_fields)

    # Discovery must accept a provider and return candidates.
    class IntegrationProvider:
        def search(self, request):
            assert request.city == "Toronto"
            assert request.country == "Canada"
            return [
                RestaurantCandidate(
                    name="VUNARO Test Restaurant",
                    city="Toronto",
                    country="Canada",
                    cuisine="Italian",
                )
            ]

    request = CandidateDiscoveryInput(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=10,
    )

    result = discover_candidates(request, IntegrationProvider())

    assert result.candidates
    assert result.candidates[0].name == "VUNARO Test Restaurant"
    assert result.candidates[0].city == "Toronto"
    assert result.candidates[0].country == "Canada"


def test_agent_01_pipeline_contract():
    """
    Contract-level integration:
    Market Strategist -> Market Intelligence -> Restaurant Discovery.

    This test intentionally does not call external APIs or LLMs.
    It verifies that all three Agent 01 modules expose stable,
    composable contracts before the production orchestrator is added.
    """
    strategist = MarketStrategist()
    intelligence = MarketIntelligence()

    assert callable(getattr(strategist, "analyze"))
    assert callable(getattr(intelligence, "analyze"))

    discovery_request = CandidateDiscoveryInput(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=10,
    )

    assert discovery_request.city == "Toronto"
    assert discovery_request.country == "Canada"
    assert discovery_request.cuisine == "Italian"
    assert discovery_request.limit == 10
