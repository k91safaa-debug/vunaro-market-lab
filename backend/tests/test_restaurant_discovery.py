import pytest

from app.agents.restaurant_discovery import CandidateDiscoveryInput


def test_candidate_discovery_input_valid():
    request = CandidateDiscoveryInput(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=10,
    )

    assert request.city == "Toronto"
    assert request.country == "Canada"
    assert request.cuisine == "Italian"
    assert request.limit == 10


def test_candidate_discovery_rejects_empty_city():
    with pytest.raises(ValueError, match="city"):
        CandidateDiscoveryInput(
            city="",
            country="Canada",
            cuisine="Italian",
            limit=10,
        )


def test_candidate_discovery_rejects_empty_country():
    with pytest.raises(ValueError, match="country"):
        CandidateDiscoveryInput(
            city="Toronto",
            country="",
            cuisine="Italian",
            limit=10,
        )


def test_candidate_discovery_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit"):
        CandidateDiscoveryInput(
            city="Toronto",
            country="Canada",
            cuisine="Italian",
            limit=0,
        )
