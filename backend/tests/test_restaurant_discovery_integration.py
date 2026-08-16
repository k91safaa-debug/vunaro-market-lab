from app.agents.restaurant_discovery import (
    CandidateDiscoveryInput,
    RestaurantCandidate,
    discover_candidates,
)


class FakeProvider:
    def __init__(self, candidates):
        self.candidates = candidates
        self.received_request = None

    def search(self, request):
        self.received_request = request
        return self.candidates


def test_discover_candidates_normalizes_filters_deduplicates_and_limits():
    provider = FakeProvider(
        [
            RestaurantCandidate(
                name="  Bella Italia  ",
                city="Toronto",
                country="Canada",
                cuisine="Italian",
            ),
            RestaurantCandidate(
                name="Bella Italia",
                city="Toronto",
                country="Canada",
                cuisine="Italian",
            ),
            RestaurantCandidate(
                name="Wrong City",
                city="Montreal",
                country="Canada",
                cuisine="Italian",
            ),
            RestaurantCandidate(
                name="Wrong Country",
                city="Toronto",
                country="USA",
                cuisine="Italian",
            ),
            RestaurantCandidate(
                name="Casa Roma",
                city="Toronto",
                country="Canada",
                cuisine=None,
            ),
        ]
    )

    request = CandidateDiscoveryInput(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=2,
    )

    result = discover_candidates(request, provider)

    assert provider.received_request == request
    assert len(result.candidates) == 2

    assert result.candidates[0].name == "Bella Italia"
    assert result.candidates[0].city == "Toronto"
    assert result.candidates[0].country == "Canada"
    assert result.candidates[0].cuisine == "Italian"

    assert result.candidates[1].name == "Casa Roma"
    assert result.candidates[1].city == "Toronto"
    assert result.candidates[1].country == "Canada"


def test_discover_candidates_returns_empty_when_provider_has_no_matches():
    provider = FakeProvider(
        [
            RestaurantCandidate(
                name="Montreal Bistro",
                city="Montreal",
                country="Canada",
                cuisine="Italian",
            ),
            RestaurantCandidate(
                name="New York Pizza",
                city="New York",
                country="USA",
                cuisine="Italian",
            ),
        ]
    )

    request = CandidateDiscoveryInput(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=10,
    )

    result = discover_candidates(request, provider)

    assert result.candidates == []
