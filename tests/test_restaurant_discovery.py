from backend.app.agents.restaurant_discovery import (
    CandidateDiscoveryInput,
    RestaurantCandidate,
    discover_candidates,
)


class FakeDiscoveryProvider:
    def __init__(self, candidates):
        self.candidates = candidates

    def search(self, request):
        return self.candidates


def test_candidate_discovery_returns_candidates():
    provider = FakeDiscoveryProvider(
        [
            RestaurantCandidate(
                name="North Star",
                city="Toronto",
                country="Canada",
                cuisine="Italian",
            ),
            RestaurantCandidate(
                name="Maple Table",
                city="Toronto",
                country="Canada",
                cuisine="French",
            ),
        ]
    )

    result = discover_candidates(
        CandidateDiscoveryInput(
            city="Toronto",
            country="Canada",
            cuisine=None,
            limit=10,
        ),
        provider,
    )

    assert result.candidates
    assert len(result.candidates) == 2
    assert all(isinstance(c, RestaurantCandidate) for c in result.candidates)


def test_candidate_has_required_identity_fields():
    provider = FakeDiscoveryProvider(
        [
            RestaurantCandidate(
                name="North Star",
                city="Toronto",
                country="Canada",
                cuisine="Italian",
            )
        ]
    )

    result = discover_candidates(
        CandidateDiscoveryInput(
            city="Toronto",
            country="Canada",
            cuisine=None,
            limit=5,
        ),
        provider,
    )

    candidate = result.candidates[0]

    assert candidate.name == "North Star"
    assert candidate.city == "Toronto"
    assert candidate.country == "Canada"


def test_candidate_discovery_respects_limit():
    provider = FakeDiscoveryProvider(
        [
            RestaurantCandidate(
                name="One",
                city="Toronto",
                country="Canada",
            ),
            RestaurantCandidate(
                name="Two",
                city="Toronto",
                country="Canada",
            ),
            RestaurantCandidate(
                name="Three",
                city="Toronto",
                country="Canada",
            ),
            RestaurantCandidate(
                name="Four",
                city="Toronto",
                country="Canada",
            ),
        ]
    )

    result = discover_candidates(
        CandidateDiscoveryInput(
            city="Toronto",
            country="Canada",
            cuisine="Italian",
            limit=3,
        ),
        provider,
    )

    assert len(result.candidates) == 3


def test_candidate_discovery_rejects_invalid_limit():
    try:
        CandidateDiscoveryInput(
            city="Toronto",
            country="Canada",
            cuisine=None,
            limit=0,
        )
    except ValueError:
        return

    raise AssertionError("limit=0 must be rejected")


def test_candidate_discovery_removes_duplicates():
    provider = FakeDiscoveryProvider(
        [
            RestaurantCandidate(
                name="North Star",
                city="Toronto",
                country="Canada",
            ),
            RestaurantCandidate(
                name=" north star ",
                city="Toronto",
                country="Canada",
            ),
        ]
    )

    result = discover_candidates(
        CandidateDiscoveryInput(
            city="Toronto",
            country="Canada",
            cuisine=None,
            limit=10,
        ),
        provider,
    )

    assert len(result.candidates) == 1


def test_candidate_discovery_rejects_wrong_location():
    provider = FakeDiscoveryProvider(
        [
            RestaurantCandidate(
                name="Toronto Restaurant",
                city="Toronto",
                country="Canada",
            ),
            RestaurantCandidate(
                name="Wrong City Restaurant",
                city="Montreal",
                country="Canada",
            ),
        ]
    )

    result = discover_candidates(
        CandidateDiscoveryInput(
            city="Toronto",
            country="Canada",
            cuisine=None,
            limit=10,
        ),
        provider,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].name == "Toronto Restaurant"
