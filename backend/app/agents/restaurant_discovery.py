from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class CandidateDiscoveryInput:
    city: str
    country: str
    cuisine: Optional[str]
    limit: int

    def __post_init__(self) -> None:
        if not self.city.strip():
            raise ValueError("city must not be empty")
        if not self.country.strip():
            raise ValueError("country must not be empty")
        if self.limit < 1:
            raise ValueError("limit must be at least 1")


@dataclass(frozen=True)
class RestaurantCandidate:
    name: str
    city: str
    country: str
    cuisine: Optional[str] = None


@dataclass(frozen=True)
class CandidateDiscoveryResult:
    candidates: list[RestaurantCandidate]


class CandidateDiscoveryProvider(Protocol):
    def search(self, request: CandidateDiscoveryInput) -> list[RestaurantCandidate]:
        ...


def _normalize_candidate(
    candidate: RestaurantCandidate,
    request: CandidateDiscoveryInput,
) -> Optional[RestaurantCandidate]:
    name = candidate.name.strip()
    city = candidate.city.strip()
    country = candidate.country.strip()
    cuisine = candidate.cuisine.strip() if candidate.cuisine else None

    if not name:
        return None

    if city.casefold() != request.city.casefold():
        return None

    if country.casefold() != request.country.casefold():
        return None

    return RestaurantCandidate(
        name=name,
        city=request.city,
        country=request.country,
        cuisine=cuisine,
    )


def discover_candidates(
    request: CandidateDiscoveryInput,
    provider: CandidateDiscoveryProvider,
) -> CandidateDiscoveryResult:
    raw_candidates = provider.search(request)

    normalized: list[RestaurantCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    for candidate in raw_candidates:
        normalized_candidate = _normalize_candidate(candidate, request)

        if normalized_candidate is None:
            continue

        identity = (
            normalized_candidate.name.casefold(),
            normalized_candidate.city.casefold(),
            normalized_candidate.country.casefold(),
        )

        if identity in seen:
            continue

        seen.add(identity)
        normalized.append(normalized_candidate)

        if len(normalized) >= request.limit:
            break

    return CandidateDiscoveryResult(candidates=normalized)
