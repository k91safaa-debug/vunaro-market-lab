from dataclasses import dataclass
from typing import Any

import httpx


GOOGLE_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


@dataclass(frozen=True)
class GooglePlacesProvider:
    api_key: str
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")

    def search(
        self,
        *,
        city: str,
        country: str,
        cuisine: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not city.strip():
            raise ValueError("city must not be empty")
        if not country.strip():
            raise ValueError("country must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        query = f"restaurants in {city}, {country}"
        if cuisine and cuisine.strip():
            query = f"{cuisine.strip()} restaurants in {city}, {country}"

        payload = {
            "textQuery": query,
            "pageSize": min(limit, 20),
        }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "places.displayName,"
                "places.formattedAddress,"
                "places.location,"
                "places.primaryType,"
                "places.types"
            ),
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                GOOGLE_PLACES_SEARCH_URL,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        return data.get("places", [])[:limit]
