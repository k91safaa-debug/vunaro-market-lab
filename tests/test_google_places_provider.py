import httpx
import pytest

from backend.app.agents.google_places_provider import (
    GOOGLE_PLACES_SEARCH_URL,
    GooglePlacesProvider,
)


def test_google_places_provider_builds_expected_request(monkeypatch):
    captured = {}

    class MockClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers

            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "places": [
                        {
                            "displayName": {"text": "Test Italian Restaurant"},
                            "formattedAddress": "Toronto, Canada",
                            "primaryType": "italian_restaurant",
                        }
                    ]
                },
            )

    monkeypatch.setattr(
        "backend.app.agents.google_places_provider.httpx.Client",
        MockClient,
    )

    provider = GooglePlacesProvider(api_key="test-key", timeout=7.5)

    result = provider.search(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=3,
    )

    assert len(result) == 1
    assert result[0]["displayName"]["text"] == "Test Italian Restaurant"

    assert captured["timeout"] == 7.5
    assert captured["url"] == GOOGLE_PLACES_SEARCH_URL
    assert captured["json"]["textQuery"] == "Italian restaurants in Toronto, Canada"
    assert captured["json"]["pageSize"] == 3
    assert captured["headers"]["X-Goog-Api-Key"] == "test-key"
    assert "places.displayName" in captured["headers"]["X-Goog-FieldMask"]


def test_google_places_provider_respects_limit(monkeypatch):
    class MockClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *, json, headers):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "places": [
                        {"displayName": {"text": "One"}},
                        {"displayName": {"text": "Two"}},
                        {"displayName": {"text": "Three"}},
                        {"displayName": {"text": "Four"}},
                    ]
                },
            )

    monkeypatch.setattr(
        "backend.app.agents.google_places_provider.httpx.Client",
        MockClient,
    )

    provider = GooglePlacesProvider(api_key="test-key")

    result = provider.search(
        city="Toronto",
        country="Canada",
        cuisine=None,
        limit=2,
    )

    assert len(result) == 2
    assert result[0]["displayName"]["text"] == "One"
    assert result[1]["displayName"]["text"] == "Two"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"city": "", "country": "Canada", "cuisine": None, "limit": 5},
        {"city": "Toronto", "country": "", "cuisine": None, "limit": 5},
        {"city": "Toronto", "country": "Canada", "cuisine": None, "limit": 0},
    ],
)
def test_google_places_provider_rejects_invalid_input(kwargs):
    provider = GooglePlacesProvider(api_key="test-key")

    with pytest.raises(ValueError):
        provider.search(**kwargs)


def test_google_places_provider_rejects_empty_api_key():
    with pytest.raises(ValueError):
        GooglePlacesProvider(api_key="   ")
