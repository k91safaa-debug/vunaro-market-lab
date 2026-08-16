import pytest

from app.agents.google_places_provider import (
    GooglePlacesProvider,
    GOOGLE_PLACES_SEARCH_URL,
)


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeClient:
    last_request = None
    response_data = {"places": []}

    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, json, headers):
        FakeClient.last_request = {
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": self.timeout,
        }
        return FakeResponse(FakeClient.response_data)


def test_provider_rejects_empty_api_key():
    with pytest.raises(ValueError, match="api_key"):
        GooglePlacesProvider(api_key="")


def test_provider_rejects_invalid_timeout():
    with pytest.raises(ValueError, match="timeout"):
        GooglePlacesProvider(api_key="test-key", timeout=0)


def test_provider_search_rejects_invalid_input(monkeypatch):
    provider = GooglePlacesProvider(api_key="test-key")

    with pytest.raises(ValueError, match="city"):
        provider.search(city="", country="Canada", cuisine="Italian", limit=10)

    with pytest.raises(ValueError, match="country"):
        provider.search(city="Toronto", country="", cuisine="Italian", limit=10)

    with pytest.raises(ValueError, match="limit"):
        provider.search(city="Toronto", country="Canada", cuisine="Italian", limit=0)


def test_provider_builds_google_places_request_with_cuisine(monkeypatch):
    FakeClient.last_request = None
    FakeClient.response_data = {
        "places": [
            {
                "displayName": {"text": "Test Italian Restaurant"},
            }
        ]
    }

    monkeypatch.setattr(
        "app.agents.google_places_provider.httpx.Client",
        FakeClient,
    )

    provider = GooglePlacesProvider(api_key="test-key", timeout=7.5)

    result = provider.search(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=5,
    )

    assert result == [
        {
            "displayName": {"text": "Test Italian Restaurant"},
        }
    ]

    request = FakeClient.last_request

    assert request["url"] == GOOGLE_PLACES_SEARCH_URL
    assert request["json"] == {
        "textQuery": "Italian restaurants in Toronto, Canada",
        "pageSize": 5,
    }
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["X-Goog-Api-Key"] == "test-key"
    assert "places.displayName" in request["headers"]["X-Goog-FieldMask"]
    assert request["timeout"] == 7.5


def test_provider_builds_google_places_request_without_cuisine(monkeypatch):
    FakeClient.last_request = None
    FakeClient.response_data = {"places": [{"name": "Restaurant A"}]}

    monkeypatch.setattr(
        "app.agents.google_places_provider.httpx.Client",
        FakeClient,
    )

    provider = GooglePlacesProvider(api_key="test-key")

    result = provider.search(
        city="Toronto",
        country="Canada",
        cuisine=None,
        limit=10,
    )

    assert result == [{"name": "Restaurant A"}]
    assert FakeClient.last_request["json"] == {
        "textQuery": "restaurants in Toronto, Canada",
        "pageSize": 10,
    }


def test_provider_returns_empty_list_when_google_returns_no_places(monkeypatch):
    FakeClient.last_request = None
    FakeClient.response_data = {}

    monkeypatch.setattr(
        "app.agents.google_places_provider.httpx.Client",
        FakeClient,
    )

    provider = GooglePlacesProvider(api_key="test-key")

    result = provider.search(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=10,
    )

    assert result == []


def test_provider_respects_limit(monkeypatch):
    FakeClient.last_request = None
    FakeClient.response_data = {
        "places": [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"},
        ]
    }

    monkeypatch.setattr(
        "app.agents.google_places_provider.httpx.Client",
        FakeClient,
    )

    provider = GooglePlacesProvider(api_key="test-key")

    result = provider.search(
        city="Toronto",
        country="Canada",
        cuisine="Italian",
        limit=2,
    )

    assert len(result) == 2
    assert result == [{"name": "A"}, {"name": "B"}]
