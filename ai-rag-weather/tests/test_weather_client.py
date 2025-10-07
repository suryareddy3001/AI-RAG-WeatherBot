import pytest
from ai_rag_weather.weather.client import WeatherClient, WeatherResponse

class MockResponse:
    def __init__(self, status_code=200, json_data=None):
        self._status = status_code
        self._json = json_data or {}
    
    def raise_for_status(self):
        if self._status >= 400:
            raise Exception("HTTP error")
    
    def json(self):
        return self._json

@pytest.fixture
def mock_httpx_get(monkeypatch):
    def fake_get(client, url, params=None, timeout=None):
        print(f"Mocked httpx.Client.get called with url={url}, params={params}")
        if params and params.get("q") == "fail":
            raise Exception("HTTP error")
        return MockResponse(200, {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {"temp": 10, "feels_like": 8, "humidity": 80},
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 5}
        })
    monkeypatch.setattr("httpx.Client.get", fake_get)

def test_weather_success(mock_httpx_get):
    client = WeatherClient()
    resp = client.fetch("London")
    assert isinstance(resp, WeatherResponse)
    assert resp.city == "London"
    assert resp.country == "GB"
    assert resp.temp == 10
    assert resp.feels_like == 8
    assert resp.description == "cloudy"
    assert resp.humidity == 80
    assert resp.wind_speed == 5

def test_weather_fail(mock_httpx_get):
    client = WeatherClient()
    resp = client.fetch("fail")
    assert resp is None