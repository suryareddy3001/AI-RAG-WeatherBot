import httpx
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ai_rag_weather.config import get_settings
from ..logging import get_logger

"""Weather API client for retrieving weather and geocoding data.

This module provides a WeatherClient class that interfaces with the OpenWeatherMap API
to fetch current weather data for a given city and search for city information using
geocoding. It includes retry logic for handling network errors and uses Pydantic for
structured data validation.
"""

logger = get_logger(__name__)

class WeatherResponse(BaseModel):
    """Pydantic model for structuring weather API responses.

    Attributes:
        city: Name of the city.
        country: Country code of the city.
        temp: Current temperature in the specified units.
        feels_like: Perceived temperature in the specified units.
        description: Weather condition description.
        humidity: Humidity percentage.
        wind_speed: Wind speed in the specified units.
        raw: Full raw response from the API.
    """
    city: str
    country: str
    temp: float
    feels_like: float
    description: str
    humidity: int
    wind_speed: float
    raw: Dict[str, Any]

_network_errors = (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError, httpx.RemoteProtocolError)

class WeatherClient:
    """Client for interacting with the OpenWeatherMap API.

    Initializes an HTTP client with configuration settings and provides methods to
    fetch weather data and search for cities.
    """
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.api_key = self.settings.OPENWEATHER_API_KEY
        self.units = self.settings.OPENWEATHER_UNITS
        self.client = httpx.Client(timeout=5.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5),
        retry=retry_if_exception_type(_network_errors),
        reraise=True
    )
    def fetch(self, city: str) -> Optional[WeatherResponse]:
        """Fetch current weather data for a specified city.

        Args:
            city: The name of the city to query weather for.

        Returns:
            A WeatherResponse object containing weather data, or None if the request fails.
        """
        params = {"q": city, "appid": self.api_key, "units": self.units}
        try:
            resp = self.client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return WeatherResponse(
                city=data["name"],
                country=data["sys"]["country"],
                temp=data["main"]["temp"],
                feels_like=data["main"]["feels_like"],
                description=data["weather"][0]["description"],
                humidity=data["main"]["humidity"],
                wind_speed=data["wind"]["speed"],
                raw=data,
            )
        except httpx.HTTPStatusError as e:
            logger.error("Weather API error", status_code=e.response.status_code, detail=str(e))
            return None
        except Exception as e:
            logger.error("Weather API exception", detail=str(e))
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5),
        retry=retry_if_exception_type(_network_errors),
        reraise=True
    )
    def search_cities(self, city_query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search for cities matching a query using the geocoding API.

        Args:
            city_query: The city name or partial name to search for.
            limit: Maximum number of results to return (default: 3).

        Returns:
            A list of dictionaries containing city information (name, country, lat, lon),
            or an empty list if the request fails.
        """
        geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city_query, "limit": limit, "appid": self.api_key}
        try:
            resp = self.client.get(geo_url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("Geocoding API error", status_code=e.response.status_code, detail=str(e))
            return []
        except Exception as e:
            logger.error("Geocoding API exception", detail=str(e))
            return []
