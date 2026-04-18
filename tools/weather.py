from langchain_core.tools import tool
import os
import requests


@tool
def weather(city: str) -> str:
    """
    Get current weather for a city.
    Uses OpenWeatherMap API if OPENWEATHER_API_KEY is set,
    otherwise returns a mock response for demo purposes.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        # Mock response so the demo works without an API key
        mock_data = {
            "mumbai": "Mumbai: 32°C, Humid, Partly Cloudy",
            "delhi": "Delhi: 28°C, Clear Sky",
            "new york": "New York: 18°C, Overcast",
            "london": "London: 12°C, Rainy",
            "paris": "Paris: 15°C, Sunny",
        }
        return mock_data.get(city.lower(), f"{city}: 25°C, Partly Cloudy (mock data)")

    url = "https://api.openweathermap.org/data/2.5/weather"
    resp = requests.get(url, params={"q": city, "appid": api_key, "units": "metric"}, timeout=10)
    if resp.status_code != 200:
        return f"Could not fetch weather for {city}: {resp.text}"
    data = resp.json()
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"{city}: {temp}°C, {desc.title()}"


def get_weather_tool():
    return weather
