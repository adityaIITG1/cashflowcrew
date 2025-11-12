
# weather.py — Open-Meteo (no API key) + spending hints

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import requests

# Endpoints
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_TIMEOUT = 8  # seconds
# city -> (timestamp, payload)
_CACHE: Dict[str, Tuple[float, dict]] = {}

# Minimal WMO/Open-Meteo weather code mapping
# Ref: https://open-meteo.com/en/docs
_WMO_MAP = {
    0: ("clear", "Clear sky", "☀️"),
    1: ("mainly_clear", "Mainly clear", "🌤️"),
    2: ("partly_cloudy", "Partly cloudy", "⛅"),
    3: ("overcast", "Overcast", "☁️"),
    45: ("fog", "Fog", "🌫️"),
    48: ("depositing_rime_fog", "Freezing fog", "🌫️"),
    51: ("drizzle_light", "Light drizzle", "🌦️"),
    53: ("drizzle_moderate", "Drizzle", "🌦️"),
    55: ("drizzle_dense", "Dense drizzle", "🌧️"),
    56: ("freezing_drizzle_light", "Light freezing drizzle", "🌧️"),
    57: ("freezing_drizzle_dense", "Freezing drizzle", "🌧️"),
    61: ("rain_slight", "Light rain", "🌧️"),
    63: ("rain_moderate", "Rain", "🌧️"),
    65: ("rain_heavy", "Heavy rain", "🌧️"),
    66: ("freezing_rain_light", "Light freezing rain", "🌧️"),
    67: ("freezing_rain_heavy", "Freezing rain", "🌧️"),
    71: ("snow_fall_slight", "Light snow", "❄️"),
    73: ("snow_fall_moderate", "Snow", "❄️"),
    75: ("snow_fall_heavy", "Heavy snow", "❄️"),
    77: ("snow_grains", "Snow grains", "❄️"),
    80: ("rain_showers_slight", "Light rain showers", "🌦️"),
    81: ("rain_showers_moderate", "Rain showers", "🌦️"),
    82: ("rain_showers_violent", "Violent rain showers", "⛈️"),
    85: ("snow_showers_slight", "Light snow showers", "🌨️"),
    86: ("snow_showers_heavy", "Heavy snow showers", "🌨️"),
    95: ("thunderstorm_slight", "Thunderstorm", "⛈️"),
    96: ("thunderstorm_hail_slight", "Thunderstorm with hail", "⛈️"),
    99: ("thunderstorm_hail_heavy", "Severe thunderstorm with hail", "⛈️"),
}


def _normalize_city(city: str) -> str:
    return (city or "").strip() or "Prayagraj"


def _geocode_city(city: str) -> Optional[dict]:
    try:
        r = requests.get(
            _GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            return None
        first = results[0]
        return {
            "city": first.get("name") or city,
            "lat": float(first.get("latitude")),
            "lon": float(first.get("longitude")),
            "admin1": first.get("admin1") or "",
            "country": first.get("country") or "",
        }
    except requests.RequestException:
        return None


def _fetch_forecast(lat: float, lon: float, tz: str = "auto") -> Optional[dict]:
    # Pull current + a small hourly slice for precipitation context
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": ["precipitation", "weather_code"],
        "timezone": tz,
        "past_days": 0,
    }
    # Open-Meteo accepts comma-separated values for arrays
    params["hourly"] = ",".join(params["hourly"])
    try:
        r = requests.get(_FORECAST_URL, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json() or {}
    except requests.RequestException:
        return None


def _normalize_payload(geo: dict, raw: dict) -> Optional[dict]:
    if not raw or "current_weather" not in raw:
        return None

    cw = raw["current_weather"] or {}
    code = int(cw.get("weathercode") or 0)
    temp_c = float(cw.get("temperature") or 0.0)
    wind_kph = float(cw.get("windspeed") or 0.0)  # already km/h
    w = _WMO_MAP.get(code, ("unknown", "Unknown", "🌡️"))
    condition, description, icon = w

    # Approx precip for the current hour (best-effort)
    rain_mm = 0.0
    hourly = raw.get("hourly") or {}
    times = hourly.get("time") or []
    prec = hourly.get("precipitation") or []
    # Find the last entry if arrays align; avoid index errors
    if times and prec and len(times) == len(prec):
        try:
            rain_mm = float(prec[-1] or 0.0)
        except Exception:
            rain_mm = 0.0

    return {
        "city": geo.get("city"),
        "lat": geo.get("lat"),
        "lon": geo.get("lon"),
        "condition": condition,           # machine-friendly
        "description": description,       # human-friendly
        "icon": icon,
        "weather_code": code,
        "temp_c": round(temp_c, 1),
        "wind_kph": round(wind_kph, 1),
        "rain_mm": round(rain_mm, 2),
    }


def get_weather(city: str, *, ttl_minutes: int = 30) -> Optional[dict]:
    """
    Fetch normalized current weather for a city using Open-Meteo.
    Returns dict or None. Caches by city for ttl_minutes.
    """
    city = _normalize_city(city)
    now = time.time()
    ttl = max(1, int(ttl_minutes)) * 60

    cached = _CACHE.get(city)
    if cached and (now - cached[0] < ttl):
        return cached[1]

    geo = _geocode_city(city)
    if not geo:
        return None

    raw = _fetch_forecast(geo["lat"], geo["lon"])
    data = _normalize_payload(geo, raw) if raw else None
    if data:
        _CACHE[city] = (now, data)
    return data


def spend_mood_hint(data: Optional[dict]) -> str:
    """
    Turn weather into a short savings-minded hint.
    Always returns a user-friendly string (no exceptions).
    """
    if not data:
        return "Weather data unavailable. Plan your spending based on your calendar instead! 🗓️"

    cond = (data.get("condition") or "unknown").lower()
    t = float(data.get("temp_c") or 0.0)
    rain = float(data.get("rain_mm") or 0.0)
    wind = float(data.get("wind_kph") or 0.0)
    city = data.get("city") or "Your city"

    # Strong signals first
    if cond.startswith("thunderstorm") or data.get("weather_code") in (95, 96, 99):
        return f"{city}: Stormy—avoid impulse rides/food delivery; batch errands and cook at home. ⛈️🍳"
    if rain >= 2.0 or cond.startswith("rain") or "drizzle" in cond:
        return f"{city}: Rainy—combine trips, skip taxis, and brew at home to save. ☔️🚶"
    if "snow" in cond:
        return f"{city}: Snowy—plan one weekly shop, avoid multiple delivery fees. ❄️🛒"
    if 18 <= t <= 30 and cond in {"clear", "mainly_clear", "partly_cloudy"}:
        return f"{city}: Perfect weather—opt for free activities; leave the cab and walk. 🌤️🚶"
    if t > 34:
        return f"{city}: Hot—carry water & snacks; dodge pricey cold drinks. 🥵💧"
    if t < 10:
        return f"{city}: Cold—meal-prep warm dishes; skip takeout splurges. 🧣🍲"
    if wind >= 35:
        return f"{city}: Windy—delay non-urgent trips; consolidate to save fares. 🌬️🚌"

    return f"{city}: Steady weather—stick to essentials and batch errands. 🧾✅"
