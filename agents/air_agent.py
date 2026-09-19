"""
Agent 1 — Air Quality Agent
Fetches live AQI from OpenWeatherMap Air Pollution API.

Zones are mapped to specific Bengaluru lat/lon coordinates so each zone
gets a real, slightly-differentiated reading from the same city.

Zone 1 (Riverside Industrial) → Bellandur industrial corridor — typically
the worst AQI in the city due to effluent plants and ring-road traffic.

Shared JSON contract:
  {"signal": "air", "severity": "low|medium|high", "value": <us_aqi>, "note": "..."}
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OWM_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# ── Bengaluru zone coordinates ──────────────────────────────────────────────
# Each zone is pinned to a real Bengaluru location so live API readings
# reflect genuine micro-geographic variation within the city.
ZONE_COORDS = {
    # Riverside Industrial → Bellandur Lake / Varthur industrial belt
    "zone 1": (12.9141, 77.6432),
    # Central Market → KR Market / City Market, central Bengaluru
    "zone 2": (12.9634, 77.5760),
    # Lakeside Residential → Ulsoor Lake neighbourhood
    "zone 3": (12.9784, 77.6183),
    # Outer Ring Road → Marathahalli / ORR junction
    "zone 4": (12.9352, 77.6940),
}

# Fallback: generic Bengaluru city centre
DEFAULT_COORDS = (12.9716, 77.5946)

# OWM AQI (1–5) → approximate US AQI midpoints
OWM_TO_US_AQI = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}

POLLUTANT_NAMES = {
    "co":    "Carbon Monoxide (CO)",
    "no":    "Nitric Oxide (NO)",
    "no2":   "Nitrogen Dioxide (NO₂)",
    "o3":    "Ozone (O₃)",
    "so2":   "Sulfur Dioxide (SO₂)",
    "pm2_5": "Fine Particulate Matter (PM2.5)",
    "pm10":  "Particulate Matter (PM10)",
    "nh3":   "Ammonia (NH₃)",
}

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}


def _zone_to_coords(zone: str) -> tuple[float, float]:
    """
    Map a zone name to Bengaluru lat/lon.
    Matches on the zone number prefix (e.g. 'zone 1', 'zone 2').
    Falls back to city centre if no match.
    """
    zone_lower = zone.strip().lower()
    for key, coords in ZONE_COORDS.items():
        if key in zone_lower:
            return coords
    return DEFAULT_COORDS


def _geocode_city(city: str) -> tuple[float, float]:
    """Fallback: geocode an arbitrary city name via OWM Geocoding API."""
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": OWM_API_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"City not found: {city!r}")
    return data[0]["lat"], data[0]["lon"]


def _get_dominant_pollutant(components: dict) -> str:
    dominant = max(components, key=lambda k: components[k])
    return POLLUTANT_NAMES.get(dominant, dominant.upper())


def _severity_from_us_aqi(us_aqi: int) -> str:
    if us_aqi >= 300:
        return "high"
    elif us_aqi >= 150:
        return "medium"
    return "low"


def run(zone: str) -> dict:
    """
    Main entry point for the Air Quality Agent.

    Args:
        zone: Zone name — used to look up Bengaluru coordinates.
              Accepts the full zone string like 'Zone 1 - Riverside Industrial'
              or a plain city name as fallback (e.g. 'Bengaluru').

    Returns:
        Shared JSON contract dict.
    """
    if not OWM_API_KEY:
        raise EnvironmentError("OPENWEATHERMAP_API_KEY not set in .env file")

    # Resolve coordinates — zone-pinned first, city-geocode as fallback
    lat, lon = _zone_to_coords(zone)

    # Fetch live air pollution data
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    item       = data["list"][0]
    owm_aqi    = item["main"]["aqi"]        # 1 (Good) → 5 (Very Poor)
    components = item["components"]
    
    # Fetch live weather data (wind, temp) for deeper context
    weather_url = "http://api.openweathermap.org/data/2.5/weather"
    w_params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}
    w_resp = requests.get(weather_url, params=w_params, timeout=10)
    w_resp.raise_for_status()
    w_data = w_resp.json()
    
    temp = w_data["main"]["temp"]
    wind_speed = w_data["wind"]["speed"]
    wind_deg = w_data["wind"].get("deg", 0)
    
    def _wind_dir(deg):
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return dirs[int((deg / 45) + 0.5) % 8]
        
    wind_direction = _wind_dir(wind_deg)

    us_aqi    = OWM_TO_US_AQI[owm_aqi]
    severity  = _severity_from_us_aqi(us_aqi)
    dominant  = _get_dominant_pollutant(components)
    label     = AQI_LABELS[owm_aqi]

    # Use the zone name in the note if it looks like a zone string
    location_label = zone if zone.lower().startswith("zone") else f"Bengaluru ({zone})"

    note = (
        f"Air quality at {location_label} is {label} (AQI ≈ {us_aqi}). "
        f"Dominant pollutant: {dominant}. "
        f"Weather context: {temp}°C, Wind {wind_speed} m/s blowing {wind_direction}."
    )

    return {
        "signal":             "air",
        "severity":           severity,
        "value":              us_aqi,
        "note":               note,
        # Extra metadata
        "location_label":     location_label,
        "lat":                lat,
        "lon":                lon,
        "owm_aqi":            owm_aqi,
        "dominant_pollutant": dominant,
        "components":         components,
        "temperature_c":      temp,
        "wind_speed_ms":      wind_speed,
        "wind_direction":     wind_direction,
    }


if __name__ == "__main__":
    import json
    zones = [
        "Zone 1 - Riverside Industrial",
        "Zone 2 - Central Market",
        "Zone 3 - Lakeside Residential",
        "Zone 4 - Outer Ring Road",
    ]
    for z in zones:
        result = run(z)
        display = {k: v for k, v in result.items() if k != "components"}
        print(f"\n{z}:")
        print(json.dumps(display, indent=2))
