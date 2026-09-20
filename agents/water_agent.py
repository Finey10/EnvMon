"""
Agent 2 — Water Quality Agent
Reads from the shared team mock CSV (mock_zones.csv).
Zone → water_ph, water_turbidity, water_coliform

Shared JSON contract:
  {"signal": "water", "severity": "low|medium|high", "value": <turbidity>, "note": "..."}
"""

import pandas as pd
import hashlib
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "mock_zones.csv"

# All four canonical zone names for reference
ZONES = [
    "Zone 1 - Riverside Industrial",
    "Zone 2 - Central Market",
    "Zone 3 - Lakeside Residential",
    "Zone 4 - Outer Ring Road",
]


def _load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"mock_zones.csv not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Normalise the coliform column: accept true/false strings or booleans
    df["water_coliform"] = df["water_coliform"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def _generate_synthetic_water_data(city: str) -> pd.Series:
    """Deterministically generate realistic water data for an unknown city."""
    city_lower = city.strip().lower()
    
    # ── HACKATHON DEMO OVERRIDE ──
    # Force terrible water quality if the user manually types Bangalore
    if "bengaluru" in city_lower or "bangalore" in city_lower:
        return pd.Series({
            "zone": city,
            "city": city,
            "water_ph": 4.2,
            "water_turbidity": 9.5,
            "water_coliform": True
        })
        
    # Force perfect water quality if the user manually types Tokyo
    if "tokyo" in city_lower:
        return pd.Series({
            "zone": city,
            "city": city,
            "water_ph": 7.1,
            "water_turbidity": 0.8,
            "water_coliform": False
        })
        
    # Seed a hash with the city name so the same city always gets the same data
    city_hash = int(hashlib.md5(city_lower.encode()).hexdigest(), 16)
    
    # pH between 5.5 and 8.5
    ph = 5.5 + (city_hash % 30) / 10.0
    
    # Turbidity between 0.5 and 10.5
    turbidity = 0.5 + ((city_hash // 10) % 100) / 10.0
    
    # Coliform 20% chance
    coliform = (city_hash % 5) == 0
    
    return pd.Series({
        "zone": city,
        "city": city,
        "water_ph": round(ph, 1),
        "water_turbidity": round(turbidity, 1),
        "water_coliform": coliform
    })


def _match_zone(df: pd.DataFrame, zone: str) -> pd.Series:
    """Case-insensitive substring match; falls back to dynamic generation."""
    zone_lower = zone.strip().lower()
    mask = df["zone"].str.lower().str.contains(zone_lower, na=False)
    matches = df[mask]
    if not matches.empty:
        return matches.iloc[0]
        
    return _generate_synthetic_water_data(zone)


def _determine_severity(row: pd.Series) -> tuple[str, str]:
    """
    Severity rules (match original spec):
      high   → turbidity > 5 NTU  OR  coliform detected
      medium → turbidity 2–5 NTU
      low    → turbidity < 2 NTU  AND  no coliform
    Returns (severity, reason_note).
    """
    turbidity = float(row["water_turbidity"])
    coliform  = bool(row["water_coliform"])

    if coliform or turbidity > 5:
        reasons = []
        if turbidity > 5:
            reasons.append(f"turbidity {turbidity:.1f} NTU (> 5 threshold)")
        if coliform:
            reasons.append("coliform bacteria detected")
        return "high", " and ".join(reasons)
    elif turbidity >= 2:
        return "medium", f"turbidity {turbidity:.1f} NTU (2–5 range)"
    else:
        return "low", f"turbidity {turbidity:.1f} NTU, no coliform"


def _calculate_wqi(ph: float, turbidity: float, coliform: bool) -> int:
    """Calculate a synthetic Water Quality Index (0-100)."""
    score = 100
    # Penalty for pH deviating from 7.0
    score -= abs(7.0 - ph) * 10
    # Penalty for turbidity
    score -= turbidity * 5
    # Heavy penalty for coliform
    if coliform:
        score -= 40
    return int(max(0, min(100, score)))


def run(zone: str) -> dict:
    """
    Main entry point for the Water Quality Agent.

    Args:
        zone: Zone name to look up — one of:
              'Zone 1 - Riverside Industrial'
              'Zone 2 - Central Market'
              'Zone 3 - Lakeside Residential'
              'Zone 4 - Outer Ring Road'

    Returns:
        Shared JSON contract dict.
    """
    df = _load_data()
    row = _match_zone(df, zone)

    matched_zone = row["zone"]
    city         = row["city"]
    ph           = float(row["water_ph"])
    turbidity    = float(row["water_turbidity"])
    coliform     = bool(row["water_coliform"])

    severity, reason = _determine_severity(row)
    wqi = _calculate_wqi(ph, turbidity, coliform)

    note = (
        f"Water in {matched_zone} ({city}): pH {ph:.1f}, WQI Score: {wqi}/100, {reason}. "
        f"Severity assessed as {severity}."
    )

    return {
        "signal":          "water",
        "severity":        severity,
        "value":           round(turbidity, 2),
        "note":            note,
        # Extra metadata for dashboard display
        "matched_zone":    matched_zone,
        "city":            city,
        "ph":              ph,
        "turbidity_ntu":   turbidity,
        "coliform_detected": coliform,
        "wqi_score":       wqi,
    }


if __name__ == "__main__":
    import json
    for z in ZONES:
        result = run(z)
        print(f"\n{z}:")
        print(json.dumps({k: v for k, v in result.items()}, indent=2))
