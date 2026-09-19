"""
Agent 2 — Water Quality Agent
Reads from the shared team mock CSV (mock_zones.csv).
Zone → water_ph, water_turbidity, water_coliform

Shared JSON contract:
  {"signal": "water", "severity": "low|medium|high", "value": <turbidity>, "note": "..."}
"""

import pandas as pd
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


def _match_zone(df: pd.DataFrame, zone: str) -> pd.Series:
    """Case-insensitive substring match; falls back to first row if no match."""
    zone_lower = zone.strip().lower()
    mask = df["zone"].str.lower().str.contains(zone_lower, na=False)
    matches = df[mask]
    if not matches.empty:
        return matches.iloc[0]
    return df.iloc[0]  # fallback


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

    note = (
        f"Water in {matched_zone} ({city}): pH {ph:.1f}, {reason}. "
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
    }


if __name__ == "__main__":
    import json
    for z in ZONES:
        result = run(z)
        print(f"\n{z}:")
        print(json.dumps({k: v for k, v in result.items()}, indent=2))
