"""
Agent 4 — Coordinator Agent (powered by Groq / llama-3.3-70b-versatile)
Fuses the three specialist signals into one reasoned risk verdict using an LLM.

Input:  three dicts from air_agent, water_agent, litter_agent
Output: {"overall_risk": "low|medium|high", "reasoning": "...", "recommended_action": "..."}
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are an environmental risk analyst. You will receive JSON readings from three independent monitoring agents:
1. Air Quality Agent — AQI-based air pollution measurement
2. Water Quality Agent — turbidity, pH, and coliform data
3. Litter Detection Agent — object detection count from a field image

Your task is to synthesize these signals and produce a holistic environmental risk assessment.

IMPORTANT ANALYSIS RULES:
- Do NOT simply average the severity scores. Reason about correlations.
- If multiple signals are degraded in the same area, identify whether they share a common cause (e.g., industrial runoff, waste dumping, urban pollution).
- A single "high" signal from any agent should raise overall risk to at least "medium".
- Two or more "high" signals, or one "high" with clear correlation evidence, should result in overall risk "high".
- Identify the most urgent intervention needed.

Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{
  "overall_risk": "low|medium|high",
  "reasoning": "2-4 sentences explaining the cross-signal reasoning and any correlations identified",
  "recommended_action": "One concrete, actionable next step for environmental authorities"
}"""


def _build_user_prompt(air: dict, water: dict, litter: dict) -> str:
    # Sanitize — only pass fields relevant for reasoning (no raw box coordinates)
    air_summary = {
        "signal": air["signal"],
        "severity": air["severity"],
        "value_us_aqi": air["value"],
        "note": air["note"],
    }
    water_summary = {
        "signal": water["signal"],
        "severity": water["severity"],
        "turbidity_ntu": water["value"],
        "ph": water.get("ph"),
        "coliform_detected": water.get("coliform_detected"),
        "note": water["note"],
    }
    litter_summary = {
        "signal": litter["signal"],
        "severity": litter["severity"],
        "object_count": litter["value"],
        "note": litter["note"],
    }

    return (
        "Here are the three agent readings for the monitored area:\n\n"
        f"AIR QUALITY:\n{json.dumps(air_summary, indent=2)}\n\n"
        f"WATER QUALITY:\n{json.dumps(water_summary, indent=2)}\n\n"
        f"LITTER DETECTION:\n{json.dumps(litter_summary, indent=2)}\n\n"
        "Produce the risk assessment JSON now."
    )


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from the LLM response string."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Fallback: find JSON block with regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not extract JSON from LLM response:\n{text}")


def _fallback_synthesis(air: dict, water: dict, litter: dict) -> dict:
    severities = [air.get("severity", "low"), water.get("severity", "low"), litter.get("severity", "low")]
    high_count = severities.count("high")
    med_count = severities.count("medium")
    
    if high_count >= 2:
        risk = "high"
        reasoning = f"Critical multi-signal alert: Multiple monitoring agents report HIGH severity contamination (Air AQI: {air.get('value')}, Water Turbidity: {water.get('value')} NTU, Litter: {litter.get('value')} objects). Combined environmental pressure poses severe local hazards."
        action = "Dispatch emergency environmental response team immediately to inspect industrial runoff, test water quality, and deploy cleanup crews."
    elif high_count == 1:
        risk = "high" if med_count >= 1 else "medium"
        high_agent = "Air" if air.get("severity") == "high" else ("Water" if water.get("severity") == "high" else "Litter")
        reasoning = f"Elevated environmental risk: High severity signal detected by {high_agent} agent ({litter.get('note') if high_agent == 'Litter' else (air.get('note') if high_agent == 'Air' else water.get('note'))})."
        action = "Issue targeted remediation advisory and schedule priority field inspection within 24 hours."
    elif med_count >= 2:
        risk = "medium"
        reasoning = f"Moderate cumulative risk: Air AQI is {air.get('value')}, Water turbidity is {water.get('value')} NTU, and {litter.get('value')} litter objects detected."
        action = "Schedule routine maintenance and step up monitoring frequency."
    else:
        risk = "low"
        reasoning = "All environmental indicators (Air, Water, Litter) are within safe baseline parameters. No critical contamination detected."
        action = "Maintain automated continuous monitoring schedule."

    return {
        "overall_risk": risk,
        "reasoning": reasoning + " (Rule-based fallback — add GROQ_API_KEY to .env for LLM synthesis)",
        "recommended_action": action
    }


def run(air_result: dict, water_result: dict, litter_result: dict) -> dict:
    """
    Main entry point for the Coordinator Agent.

    Args:
        air_result:    Output dict from air_agent.run()
        water_result:  Output dict from water_agent.run()
        litter_result: Output dict from litter_agent.run()

    Returns:
        {"overall_risk": str, "reasoning": str, "recommended_action": str}
    """
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key.strip() == "" or api_key.startswith("your_"):
        return _fallback_synthesis(air_result, water_result, litter_result)

    try:
        client = Groq(api_key=api_key)
        user_prompt = _build_user_prompt(air_result, water_result, litter_result)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )

        raw_text = response.choices[0].message.content
        result = _extract_json(raw_text)

        for key in ("overall_risk", "reasoning", "recommended_action"):
            if key not in result:
                raise ValueError(f"LLM response missing required key: {key!r}")

        return result
    except Exception as exc:
        print(f"[coordinator_agent] Groq API call failed: {exc}. Using fallback synthesis.")
        fallback = _fallback_synthesis(air_result, water_result, litter_result)
        fallback["reasoning"] = fallback["reasoning"].replace("(Rule-based fallback — add GROQ_API_KEY to .env for LLM synthesis)", f"(Groq API fallback: {exc})")
        return fallback


if __name__ == "__main__":
    # Quick test with mock inputs
    mock_air = {
        "signal": "air", "severity": "high", "value": 310,
        "note": "Air quality in Delhi is Very Poor (AQI ≈ 310). Dominant pollutant: PM2.5.",
        "ph": None, "coliform_detected": False,
    }
    mock_water = {
        "signal": "water", "severity": "high", "value": 13.5,
        "note": "Water in Industrial Area: pH 5.1, turbidity 13.5 NTU. Severity: high.",
        "ph": 5.1, "coliform_detected": True,
    }
    mock_litter = {
        "signal": "litter", "severity": "high", "value": 9,
        "note": "Detected 9 objects — heavy litter contamination.",
        "boxes": [],
    }
    result = run(mock_air, mock_water, mock_litter)
    print(json.dumps(result, indent=2))
