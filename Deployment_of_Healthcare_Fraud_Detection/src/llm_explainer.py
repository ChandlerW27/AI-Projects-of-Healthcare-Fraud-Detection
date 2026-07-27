import json

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL


INSTRUCTIONS = """
You are a healthcare fraud analytics assistant.

Explain a provider-level model prediction using only the supplied fraud
probability and Logistic Regression feature contributions.

Rules:
- Explain the strongest risk signals in plain language.
- Do not invent diagnoses, claims, services, intent, or legal conclusions.
- Clearly separate model signals from verified facts.
- State that the result is a screening flag, not proof of fraud.
- Recommend human review of claims and supporting documentation.
- Keep the explanation under 180 words.
"""


def explain_flagged_case(probability, contributions, input_row):
    """Return an LLM explanation, or None when no API key is configured."""
    if not OPENAI_API_KEY:
        return None

    factors = [
        {
            "feature": feature,
            "input_value": round(float(input_row[feature]), 4),
            "risk_contribution": round(float(value), 4),
        }
        for feature, value in contributions.items()
    ]

    prompt = f"""
Fraud probability: {probability:.1%}

Strongest positive model contributions:
{json.dumps(factors, indent=2)}

Explain why this provider profile was flagged.
"""

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text.strip()
