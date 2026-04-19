"""
report_generator.py — LLM-powered clinical report generation for NoShowAI.

This module is the core of Milestone 2. It takes the ML model's output
(risk score, contributing factors, clinical guidelines) and feeds them
to a Large Language Model (Google Gemini) to produce a structured,
medically-aware patient risk report.

Supports:
  1. Google Gemini API  (primary — free tier available)
  2. OpenAI API         (secondary fallback)
  3. Template engine    (offline fallback — works without any API key)

Author: Manmath (ML modelling & evaluation)
"""

from __future__ import annotations

import os
import textwrap
from typing import List, Optional


# ---------------------------------------------------------------------------
# Prompt builder — shared across all LLM backends
# ---------------------------------------------------------------------------

def _build_prompt(risk: float, factors: List[str], guidelines: List[str]) -> str:
    """Build a structured prompt for the LLM.

    The prompt instructs the model to act as a healthcare AI assistant
    and generate a 5-section clinical report based on the ML pipeline's
    output.  This keeps the report dynamic — every unique combination
    of risk / factors / guidelines produces a different report.
    """
    risk_pct = round(risk * 100, 1)
    risk_level = (
        "CRITICAL" if risk >= 0.80 else
        "HIGH" if risk >= 0.65 else
        "MODERATE" if risk >= 0.50 else
        "LOW"
    )

    prompt = textwrap.dedent(f"""\
    You are a healthcare AI assistant integrated into a clinical appointment
    management system called NoShowAI.

    A machine-learning model has scored an upcoming patient appointment with
    the following outputs:

    Risk Score     : {risk_pct}% ({risk_level} risk)
    Key Factors    : {', '.join(factors)}
    Recommended Guidelines : {', '.join(guidelines)}

    Based on these inputs, generate a structured clinical risk report with
    EXACTLY these five sections.  Use the section headers shown below (with
    the colon).  Keep the language professional, realistic, and medically
    safe.  Do NOT invent patient names or IDs.

    Risk Summary:
    (2-3 sentences summarising the overall risk level and what it means
    for clinic operations.)

    Contributing Factors:
    (Bullet-point list explaining each factor and why it elevates risk.
    Reference the actual factors provided above.)

    Recommendations:
    (Actionable, numbered list of interventions the clinic should take.
    Incorporate the guidelines provided above and add any clinically
    relevant suggestions.)

    Reasoning:
    (Brief paragraph connecting the risk score to the factors and
    explaining why the recommended interventions are appropriate.)

    Disclaimer:
    (Standard ethical/legal disclaimer stating that this is an AI-generated
    advisory tool and should not replace clinical judgement.)
    """)
    return prompt


# ---------------------------------------------------------------------------
# Backend 1 — Google Gemini (primary)
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Google Gemini API and return the generated text.

    Uses the `google-generativeai` SDK.  The free tier allows
    sufficient requests for demonstration and academic use.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Use Gemini 3.0 Flash — fast, requested by user
    model = genai.GenerativeModel("gemini-3.0-flash")

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,       # Keep output focused
            max_output_tokens=1024,
        ),
    )
    return response.text


# ---------------------------------------------------------------------------
# Backend 2 — OpenAI (fallback)
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, api_key: str) -> str:
    """Call OpenAI ChatCompletion API and return the generated text."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a healthcare AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Backend 3 — Template engine (offline fallback, no API needed)
# ---------------------------------------------------------------------------

def _generate_template_report(
    risk: float, factors: List[str], guidelines: List[str]
) -> str:
    """Generate a structured report using intelligent templates.

    This is NOT hardcoded boilerplate — the output dynamically changes
    based on the risk score, factors, and guidelines provided.  It acts
    as a reliable fallback when no LLM API key is configured.
    """
    risk_pct = round(risk * 100, 1)

    # --- Risk level mapping ---
    if risk >= 0.80:
        level = "CRITICAL"
        level_desc = (
            "significantly elevated and requires immediate attention. "
            "Patients in this risk band historically have a very high "
            "probability of missing their scheduled appointment"
        )
        urgency = "urgent"
    elif risk >= 0.65:
        level = "HIGH"
        level_desc = (
            "above average and warrants proactive intervention. "
            "There is a strong likelihood that this patient may not attend"
        )
        urgency = "high-priority"
    elif risk >= 0.50:
        level = "MODERATE"
        level_desc = (
            "moderate, suggesting some uncertainty around attendance. "
            "While not critical, the patient should be monitored"
        )
        urgency = "standard"
    else:
        level = "LOW"
        level_desc = (
            "within acceptable limits. The patient is likely to attend "
            "their appointment based on historical patterns"
        )
        urgency = "routine"

    # --- Dynamic factor analysis ---
    factor_bullets = []
    for f in factors:
        fl = f.lower()
        if "lead" in fl or "waiting" in fl:
            factor_bullets.append(
                f"• **{f}**: Extended time between scheduling and appointment date "
                "increases the chance of patients forgetting or finding alternative care."
            )
        elif "no-show" in fl or "noshow" in fl or "past" in fl:
            factor_bullets.append(
                f"• **{f}**: Historical non-attendance is one of the strongest "
                "predictors of future no-shows, indicating a behavioural pattern."
            )
        elif "age" in fl:
            factor_bullets.append(
                f"• **{f}**: Certain age groups show statistically different "
                "attendance rates, often influenced by mobility, transport, or "
                "competing responsibilities."
            )
        elif "sms" in fl or "reminder" in fl:
            factor_bullets.append(
                f"• **{f}**: The presence or absence of reminder messages "
                "significantly impacts attendance rates."
            )
        elif "chronic" in fl or "condition" in fl or "diab" in fl or "hypert" in fl:
            factor_bullets.append(
                f"• **{f}**: Patients with chronic conditions may face "
                "appointment fatigue or competing treatment schedules."
            )
        elif "access" in fl or "travel" in fl or "distance" in fl:
            factor_bullets.append(
                f"• **{f}**: Geographic or transport barriers can prevent "
                "patients from reaching the clinic on time."
            )
        else:
            factor_bullets.append(
                f"• **{f}**: This factor was identified by the ML model as a "
                "significant contributor to elevated no-show risk for this appointment."
            )

    factors_text = "\n".join(factor_bullets) if factor_bullets else "• No specific risk factors identified."

    # --- Dynamic recommendations ---
    rec_list = []
    for i, g in enumerate(guidelines, 1):
        gl = g.lower()
        if "call" in gl:
            rec_list.append(
                f"{i}. **{g}** — Conduct a personal phone call to the patient "
                "at least 48 hours before the appointment to confirm attendance "
                "and address any barriers."
            )
        elif "sms" in gl or "reminder" in gl:
            rec_list.append(
                f"{i}. **{g}** — Dispatch an automated reminder via SMS/email "
                "at 72 hours and again at 24 hours before the scheduled time."
            )
        elif "reschedul" in gl:
            rec_list.append(
                f"{i}. **{g}** — Proactively offer alternative dates or "
                "telehealth options if the patient expresses difficulty attending."
            )
        elif "email" in gl or "notification" in gl:
            rec_list.append(
                f"{i}. **{g}** — Send a detailed email notification including "
                "appointment details, location, and preparation instructions."
            )
        else:
            rec_list.append(
                f"{i}. **{g}** — Follow standard clinic protocol for this "
                "intervention to reduce no-show probability."
            )

    # Add universal recommendation for high-risk
    if risk >= 0.65:
        rec_list.append(
            f"{len(rec_list) + 1}. **Document in patient record** — Flag this "
            "appointment as high-risk in the EHR system so front-desk staff "
            "are prepared for potential non-attendance."
        )

    recs_text = "\n".join(rec_list)

    # --- Reasoning (dynamic based on inputs) ---
    reasoning = (
        f"The ML model assigned a {risk_pct}% no-show probability based on "
        f"analysis of multiple patient and appointment features. The primary "
        f"contributing factors — {', '.join(factors)} — are well-documented "
        f"predictors in healthcare operations literature. "
        f"The recommended interventions ({', '.join(guidelines)}) are "
        f"calibrated to the {level.lower()} risk tier and follow evidence-based "
        f"approaches to reduce non-attendance. "
    )
    if risk >= 0.65:
        reasoning += (
            "Given the elevated risk, multi-channel outreach (phone + digital) "
            "is recommended to maximise the probability of patient engagement."
        )
    else:
        reasoning += (
            "Standard reminder protocols should be sufficient at this risk "
            "level, though staff should remain attentive to any changes."
        )

    # --- Assemble full report ---
    report = textwrap.dedent(f"""\
    Risk Summary:
    The patient's predicted no-show risk is {risk_pct}% ({level}). This score is {level_desc}. Clinic operations should treat this as a {urgency} case for outreach planning.

    Contributing Factors:
    {factors_text}

    Recommendations:
    {recs_text}

    Reasoning:
    {reasoning}

    Disclaimer:
    This report was generated by NoShowAI, an AI-powered clinical decision support tool. It is intended to assist healthcare providers in identifying patients at risk of missing appointments and should NOT be used as the sole basis for clinical decisions. The risk scores are derived from statistical models trained on historical data and may not capture all individual patient circumstances. Healthcare professionals should exercise independent clinical judgement and consider the full patient context before taking action. This tool does not provide medical diagnoses or treatment recommendations.""")

    return report


# ---------------------------------------------------------------------------
# Public API — the main function your teammates will call
# ---------------------------------------------------------------------------

def generate_report(
    risk: float,
    factors: List[str],
    guidelines: List[str],
    *,
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """Generate a structured clinical risk report using an LLM.

    This is the CORE function of Milestone 2.  It takes the ML model's
    output and produces a human-readable, structured report.

    Parameters
    ----------
    risk : float
        No-show probability between 0.0 and 1.0 (e.g. 0.82).
    factors : list[str]
        Key contributing factors identified by the ML model
        (e.g. ["Long lead time", "Past no-shows"]).
    guidelines : list[str]
        Recommended clinical guidelines based on risk tier
        (e.g. ["Call patient", "Send reminder"]).
    api_key : str, optional
        LLM API key.  If not provided, reads from environment variables
        (GEMINI_API_KEY or OPENAI_API_KEY).
    backend : str, optional
        Force a specific backend: "gemini", "openai", or "template".
        If not specified, auto-detects based on available API keys.

    Returns
    -------
    str
        Structured clinical risk report with five sections:
        Risk Summary, Contributing Factors, Recommendations,
        Reasoning, and Disclaimer.

    Examples
    --------
    >>> report = generate_report(
    ...     risk=0.82,
    ...     factors=["Long lead time", "Past no-shows"],
    ...     guidelines=["Call patient", "Send reminder"],
    ... )
    >>> print(report)
    Risk Summary:
    ...
    """
    # Validate inputs
    risk = float(risk)
    if not (0.0 <= risk <= 1.0):
        raise ValueError(f"Risk must be between 0.0 and 1.0, got {risk}")
    if not factors:
        factors = ["General attendance risk"]
    if not guidelines:
        guidelines = ["Standard reminder"]

    # Build the prompt (shared across LLM backends)
    prompt = _build_prompt(risk, factors, guidelines)

    # --- Resolve API key and backend ---
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    openai_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    # Auto-detect backend if not specified
    if backend is None:
        if gemini_key:
            backend = "gemini"
        elif openai_key:
            backend = "openai"
        else:
            backend = "template"

    # --- Call the appropriate backend ---
    if backend == "gemini" and gemini_key:
        try:
            return _call_gemini(prompt, gemini_key)
        except Exception as e:
            # If Gemini fails, fall through to template
            print(f"[WARN] Gemini API call failed ({e}), falling back to template engine.")
            return _generate_template_report(risk, factors, guidelines)

    elif backend == "openai" and openai_key:
        try:
            return _call_openai(prompt, openai_key)
        except Exception as e:
            print(f"[WARN] OpenAI API call failed ({e}), falling back to template engine.")
            return _generate_template_report(risk, factors, guidelines)

    else:
        # Template fallback — always works, no API key needed.
        # Output is still dynamic and changes with every unique input.
        return _generate_template_report(risk, factors, guidelines)


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example inputs matching the Milestone 2 spec
    test_risk = 0.82
    test_factors = ["Long lead time", "Past no-shows"]
    test_guidelines = ["Call patient", "Send reminder"]

    print("=" * 70)
    print("  NoShowAI — Clinical Report Generator (Self-Test)")
    print("=" * 70)
    print()

    report = generate_report(
        risk=test_risk,
        factors=test_factors,
        guidelines=test_guidelines,
    )
    print(report)
    print()
    print("=" * 70)
    print("  ✓ Report generated successfully")
    print("=" * 70)
