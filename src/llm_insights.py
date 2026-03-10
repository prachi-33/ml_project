"""LLM-based insights generation (optional).

This module is intentionally lightweight and only sends *aggregated* statistics
to the LLM (not raw row-level data) to reduce privacy risk and token usage.

API key handling:
- Prefer environment variable `OPENAI_API_KEY`
- Or Streamlit secrets: `st.secrets["OPENAI_API_KEY"]`
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict

import numpy as _np


DEFAULT_MODEL = "gpt-4o-mini"


def _json_default(obj: Any) -> Any:
    """
    Make common scientific / pandas objects JSON-serializable.
    - pandas.Timestamp -> ISO string (handled via datetime interface / fallback to str)
    - numpy scalars -> Python scalars
    """
    # numpy numbers / booleans
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)

    # datetime-like
    if isinstance(obj, (_dt.datetime, _dt.date)):
        # Use ISO to keep it readable and stable
        return obj.isoformat()

    # pandas Timestamp (avoid importing pandas just for this)
    # It behaves like datetime, but if not caught above, stringify.
    return str(obj)


def generate_conclusions(
    *,
    api_key: str,
    payload: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate conclusions + suggestions from a compact payload of aggregated stats.
    Returns markdown text.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "OpenAI SDK is not installed. Install it with: `pip install openai`"
        ) from e

    client = OpenAI(api_key=api_key)

    system = (
        "You are a government analytics assistant. "
        "Write concise, evidence-based conclusions and actionable recommendations. "
        "Be explicit about limitations and what additional data would improve confidence. "
        "Do not invent fields that are not present in the provided payload."
    )

    user = (
        "Using the aggregated stats below, generate:\n"
        "1) Executive summary (5 bullets)\n"
        "2) Key insights (8–12 bullets)\n"
        "3) Priority anomalies to investigate (top 5, with why)\n"
        "4) Recommendations (operational + policy + outreach)\n"
        "5) Limitations / data gaps\n\n"
        "Return in markdown.\n\n"
        f"Payload JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)}"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    # The python SDK returns text via output_text helper
    return getattr(resp, "output_text", None) or ""

