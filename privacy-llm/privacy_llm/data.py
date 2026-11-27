"""Load AirGapAgent-R records and turn them into tidy DataFrames.

One record = (user profile, scenario, field): "in this scenario, should the agent
share this field of the user's profile?"  `label` 1 = share, 0 = withhold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RAW_DIR, SPLITS

# Our own grouping of the 26 profile fields (not part of the dataset); used only
# to make the EDA readable.
FIELD_GROUPS: dict[str, list[str]] = {
    "identity": ["name", "age", "gender", "ethnicity"],
    "contact": ["address", "phone_number", "email"],
    "government_id": ["social_security_number", "driver_licence_number"],
    "health": [
        "current_physical_health_conditions", "current_mental_health_conditions",
        "allergies", "smoker", "family_medical_history", "current_medications",
        "disabilities",
    ],
    "lifestyle": ["average_exercise_hours_per_week", "diet_type", "pet_ownership"],
    "sensitive_personal": ["relationship_status", "religious_beliefs", "sexual_orientation"],
    "preferences": [
        "preferred_movie_genres", "vacation_preferences", "favorite_food", "favorite_hobbies",
    ],
}
FIELDS: list[str] = [f for group in FIELD_GROUPS.values() for f in group]
FIELD_TO_GROUP: dict[str, str] = {f: g for g, fs in FIELD_GROUPS.items() for f in fs}

REQUIRED_KEYS = {"profile", "domain", "scenario", "field", "prompt", "label", "ref_answer"}


def load_records(split: str = "full", raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    path = raw_dir / SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/download_data.py first")
    records = json.loads(path.read_text())
    validate(records)
    return records


def validate(records: list[dict[str, Any]]) -> None:
    """Schema checks that every split must pass; raises ValueError on the first violation."""
    for i, r in enumerate(records):
        missing = REQUIRED_KEYS - r.keys()
        if missing:
            raise ValueError(f"record {i}: missing keys {sorted(missing)}")
        if r["label"] not in (0, 1):
            raise ValueError(f"record {i}: label {r['label']!r} not in {{0, 1}}")
        if r["field"] not in FIELD_TO_GROUP:
            raise ValueError(f"record {i}: unknown field {r['field']!r}")
        if set(r["profile"]) != set(FIELDS):
            raise ValueError(f"record {i}: profile keys differ from the 26 known fields")


def value_kind(value: Any) -> str:
    """Coarse JSON type of a profile value: list / bool / int / str / null."""
    if value is None:
        return "null"
    if isinstance(value, bool):  # before int: bool is a subclass of int
        return "bool"
    return type(value).__name__


def is_empty(value: Any) -> bool:
    return value is None or value == [] or value == ""


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per record; the profile is reduced to the value of the asked field."""
    rows = []
    for r in records:
        value = r["profile"][r["field"]]
        rows.append({
            "profile_name": r["profile"]["name"],
            "domain": r["domain"],
            "scenario": r["scenario"],
            "field": r["field"],
            "field_group": FIELD_TO_GROUP[r["field"]],
            "label": r["label"],
            "value": value,
            "value_kind": value_kind(value),
            "value_empty": is_empty(value),
            "ref_answer": r["ref_answer"],
            "injected_answer": r.get("injected_answer"),
            "prompt_empty": r["prompt"] == "",
        })
    df = pd.DataFrame(rows)
    # Stable integer ids in order of first appearance.
    df.insert(0, "profile_id", pd.factorize(df["profile_name"])[0])
    return df


def profiles_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per distinct user profile (deduplicated by full content)."""
    seen: dict[str, dict[str, Any]] = {}
    for r in records:
        seen.setdefault(json.dumps(r["profile"], sort_keys=True), r["profile"])
    return pd.DataFrame(list(seen.values()))[FIELDS]
