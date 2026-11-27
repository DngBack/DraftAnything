"""Pure analysis functions: DataFrames/records in, DataFrames/dicts out (no I/O, no plotting)."""
from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from .data import FIELD_TO_GROUP, FIELDS

NONE_SENTINEL = "None"  # the dataset writes withheld answers (and "no religion") as the string "None"


# ----------------------------------------------------------------------------- structure
def overview(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "records": len(df),
        "profiles": df["profile_name"].nunique(),
        "domains": df["domain"].nunique(),
        "scenarios": df["scenario"].nunique(),
        "fields": df["field"].nunique(),
        "share": int(df["label"].sum()),
        "withhold": int((df["label"] == 0).sum()),
        "share_rate": round(float(df["label"].mean()), 4),
        "full_grid": len(df) == df["profile_name"].nunique() * df["scenario"].nunique() * df["field"].nunique(),
        "duplicate_keys": int(df.duplicated(["profile_name", "scenario", "field"]).sum()),
        "empty_prompts": int(df["prompt_empty"].sum()),
    }


def mixed_label_cells(df: pd.DataFrame) -> pd.DataFrame:
    """(scenario, field) cells whose label differs between profiles."""
    n = df.groupby(["scenario", "field"])["label"].nunique()
    return n[n > 1].reset_index(name="n_labels")


def norm_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Scenario x field matrix of the share label (columns in FIELD_GROUPS order).

    Only meaningful when labels do not depend on the profile, so that is asserted.
    """
    if not mixed_label_cells(df).empty:
        raise ValueError("labels vary across profiles; the norm matrix is not well defined")
    m = df.groupby(["scenario", "field"])["label"].first().unstack()
    return m[[f for f in FIELDS if f in m.columns]]


def share_rate(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    g = df.groupby(by)["label"]
    out = pd.DataFrame({"n": g.size(), "n_share": g.sum()})
    out["share_rate"] = (out["n_share"] / out["n"]).round(4)
    return out.sort_values("share_rate", ascending=False)


def scenario_similarity(norm: pd.DataFrame) -> pd.DataFrame:
    """Jaccard overlap between the sets of shareable fields of two scenarios."""
    sets = {s: set(norm.columns[norm.loc[s] == 1]) for s in norm.index}
    sim = pd.DataFrame(index=norm.index, columns=norm.index, dtype=float)
    for a, b in itertools.product(norm.index, repeat=2):
        union = sets[a] | sets[b]
        sim.loc[a, b] = len(sets[a] & sets[b]) / len(union) if union else 1.0
    return sim


# ----------------------------------------------------------------------------- baselines
def trivial_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of label predictors that never read the profile or the scenario text.

    Each predictor is a majority vote over training profiles, evaluated
    leave-one-profile-out, so a high score means the label is predictable
    without any reasoning about the specific user.
    """
    predictors = {
        "always_withhold": [],
        "by_field": ["field"],
        "by_scenario": ["scenario"],
        "by_domain_field": ["domain", "field"],
        "by_scenario_field": ["scenario", "field"],
    }
    rows = []
    for name, keys in predictors.items():
        correct = 0
        for pid in df["profile_id"].unique():
            train, test = df[df["profile_id"] != pid], df[df["profile_id"] == pid]
            if keys:
                rule = (train.groupby(keys)["label"].mean() >= 0.5).astype(int)
                pred = test.join(rule.rename("pred"), on=keys)["pred"]
            else:
                pred = pd.Series(0, index=test.index)
            correct += int((pred == test["label"]).sum())
        rows.append({"predictor": name, "keys": "+".join(keys) or "-", "accuracy": round(correct / len(df), 4)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- answers
def answer_checks(df: pd.DataFrame) -> dict[str, int]:
    shared, withheld = df[df["label"] == 1], df[df["label"] == 0]
    return {
        "shared_ref_equals_profile_value": int((shared["ref_answer"].map(repr) == shared["value"].map(repr)).sum()),
        "shared_total": len(shared),
        "withheld_ref_is_none_string": int((withheld["ref_answer"] == NONE_SENTINEL).sum()),
        "withheld_total": len(withheld),
        "shared_with_empty_answer": int(shared["value_empty"].sum()),
        "profile_values_equal_none_string": int((df.drop_duplicates(["profile_name", "field"])["value"] == NONE_SENTINEL).sum()),
    }


def field_value_stats(profiles: pd.DataFrame) -> pd.DataFrame:
    """Per field: JSON type, how many distinct values across profiles, how often empty."""
    rows = []
    for f in FIELDS:
        col = profiles[f]
        rows.append({
            "field": f,
            "group": FIELD_TO_GROUP[f],
            "kind": col.map(lambda v: type(v).__name__).mode()[0],
            "n_distinct": col.map(repr).nunique(),
            "n_empty": int(col.map(lambda v: v in ([], "", None)).sum()),
            "n_none_string": int((col == NONE_SENTINEL).sum()),
            "example": repr(col.tolist()[0])[:60],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- leakage surface
def atoms(value: Any) -> list[str]:
    """Strings a leak detector would search for: list items, or the scalar itself."""
    items = value if isinstance(value, list) else [value]
    return [str(x) for x in items]


def is_matchable(value: Any, min_len: int = 4) -> bool:
    """Can a verbatim string match reasonably detect this value in model output?

    Not matchable: empty lists, booleans, the "None" sentinel, and short tokens
    (e.g. ages "34") that collide with unrelated numbers in text.
    """
    if isinstance(value, bool):
        return False
    a = [x for x in atoms(value) if x != NONE_SENTINEL]
    return bool(a) and all(len(x) >= min_len for x in a)


def leak_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Per field, over withheld records: share of values a string-match detector can see."""
    w = df[df["label"] == 0].copy()
    w["matchable"] = w["value"].map(is_matchable)
    out = w.groupby("field").agg(n_withheld=("label", "size"), matchable_rate=("matchable", "mean"))
    out["group"] = out.index.map(FIELD_TO_GROUP)
    return out.sort_values("matchable_rate").round(3)


def value_collisions(profiles: pd.DataFrame) -> pd.DataFrame:
    """Within one profile, values shared by two different fields (case-insensitive).

    Such pairs make leak attribution ambiguous: sharing an allowed field can look
    like leaking a withheld one.
    """
    rows = []
    for _, p in profiles.iterrows():
        tokens = {f: {a.lower() for a in atoms(p[f]) if a != NONE_SENTINEL and len(a) >= 3} for f in FIELDS}
        for fa, fb in itertools.combinations(FIELDS, 2):
            for v in sorted(tokens[fa] & tokens[fb]):
                rows.append({"profile": p["name"], "field_a": fa, "field_b": fb, "value": v})
    return pd.DataFrame(rows, columns=["profile", "field_a", "field_b", "value"])


# ----------------------------------------------------------------------------- split relations
def _key(r: dict[str, Any]) -> tuple:
    return (r["profile"]["name"], r["scenario"], r["field"])


def subset_report(sub: list[dict], full: list[dict]) -> dict[str, Any]:
    index = {_key(r): r for r in full}
    same = sum(1 for r in sub if _key(r) in index and index[_key(r)] == r)
    return {"records": len(sub), "identical_in_full": same,
            "profiles": sorted({r["profile"]["name"] for r in sub})}


def swap_report(original: list[dict], flipped: list[dict], full: list[dict]) -> dict[str, Any]:
    """Check the swap ablation: same tasks, one profile field rewritten to a paraphrase."""
    pairs = list(zip(original, flipped))
    changed = [tuple(f for f in FIELDS if o["profile"][f] != s["profile"][f]) for o, s in pairs]
    full_index = {_key(r): r for r in full}
    profiles_all = {r["profile"]["name"] for r in full}
    by_field: dict[str, set] = {}
    for r in original:
        by_field.setdefault(r["field"], set()).add(r["profile"]["name"])
    return {
        "records": len(pairs),
        "same_task_and_label": all(all(o[k] == s[k] for k in ("scenario", "field", "label")) for o, s in pairs),
        "only_asked_field_changed": all(c == (o["field"],) for c, (o, _) in zip(changed, pairs)),
        "answers_swapped_when_shared": all(
            o["label"] == 0 or (s["ref_answer"] == o["injected_answer"] and s["injected_answer"] == o["ref_answer"])
            for o, s in pairs),
        "original_labels_match_full": all(
            full_index[_key(o)]["label"] == o["label"] for o in original if _key(o) in full_index),
        "fields": {f: len(v) for f, v in by_field.items()},
        "profiles_missing": {f: sorted(profiles_all - v) for f, v in by_field.items() if profiles_all - v},
        "share": int(sum(o["label"] for o in original)),
    }
