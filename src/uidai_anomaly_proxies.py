"""Anomaly proxies computable from UIDAI aggregated CSV exports.

Your downloaded datasets are aggregated by:
- date, state, district, pincode
- plus age-bucket count columns (enrolment) / update count columns (demo/bio)

This module implements *proxies* for hackathon-style anomaly detection that only
use fields present in those aggregated CSVs.

Notes:
- Some anomalies (Census saturation, institutional births) need external baselines.
- Some anomalies (wait time volatility, rejection clusters, error codes) need
  operational logs and are not computable from these CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ensure_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    df["year_month"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    return df


def monthly_sum(df: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    """Monthly sum for given value columns."""
    d = _ensure_year_month(df)
    cols = [c for c in value_cols if c in d.columns]
    if not cols:
        raise ValueError(f"None of {value_cols} found in dataframe columns.")
    out = d.groupby(group_cols + ["year_month"], as_index=False)[cols].sum()
    return out


def zscore_over_time(df: pd.DataFrame, by: list[str], value_col: str) -> pd.DataFrame:
    """Compute z-score of a value across time for each group."""
    d = df.copy()
    mu = d.groupby(by)[value_col].transform("mean")
    sigma = d.groupby(by)[value_col].transform("std").replace(0, np.nan)
    d["z"] = (d[value_col] - mu) / sigma
    return d


def top_outliers(df: pd.DataFrame, score_col: str, n: int = 50, ascending: bool = False) -> pd.DataFrame:
    """Return top rows by score (handles NaNs)."""
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[score_col])
    return d.sort_values(score_col, ascending=ascending).head(n)


def detect_adult_enrolment_surges(enrol: pd.DataFrame, *, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Adult enrolment surges: strong z-score bursts in age_18_greater at state level.
    """
    m = monthly_sum(enrol, ["age_18_greater"], ["state"])
    m = zscore_over_time(m, ["state"], "age_18_greater")
    m["is_surge"] = m["z"] >= z_threshold
    return m.sort_values(["is_surge", "z"], ascending=[False, False])


def detect_enrolment_deserts(
    enrol: pd.DataFrame,
    *,
    latest_only: bool = True,
    min_district_under5: float = 500.0,
    max_pincode_under5: float = 5.0,
) -> pd.DataFrame:
    """
    Enrolment deserts proxy (0-5): district has high 0-5 enrolment, but a pincode is near-zero.
    """
    m = monthly_sum(enrol, ["age_0_5"], ["state", "district", "pincode"])
    if latest_only:
        latest = m["year_month"].max()
        m = m[m["year_month"] == latest].copy()

    district = m.groupby(["state", "district", "year_month"], as_index=False)["age_0_5"].sum()
    district = district.rename(columns={"age_0_5": "district_under5"})
    out = m.merge(district, on=["state", "district", "year_month"], how="left")
    out["is_desert"] = (out["district_under5"] >= min_district_under5) & (out["age_0_5"] <= max_pincode_under5)
    out["desert_score"] = out["district_under5"] / (out["age_0_5"] + 1.0)
    return out.sort_values(["is_desert", "desert_score"], ascending=[False, False])


def detect_mbu_backlog(
    demo: pd.DataFrame,
    bio: pd.DataFrame,
    *,
    latest_only: bool = True,
    min_demo: float = 100.0,
) -> pd.DataFrame:
    """
    Mandatory Biometric Update backlog proxy:
    high demo_age_5_17 but low bio_age_5_17 in same pincode (same month).
    """
    dm = monthly_sum(demo, ["demo_age_5_17"], ["state", "district", "pincode"])
    bm = monthly_sum(bio, ["bio_age_5_17"], ["state", "district", "pincode"])

    if latest_only:
        latest = min(dm["year_month"].max(), bm["year_month"].max())
        dm = dm[dm["year_month"] == latest].copy()
        bm = bm[bm["year_month"] == latest].copy()

    out = dm.merge(bm, on=["state", "district", "pincode", "year_month"], how="left")
    out["bio_age_5_17"] = out["bio_age_5_17"].fillna(0)
    out["backlog_ratio"] = out["demo_age_5_17"] / (out["bio_age_5_17"] + 1.0)
    out["is_backlog"] = out["demo_age_5_17"] >= min_demo
    return out.sort_values(["is_backlog", "backlog_ratio"], ascending=[False, False])


def detect_digital_divide(
    demo: pd.DataFrame,
    bio: pd.DataFrame,
    *,
    latest_only: bool = True,
    min_demo_adult: float = 100.0,
) -> pd.DataFrame:
    """
    Digital divide / access disparity proxy:
    demo_age_17_ high (often online) while bio_age_17_ low/stagnant (must be in-person).
    """
    dm = monthly_sum(demo, ["demo_age_17_"], ["state", "district", "pincode"])
    bm = monthly_sum(bio, ["bio_age_17_"], ["state", "district", "pincode"])
    if latest_only:
        latest = min(dm["year_month"].max(), bm["year_month"].max())
        dm = dm[dm["year_month"] == latest].copy()
        bm = bm[bm["year_month"] == latest].copy()

    out = dm.merge(bm, on=["state", "district", "pincode", "year_month"], how="left")
    out["bio_age_17_"] = out["bio_age_17_"].fillna(0)
    out["demo_to_bio_ratio"] = out["demo_age_17_"] / (out["bio_age_17_"] + 1.0)
    out["is_gap"] = out["demo_age_17_"] >= min_demo_adult
    return out.sort_values(["is_gap", "demo_to_bio_ratio"], ascending=[False, False])


def detect_migration_trail_signature(
    demo: pd.DataFrame,
    *,
    z_threshold: float = 3.0,
    top_urban_quantile: float = 0.9,
) -> pd.DataFrame:
    """
    Migration trail proxy:
    pincode-level z-score bursts in demo_age_17_ (adult) focusing on high-volume pincodes.
    """
    m = monthly_sum(demo, ["demo_age_17_"], ["state", "district", "pincode"])
    m = zscore_over_time(m, ["pincode"], "demo_age_17_")

    # "Urban" proxy: high average activity pincodes
    avg = m.groupby("pincode", as_index=False)["demo_age_17_"].mean().rename(columns={"demo_age_17_": "avg_demo_adult"})
    m = m.merge(avg, on="pincode", how="left")
    thresh = avg["avg_demo_adult"].quantile(top_urban_quantile)
    m["is_urban_proxy"] = m["avg_demo_adult"] >= thresh
    m["is_spike"] = (m["z"] >= z_threshold) & m["is_urban_proxy"]
    return m.sort_values(["is_spike", "z"], ascending=[False, False])


def detect_pincode_activity_blackouts(
    enrol: pd.DataFrame,
    demo: pd.DataFrame,
    bio: pd.DataFrame,
    *,
    z_threshold: float = -3.0,
) -> pd.DataFrame:
    """
    Cross-dataset blackouts proxy:
    z-score <= -3 for monthly totals across ALL three datasets for the same pincode.
    """
    em = monthly_sum(enrol, ["total"], ["state", "district", "pincode"]).rename(columns={"total": "enrol_total"})
    dm = monthly_sum(demo, ["total"], ["state", "district", "pincode"]).rename(columns={"total": "demo_total"})
    bm = monthly_sum(bio, ["total"], ["state", "district", "pincode"]).rename(columns={"total": "bio_total"})

    merged = em.merge(dm, on=["state", "district", "pincode", "year_month"], how="outer").merge(
        bm, on=["state", "district", "pincode", "year_month"], how="outer"
    )
    merged[["enrol_total", "demo_total", "bio_total"]] = merged[["enrol_total", "demo_total", "bio_total"]].fillna(0)

    for col in ["enrol_total", "demo_total", "bio_total"]:
        zdf = zscore_over_time(merged[["pincode", "year_month", col]].copy(), ["pincode"], col)
        merged[f"z_{col}"] = zdf["z"].values

    merged["is_blackout"] = (
        (merged["z_enrol_total"] <= z_threshold)
        & (merged["z_demo_total"] <= z_threshold)
        & (merged["z_bio_total"] <= z_threshold)
    )
    merged["blackout_score"] = merged[["z_enrol_total", "z_demo_total", "z_bio_total"]].min(axis=1)
    return merged.sort_values(["is_blackout", "blackout_score"], ascending=[False, True])


def adult_bio_concentration_by_district(
    bio: pd.DataFrame, *, latest_only: bool = True, top_k: int = 5
) -> pd.DataFrame:
    """
    "Elderly exclusion" proxy (limited by available columns):
    measures adult biometric updates concentration within a district.

    High concentration in a few pincodes suggests service access is clustered in urban pockets.
    """
    m = monthly_sum(bio, ["bio_age_17_"], ["state", "district", "pincode"])
    if latest_only:
        m = m[m["year_month"] == m["year_month"].max()].copy()

    # District totals
    dist = m.groupby(["state", "district", "year_month"], as_index=False)["bio_age_17_"].sum().rename(
        columns={"bio_age_17_": "district_bio_adult"}
    )
    top = (
        m.sort_values(["state", "district", "year_month", "bio_age_17_"], ascending=[True, True, True, False])
        .groupby(["state", "district", "year_month"], as_index=False)
        .head(top_k)
    )
    top_sum = top.groupby(["state", "district", "year_month"], as_index=False)["bio_age_17_"].sum().rename(
        columns={"bio_age_17_": f"top{top_k}_bio_adult"}
    )
    out = dist.merge(top_sum, on=["state", "district", "year_month"], how="left").fillna(0)
    out["top_share"] = out[f"top{top_k}_bio_adult"] / (out["district_bio_adult"] + 1.0)
    return out.sort_values("top_share", ascending=False)


def census_saturation_anomaly(
    enrol: pd.DataFrame,
    *,
    population_2011_csv: Path,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Requires an external 2011 Census population baseline by district.

    Expected columns in population CSV:
    - state
    - district
    - population_2011
    """
    pop = pd.read_csv(population_2011_csv)
    pop.columns = pop.columns.str.lower().str.strip().str.replace(" ", "_").str.replace("-", "_")
    if not {"state", "district", "population_2011"}.issubset(pop.columns):
        raise ValueError("population_2011_csv must contain columns: state, district, population_2011")

    d = _ensure_year_month(enrol)
    latest = d["year_month"].max()
    d = d[d["year_month"] == latest].copy()

    dist = d.groupby(["state", "district"], as_index=False)["total"].sum().rename(columns={"total": "enrol_total"})
    out = dist.merge(pop[["state", "district", "population_2011"]], on=["state", "district"], how="left")
    out["saturation"] = out["enrol_total"] / (out["population_2011"] + 1.0)
    out["is_over_100pct"] = out["saturation"] > threshold
    return out.sort_values(["is_over_100pct", "saturation"], ascending=[False, False])

