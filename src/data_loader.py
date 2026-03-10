"""Data loading and preprocessing utilities.

These UIDAI public datasets are aggregated at date/state/district/pincode level
with age-bucket columns. This module standardises them into a common shape:

- `date` parsed as datetime (day-first)
- `pincode` kept as 6-digit string
- `total` = sum of metric columns (age buckets / update counts)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings
from pandas.api.types import is_datetime64_any_dtype

from src.config import (
    ENROLMENT_FILE,
    DEMOGRAPHIC_UPDATE_FILE,
    BIOMETRIC_UPDATE_FILE,
    DATE_COLUMNS
    ,ENROLMENT_DIR
    ,DEMOGRAPHIC_UPDATE_DIR
    ,BIOMETRIC_UPDATE_DIR
)

warnings.filterwarnings('ignore')

def _read_csv_many(path: Path, *, dtype: Optional[Dict] = None) -> pd.DataFrame:
    """
    Read either:
    - a single CSV file, or
    - a directory containing multiple CSV files (chunks)
    and concatenate them.
    """
    if path.is_file():
        return pd.read_csv(path, low_memory=False, dtype=dtype)

    if path.is_dir():
        files = sorted([p for p in path.glob("*.csv") if p.is_file()])
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        frames: List[pd.DataFrame] = []
        for f in files:
            frames.append(pd.read_csv(f, low_memory=False, dtype=dtype))
        return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError(f"Path not found: {path}")


def _parse_date_series(s: pd.Series) -> pd.Series:
    # UIDAI downloads typically use dd-mm-yyyy.
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _standardise_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names (lowercase, replace spaces with underscores)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_").str.replace("-", "_")

    # Parse date
    if "date" in df.columns and not is_datetime64_any_dtype(df["date"]):
        df["date"] = _parse_date_series(df["date"])

    # Keep pincode as a 6-digit string (preserve leading zeros)
    if "pincode" in df.columns:
        df["pincode"] = df["pincode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

    return df


def _infer_metric_columns(df: pd.DataFrame) -> list[str]:
    """
    Metric columns are the numeric 'counts' we want to sum into `total`.
    In these datasets they typically contain 'age' in the name.
    """
    cols = [c for c in df.columns if "age" in c and c not in {"age", "page"}]
    if cols:
        return cols

    # Fallback: all numeric columns except obvious identifiers/derived fields
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = {"pincode", "year", "month", "quarter", "day_of_week", "is_weekend"}
    return [c for c in numeric_cols if c not in drop]


def load_enrolment_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load enrolment dataset"""
    # Support both single-file and chunked-folder layout
    path = filepath or (ENROLMENT_FILE if ENROLMENT_FILE.exists() else ENROLMENT_DIR)
    dtype = {
        "date": "string",
        "state": "string",
        "district": "string",
        "pincode": "string",
        "age_0_5": "Int32",
        "age_5_17": "Int32",
        "age_18_greater": "Int32",
    }
    df = _read_csv_many(path, dtype=dtype)
    return _standardise_common_columns(df)


def load_demographic_update_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load demographic update dataset"""
    path = filepath or (DEMOGRAPHIC_UPDATE_FILE if DEMOGRAPHIC_UPDATE_FILE.exists() else DEMOGRAPHIC_UPDATE_DIR)
    dtype = {
        "date": "string",
        "state": "string",
        "district": "string",
        "pincode": "string",
        # handle both possible column spellings from API exports
        "demo_age_5_17": "Int32",
        "demo_age_17_": "Int32",
    }
    df = _read_csv_many(path, dtype=dtype)
    return _standardise_common_columns(df)


def load_biometric_update_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load biometric update dataset"""
    path = filepath or (BIOMETRIC_UPDATE_FILE if BIOMETRIC_UPDATE_FILE.exists() else BIOMETRIC_UPDATE_DIR)
    dtype = {
        "date": "string",
        "state": "string",
        "district": "string",
        "pincode": "string",
        "bio_age_5_17": "Int32",
        "bio_age_17_": "Int32",
    }
    df = _read_csv_many(path, dtype=dtype)
    return _standardise_common_columns(df)


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three datasets"""
    enrolment = load_enrolment_data()
    demographic = load_demographic_update_data()
    biometric = load_biometric_update_data()
    return enrolment, demographic, biometric


def preprocess_data(df: pd.DataFrame, dataset_type: str = 'enrolment') -> pd.DataFrame:
    """Preprocess data for analysis"""
    df = _standardise_common_columns(df)

    # Ensure metric cols are numeric
    metric_cols = _infer_metric_columns(df)
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Total activity for plotting/anomalies
    if metric_cols:
        df["total"] = df[metric_cols].sum(axis=1)
    else:
        df["total"] = 0

    # Extract date components
    if "date" in df.columns and is_datetime64_any_dtype(df["date"]):
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["date"].dt.dayofweek >= 5

    return df


def get_aggregated_stats(df: pd.DataFrame, group_by_cols: list) -> pd.DataFrame:
    """Get aggregated statistics by grouping columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return pd.DataFrame()
    
    agg_dict = {col: ['sum', 'mean', 'std', 'count'] for col in numeric_cols}
    
    return df.groupby(group_by_cols).agg(agg_dict).reset_index()
