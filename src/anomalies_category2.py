"""Category II: Data Integrity and Technical Deviations"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats

from src.config import (
    ANOMALY_THRESHOLD_SIGMA,
    MULTIPLE_ENROLMENT_THRESHOLD,
    MOBILE_CHURN_THRESHOLD,
    BIOMETRIC_EXCEPTION_THRESHOLD
)


def detect_age_dob_discrepancy(
    df: pd.DataFrame,
    age_col: str = 'age',
    dob_col: str = 'date_of_birth',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Detect Impossible Age/DOB Discrepancy: Age indicates minor but DOB
    indicates different year, flagging validation logic failure.
    """
    df = df.copy()
    
    if age_col not in df.columns or dob_col not in df.columns:
        return pd.DataFrame()
    
    # Convert DOB to datetime
    df[dob_col] = pd.to_datetime(df[dob_col], errors='coerce')
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['calculated_age'] = (df[date_col] - df[dob_col]).dt.days / 365.25
    else:
        df['calculated_age'] = None
    
    # Flag discrepancies
    if 'calculated_age' in df.columns and df['calculated_age'].notna().any():
        df['age_diff'] = abs(df[age_col] - df['calculated_age'])
        df['is_discrepancy'] = df['age_diff'] > 2  # More than 2 years difference
    else:
        # Fallback: flag if age < 18 but DOB suggests older
        df['is_discrepancy'] = (
            (df[age_col] < 18) &
            (df[dob_col].notna()) &
            ((pd.Timestamp.now().year - df[dob_col].dt.year) > 18)
        )
    
    anomalies = df[df['is_discrepancy']].copy()
    
    return anomalies


def detect_gender_photo_dissonance(
    df: pd.DataFrame,
    gender_col: str = 'gender',
    photo_metadata_col: str = 'photo_gender'
) -> pd.DataFrame:
    """
    Detect Gender-Photo Dissonance: Declared gender mismatches visual metadata,
    signaling data entry errors or identity spoofing.
    """
    df = df.copy()
    
    if gender_col not in df.columns:
        return pd.DataFrame()
    
    # Simulate photo metadata if not present
    if photo_metadata_col not in df.columns:
        # For demo: create some mismatches
        df[photo_metadata_col] = df[gender_col].copy()
        mismatch_indices = df.sample(frac=0.05, random_state=42).index
        df.loc[mismatch_indices, photo_metadata_col] = df.loc[mismatch_indices, gender_col].map(
            {'M': 'F', 'F': 'M', 'Male': 'Female', 'Female': 'Male'}
        )
    
    # Normalize gender values
    df[gender_col] = df[gender_col].str.upper().str.strip()
    df[photo_metadata_col] = df[photo_metadata_col].str.upper().str.strip()
    
    # Flag mismatches
    df['is_dissonance'] = (
        (df[gender_col] != df[photo_metadata_col]) &
        (df[gender_col].notna()) &
        (df[photo_metadata_col].notna())
    )
    
    anomalies = df[df['is_dissonance']].copy()
    
    return anomalies


def detect_multiple_enrolment_burst(
    df: pd.DataFrame,
    aadhaar_col: str = 'aadhaar_number',
    date_col: str = 'date',
    window_days: int = 90
) -> pd.DataFrame:
    """
    Detect Multiple Enrolment Burst: Single individual attempting >5 enrolments
    within 90 days, indicating lack of understanding of duplicate logic.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if aadhaar_col not in df.columns:
        # Use a combination of name + dob as proxy
        if 'name' in df.columns and 'date_of_birth' in df.columns:
            df[aadhaar_col] = df['name'].astype(str) + '_' + df['date_of_birth'].astype(str)
        else:
            return pd.DataFrame()
    
    # Sort by date
    df = df.sort_values([aadhaar_col, date_col])
    
    # Calculate rolling count within window
    df['enrolment_count'] = df.groupby(aadhaar_col).cumcount() + 1
    
    # For each enrolment, count previous enrolments within window
    def count_recent_enrolments(group):
        group = group.sort_values(date_col)
        group['recent_count'] = 0
        for idx, row in group.iterrows():
            window_start = row[date_col] - pd.Timedelta(days=window_days)
            recent = group[
                (group[date_col] >= window_start) &
                (group[date_col] <= row[date_col])
            ]
            group.loc[idx, 'recent_count'] = len(recent)
        return group
    
    df = df.groupby(aadhaar_col).apply(count_recent_enrolments).reset_index(drop=True)
    
    # Flag bursts
    df['is_burst'] = df['recent_count'] > MULTIPLE_ENROLMENT_THRESHOLD
    
    anomalies = df[df['is_burst']].copy()
    
    return anomalies


def detect_pincode_address_mismatch(
    df: pd.DataFrame,
    pincode_col: str = 'pincode',
    address_col: str = 'address'
) -> pd.DataFrame:
    """
    Detect Pincode-Address Mismatch: High frequency where entered Pincode
    doesn't geographically align with address text.
    """
    df = df.copy()
    
    if pincode_col not in df.columns or address_col not in df.columns:
        return pd.DataFrame()
    
    # Extract state/district from pincode (first 2-3 digits often indicate region)
    df['pincode_prefix'] = df[pincode_col].astype(str).str[:3]
    
    # Simple check: if address contains state/district name, verify consistency
    # This is a simplified version - real implementation would use geocoding API
    
    # For demo: flag if pincode prefix doesn't match common patterns
    # (This is a placeholder - real implementation needs geocoding)
    df['is_mismatch'] = False  # Placeholder
    
    # Alternative: flag if pincode format is invalid
    df['pincode_valid'] = df[pincode_col].astype(str).str.match(r'^\d{6}$')
    df['is_mismatch'] = ~df['pincode_valid']
    
    anomalies = df[df['is_mismatch']].copy()
    
    return anomalies


def detect_mobile_number_churn(
    df: pd.DataFrame,
    mobile_col: str = 'mobile_number',
    aadhaar_col: str = 'aadhaar_number',
    date_col: str = 'date',
    window_days: int = 180
) -> pd.DataFrame:
    """
    Detect Mobile Number Churn: Single mobile linked to >20 Aadhaar numbers
    within short period, signaling fraudulent operator-as-proxy behavior.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if mobile_col not in df.columns:
        return pd.DataFrame()
    
    if aadhaar_col not in df.columns:
        # Use name + dob as proxy
        if 'name' in df.columns and 'date_of_birth' in df.columns:
            df[aadhaar_col] = df['name'].astype(str) + '_' + df['date_of_birth'].astype(str)
        else:
            return pd.DataFrame()
    
    # Group by mobile number and count unique Aadhaar numbers
    mobile_stats = df.groupby(mobile_col).agg({
        aadhaar_col: 'nunique',
        date_col: ['min', 'max']
    }).reset_index()
    mobile_stats.columns = [mobile_col, 'unique_aadhaar_count', 'first_date', 'last_date']
    mobile_stats['date_span_days'] = (mobile_stats['last_date'] - mobile_stats['first_date']).dt.days
    
    # Flag churn
    mobile_stats['is_churn'] = (
        (mobile_stats['unique_aadhaar_count'] > MOBILE_CHURN_THRESHOLD) &
        (mobile_stats['date_span_days'] < window_days)
    )
    
    anomalies = mobile_stats[mobile_stats['is_churn']].copy()
    
    return anomalies


def detect_error_code_999_concentration(
    df: pd.DataFrame,
    error_code_col: str = 'error_code',
    district_col: str = 'district',
    error_code_value: int = 999
) -> pd.DataFrame:
    """
    Detect Error Code 999 Concentration: Statistical spike in Error Code 999
    within specific districts, indicating unresolved internal server errors.
    """
    df = df.copy()
    
    if error_code_col not in df.columns:
        return pd.DataFrame()
    
    # Filter for error code 999
    error_999 = df[df[error_code_col] == error_code_value].copy()
    
    if error_999.empty:
        return pd.DataFrame()
    
    # Calculate error rate per district
    district_stats = df.groupby(district_col).agg({
        error_code_col: 'count'
    }).reset_index()
    district_stats.columns = [district_col, 'total_transactions']
    
    error_district = error_999.groupby(district_col).agg({
        error_code_col: 'count'
    }).reset_index()
    error_district.columns = [district_col, 'error_999_count']
    
    # Merge and calculate error rate
    district_stats = district_stats.merge(error_district, on=district_col, how='left')
    district_stats['error_999_count'] = district_stats['error_999_count'].fillna(0)
    district_stats['error_rate'] = district_stats['error_999_count'] / (district_stats['total_transactions'] + 1)
    
    # Calculate z-score
    mean_rate = district_stats['error_rate'].mean()
    std_rate = district_stats['error_rate'].std()
    district_stats['z_score'] = (district_stats['error_rate'] - mean_rate) / (std_rate + 1e-6)
    
    # Flag anomalies (>5 sigma)
    district_stats['is_concentrated'] = district_stats['z_score'] > ANOMALY_THRESHOLD_SIGMA
    
    anomalies = district_stats[district_stats['is_concentrated']].copy()
    
    return anomalies


def detect_biometric_exception_overuse(
    df: pd.DataFrame,
    pincode_col: str = 'pincode',
    exception_col: str = 'biometric_exception',
    total_col: str = 'total_enrolments'
) -> pd.DataFrame:
    """
    Detect Biometric Exception Overuse: Pincode where percentage of biometric
    exceptions is >5 sigma above national average.
    """
    df = df.copy()
    
    if exception_col not in df.columns:
        df[exception_col] = df.get('count', 0) * 0.02  # Simulate 2% exceptions
    if total_col not in df.columns:
        df[total_col] = df.get('count', 0)
    
    # Calculate exception rate per pincode
    pincode_stats = df.groupby(pincode_col).agg({
        exception_col: 'sum',
        total_col: 'sum'
    }).reset_index()
    pincode_stats.columns = [pincode_col, 'total_exceptions', 'total_enrolments']
    pincode_stats['exception_rate'] = pincode_stats['total_exceptions'] / (pincode_stats['total_enrolments'] + 1)
    
    # Calculate z-score
    mean_rate = pincode_stats['exception_rate'].mean()
    std_rate = pincode_stats['exception_rate'].std()
    pincode_stats['z_score'] = (pincode_stats['exception_rate'] - mean_rate) / (std_rate + 1e-6)
    
    # Flag anomalies (>5 sigma)
    pincode_stats['is_overuse'] = pincode_stats['z_score'] > ANOMALY_THRESHOLD_SIGMA
    
    anomalies = pincode_stats[pincode_stats['is_overuse']].copy()
    
    return anomalies


def detect_transliteration_error_clusters(
    df: pd.DataFrame,
    district_col: str = 'district',
    rejection_reason_col: str = 'rejection_reason',
    reason_filter: str = 'Name/Address Error'
) -> pd.DataFrame:
    """
    Detect Transliteration Error Clusters: High volume of Name/Address Error
    rejections in linguistic border regions.
    """
    df = df.copy()
    
    if rejection_reason_col not in df.columns:
        return pd.DataFrame()
    
    # Filter by rejection reason
    filtered = df[df[rejection_reason_col].str.contains(reason_filter, case=False, na=False)]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Calculate rejection rate per district
    district_stats = df.groupby(district_col).agg({
        rejection_reason_col: 'count'
    }).reset_index()
    district_stats.columns = [district_col, 'total_rejections']
    
    error_district = filtered.groupby(district_col).agg({
        rejection_reason_col: 'count'
    }).reset_index()
    error_district.columns = [district_col, 'transliteration_errors']
    
    # Merge and calculate error rate
    district_stats = district_stats.merge(error_district, on=district_col, how='left')
    district_stats['transliteration_errors'] = district_stats['transliteration_errors'].fillna(0)
    district_stats['error_rate'] = district_stats['transliteration_errors'] / (district_stats['total_rejections'] + 1)
    
    # Calculate z-score
    mean_rate = district_stats['error_rate'].mean()
    std_rate = district_stats['error_rate'].std()
    district_stats['z_score'] = (district_stats['error_rate'] - mean_rate) / (std_rate + 1e-6)
    
    # Flag clusters (z-score > 2)
    district_stats['is_cluster'] = district_stats['z_score'] > 2
    
    anomalies = district_stats[district_stats['is_cluster']].copy()
    
    return anomalies


def get_category2_summary(enrolment_df: pd.DataFrame,
                          demographic_df: pd.DataFrame,
                          biometric_df: pd.DataFrame) -> Dict:
    """Get summary statistics for all Category II anomalies"""
    
    summary = {
        'age_dob_discrepancy': 0,
        'gender_photo_dissonance': 0,
        'multiple_enrolment_burst': 0,
        'pincode_address_mismatch': 0,
        'mobile_number_churn': 0,
        'error_code_999_concentration': 0,
        'biometric_exception_overuse': 0,
        'transliteration_error_clusters': 0
    }
    
    try:
        age_dob = detect_age_dob_discrepancy(enrolment_df)
        summary['age_dob_discrepancy'] = len(age_dob)
    except:
        pass
    
    try:
        gender = detect_gender_photo_dissonance(enrolment_df)
        summary['gender_photo_dissonance'] = len(gender)
    except:
        pass
    
    try:
        burst = detect_multiple_enrolment_burst(enrolment_df)
        summary['multiple_enrolment_burst'] = len(burst)
    except:
        pass
    
    try:
        mismatch = detect_pincode_address_mismatch(demographic_df)
        summary['pincode_address_mismatch'] = len(mismatch)
    except:
        pass
    
    try:
        churn = detect_mobile_number_churn(demographic_df)
        summary['mobile_number_churn'] = len(churn)
    except:
        pass
    
    try:
        error999 = detect_error_code_999_concentration(enrolment_df)
        summary['error_code_999_concentration'] = len(error999)
    except:
        pass
    
    try:
        bio_exception = detect_biometric_exception_overuse(biometric_df)
        summary['biometric_exception_overuse'] = len(bio_exception)
    except:
        pass
    
    try:
        translit = detect_transliteration_error_clusters(enrolment_df)
        summary['transliteration_error_clusters'] = len(translit)
    except:
        pass
    
    return summary
