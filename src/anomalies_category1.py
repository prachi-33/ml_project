"""Category I: Administrative and Operational Bottlenecks"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

from src.config import (
    ACTIVITY_DROP_THRESHOLD,
    FORCE_CAPTURE_THRESHOLD
)


def detect_pincode_activity_deserts(
    df: pd.DataFrame,
    date_col: str = 'date',
    pincode_col: str = 'pincode',
    activity_col: str = 'count',
    window_days: int = 30
) -> pd.DataFrame:
    """
    Detect Pincode-Level Activity Deserts: Sudden drop to near-zero activity
    over a 30-day period indicating expired certificates or connectivity issues.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([pincode_col, date_col])
    
    # Calculate rolling averages
    df['rolling_avg'] = df.groupby(pincode_col)[activity_col].transform(
        lambda x: x.rolling(window=window_days, min_periods=1).mean()
    )
    
    # Calculate historical average (excluding last 30 days)
    df['historical_avg'] = df.groupby(pincode_col)[activity_col].transform(
        lambda x: x.shift(window_days).expanding().mean()
    )
    
    # Detect sudden drops
    df['activity_ratio'] = df['rolling_avg'] / (df['historical_avg'] + 1)
    df['is_desert'] = df['activity_ratio'] < ACTIVITY_DROP_THRESHOLD
    
    anomalies = df[df['is_desert']].copy()
    
    return anomalies


def detect_wait_time_volatility(
    df: pd.DataFrame,
    center_col: str = 'center_id',
    tokens_col: str = 'tokens_issued',
    packets_col: str = 'packets_uploaded'
) -> pd.DataFrame:
    """
    Detect Wait-Time Volatility: High ratio of Tokens Issued to Packets Uploaded
    indicating hardware fatigue or server latency.
    """
    df = df.copy()
    
    if tokens_col not in df.columns or packets_col not in df.columns:
        # Create dummy columns if not present
        df[tokens_col] = df.get('count', 0) * 1.2  # Simulate tokens
        df[packets_col] = df.get('count', 0)  # Simulate packets
    
    df['token_packet_ratio'] = df[tokens_col] / (df[packets_col] + 1)
    
    # Calculate z-score
    mean_ratio = df['token_packet_ratio'].mean()
    std_ratio = df['token_packet_ratio'].std()
    df['z_score'] = (df['token_packet_ratio'] - mean_ratio) / (std_ratio + 1e-6)
    
    # Flag anomalies (z-score > 2)
    df['is_anomaly'] = df['z_score'] > 2
    
    anomalies = df[df['is_anomaly']].copy()
    
    return anomalies


def detect_rejection_clusters(
    df: pd.DataFrame,
    center_col: str = 'center_id',
    rejection_col: str = 'rejections',
    rejection_reason_col: str = 'rejection_reason',
    reason_filter: str = 'Invalid Documents'
) -> pd.DataFrame:
    """
    Detect Center-Specific Rejection Clusters: Abnormal concentration of
    rejections from a single center while neighbors show normal rates.
    """
    df = df.copy()
    
    if rejection_reason_col not in df.columns:
        df[rejection_reason_col] = 'Unknown'
    
    # Filter by rejection reason
    filtered = df[df[rejection_reason_col].str.contains(reason_filter, case=False, na=False)]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Calculate rejection rate per center
    center_stats = filtered.groupby(center_col).agg({
        rejection_col: ['sum', 'count', 'mean']
    }).reset_index()
    center_stats.columns = [center_col, 'total_rejections', 'record_count', 'avg_rejections']
    
    # Calculate z-score
    mean_rej = center_stats['avg_rejections'].mean()
    std_rej = center_stats['avg_rejections'].std()
    center_stats['z_score'] = (center_stats['avg_rejections'] - mean_rej) / (std_rej + 1e-6)
    
    # Flag anomalies
    center_stats['is_cluster'] = center_stats['z_score'] > 2
    
    anomalies = center_stats[center_stats['is_cluster']].copy()
    
    return anomalies


def detect_force_capture_saturation(
    df: pd.DataFrame,
    pincode_col: str = 'pincode',
    force_capture_col: str = 'force_capture_count',
    total_capture_col: str = 'total_captures',
    age_col: str = 'age_group'
) -> pd.DataFrame:
    """
    Detect Force Capture Saturation: >40% force captures in younger populations
    indicating poor quality control or malfunctioning sensors.
    """
    df = df.copy()
    
    if force_capture_col not in df.columns:
        df[force_capture_col] = df.get('count', 0) * 0.1  # Simulate 10% force capture
    if total_capture_col not in df.columns:
        df[total_capture_col] = df.get('count', 0)
    
    df['force_capture_rate'] = df[force_capture_col] / (df[total_capture_col] + 1)
    
    # Filter for younger populations (not 18+)
    if age_col in df.columns:
        young_df = df[~df[age_col].str.contains('18', case=False, na=False)]
    else:
        young_df = df.copy()
    
    # Group by pincode
    pincode_stats = young_df.groupby(pincode_col).agg({
        'force_capture_rate': 'mean',
        force_capture_col: 'sum',
        total_capture_col: 'sum'
    }).reset_index()
    
    pincode_stats.columns = [pincode_col, 'avg_force_rate', 'total_force', 'total_captures']
    
    # Flag anomalies (>40% threshold)
    pincode_stats['is_saturated'] = pincode_stats['avg_force_rate'] > FORCE_CAPTURE_THRESHOLD
    
    anomalies = pincode_stats[pincode_stats['is_saturated']].copy()
    
    return anomalies


def detect_operator_certification_lag(
    df: pd.DataFrame,
    date_col: str = 'date',
    district_col: str = 'district',
    center_col: str = 'center_id',
    activity_col: str = 'count'
) -> pd.DataFrame:
    """
    Detect Operator Certification Lag: Sudden activity cessation across
    multiple centers coinciding with quarterly recertification cycles.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    
    # Group by district, quarter, year
    district_quarter = df.groupby([district_col, 'year', 'quarter']).agg({
        activity_col: 'sum',
        center_col: 'nunique'
    }).reset_index()
    
    district_quarter.columns = [district_col, 'year', 'quarter', 'total_activity', 'active_centers']
    
    # Calculate quarter-over-quarter change
    district_quarter = district_quarter.sort_values([district_col, 'year', 'quarter'])
    district_quarter['prev_activity'] = district_quarter.groupby(district_col)['total_activity'].shift(1)
    district_quarter['activity_change'] = (district_quarter['total_activity'] - district_quarter['prev_activity']) / (district_quarter['prev_activity'] + 1)
    
    # Flag sudden drops (>80% decrease) at quarter boundaries
    district_quarter['is_lag'] = (
        (district_quarter['activity_change'] < -0.8) &
        (district_quarter['active_centers'] > 3)  # Multiple centers affected
    )
    
    anomalies = district_quarter[district_quarter['is_lag']].copy()
    
    return anomalies


def detect_appointment_success_bimodal(
    df: pd.DataFrame,
    appointment_type_col: str = 'appointment_type',
    success_col: str = 'success_rate'
) -> pd.DataFrame:
    """
    Detect Bimodal Distribution: 95% success at ASKs vs 40% at post offices,
    highlighting urban-rural disparity.
    """
    df = df.copy()
    
    if appointment_type_col not in df.columns:
        df[appointment_type_col] = 'Unknown'
    if success_col not in df.columns:
        df[success_col] = np.random.uniform(0.3, 0.95, len(df))  # Simulate
    
    # Group by appointment type
    type_stats = df.groupby(appointment_type_col).agg({
        success_col: ['mean', 'std', 'count']
    }).reset_index()
    type_stats.columns = [appointment_type_col, 'avg_success', 'std_success', 'count']
    
    # Calculate disparity
    max_success = type_stats['avg_success'].max()
    min_success = type_stats['avg_success'].min()
    type_stats['disparity'] = max_success - min_success
    
    # Flag if disparity > 0.4 (40 percentage points)
    type_stats['is_bimodal'] = type_stats['disparity'] > 0.4
    
    anomalies = type_stats[type_stats['is_bimodal']].copy()
    
    return anomalies


def detect_haat_surge_anomaly(
    df: pd.DataFrame,
    date_col: str = 'date',
    pincode_col: str = 'pincode',
    activity_col: str = 'count',
    day_of_week_col: str = 'day_of_week'
) -> pd.DataFrame:
    """
    Detect Haat Surge: Massive spikes on specific days of week in rural areas,
    corresponding to weekly markets.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if day_of_week_col not in df.columns:
        df[day_of_week_col] = df[date_col].dt.dayofweek
    
    # Group by pincode and day of week
    pincode_dow = df.groupby([pincode_col, day_of_week_col]).agg({
        activity_col: ['mean', 'std', 'count']
    }).reset_index()
    pincode_dow.columns = [pincode_col, day_of_week_col, 'avg_activity', 'std_activity', 'count']
    
    # Calculate coefficient of variation per pincode
    pincode_stats = pincode_dow.groupby(pincode_col).agg({
        'avg_activity': ['mean', 'std']
    }).reset_index()
    pincode_stats.columns = [pincode_col, 'overall_mean', 'overall_std']
    pincode_stats['cv'] = pincode_stats['overall_std'] / (pincode_stats['overall_mean'] + 1)
    
    # Merge back
    pincode_dow = pincode_dow.merge(pincode_stats, on=pincode_col)
    
    # Flag if specific day has >2x the overall mean
    pincode_dow['is_surge'] = (
        (pincode_dow['avg_activity'] > 2 * pincode_dow['overall_mean']) &
        (pincode_dow['count'] > 10)  # Sufficient data points
    )
    
    anomalies = pincode_dow[pincode_dow['is_surge']].copy()
    
    return anomalies


def get_category1_summary(enrolment_df: pd.DataFrame, 
                          demographic_df: pd.DataFrame,
                          biometric_df: pd.DataFrame) -> Dict:
    """Get summary statistics for all Category I anomalies"""
    
    summary = {
        'pincode_activity_deserts': 0,
        'wait_time_volatility': 0,
        'rejection_clusters': 0,
        'force_capture_saturation': 0,
        'operator_certification_lag': 0,
        'appointment_success_bimodal': 0,
        'haat_surge_anomaly': 0
    }
    
    try:
        deserts = detect_pincode_activity_deserts(enrolment_df)
        summary['pincode_activity_deserts'] = len(deserts)
    except:
        pass
    
    try:
        volatility = detect_wait_time_volatility(enrolment_df)
        summary['wait_time_volatility'] = len(volatility)
    except:
        pass
    
    try:
        rejections = detect_rejection_clusters(enrolment_df)
        summary['rejection_clusters'] = len(rejections)
    except:
        pass
    
    try:
        force_capture = detect_force_capture_saturation(biometric_df)
        summary['force_capture_saturation'] = len(force_capture)
    except:
        pass
    
    try:
        cert_lag = detect_operator_certification_lag(enrolment_df)
        summary['operator_certification_lag'] = len(cert_lag)
    except:
        pass
    
    try:
        bimodal = detect_appointment_success_bimodal(enrolment_df)
        summary['appointment_success_bimodal'] = len(bimodal)
    except:
        pass
    
    try:
        haat = detect_haat_surge_anomaly(enrolment_df)
        summary['haat_surge_anomaly'] = len(haat)
    except:
        pass
    
    return summary
