"""Category IV: Technical Infrastructure and External Factors"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats


def detect_power_network_outage_signature(
    df: pd.DataFrame,
    date_col: str = 'date',
    pincode_col: str = 'pincode',
    activity_col: str = 'count',
    hour_col: str = 'hour'
) -> pd.DataFrame:
    """
    Detect Power/Network Outage Signature: Activity pattern mirroring
    load-shedding schedules, with uploads only in specific 4-hour windows.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if hour_col not in df.columns:
        df[hour_col] = df[date_col].dt.hour
    
    # Group by pincode and hour
    pincode_hour = df.groupby([pincode_col, hour_col]).agg({
        activity_col: 'mean'
    }).reset_index()
    
    # Calculate activity distribution across hours
    pincode_stats = pincode_hour.groupby(pincode_col).agg({
        activity_col: ['sum', 'std', 'count']
    }).reset_index()
    pincode_stats.columns = [pincode_col, 'total_activity', 'activity_std', 'hour_count']
    
    # Calculate concentration (if activity is concentrated in few hours)
    pincode_hour_pivot = pincode_hour.pivot(index=pincode_col, columns=hour_col, values=activity_col).fillna(0)
    pincode_hour_pivot['max_hour_activity'] = pincode_hour_pivot.max(axis=1)
    pincode_hour_pivot['total_activity'] = pincode_hour_pivot.sum(axis=1)
    pincode_hour_pivot['concentration'] = pincode_hour_pivot['max_hour_activity'] / (pincode_hour_pivot['total_activity'] + 1)
    
    # Flag if >60% activity in 4-hour window (indicating outage pattern)
    pincode_hour_pivot = pincode_hour_pivot.reset_index()
    pincode_hour_pivot['is_outage'] = pincode_hour_pivot['concentration'] > 0.6
    
    anomalies = pincode_hour_pivot[pincode_hour_pivot['is_outage']].copy()
    
    return anomalies


def detect_device_reputation_failure(
    df: pd.DataFrame,
    device_id_col: str = 'device_id',
    error_code_col: str = 'error_code',
    error_code_value: int = 521  # Invalid Device Code
) -> pd.DataFrame:
    """
    Detect Device Reputation Failure: High volume of Error Code 521 from
    specific batch of biometric scanners, indicating hardware defect.
    """
    df = df.copy()
    
    if error_code_col not in df.columns or device_id_col not in df.columns:
        return pd.DataFrame()
    
    # Filter for error code 521
    error_521 = df[df[error_code_col] == error_code_value].copy()
    
    if error_521.empty:
        return pd.DataFrame()
    
    # Calculate error rate per device
    device_stats = df.groupby(device_id_col).agg({
        error_code_col: 'count'
    }).reset_index()
    device_stats.columns = [device_id_col, 'total_transactions']
    
    error_device = error_521.groupby(device_id_col).agg({
        error_code_col: 'count'
    }).reset_index()
    error_device.columns = [device_id_col, 'error_521_count']
    
    # Merge and calculate error rate
    device_stats = device_stats.merge(error_device, on=device_id_col, how='left')
    device_stats['error_521_count'] = device_stats['error_521_count'].fillna(0)
    device_stats['error_rate'] = device_stats['error_521_count'] / (device_stats['total_transactions'] + 1)
    
    # Calculate z-score
    mean_rate = device_stats['error_rate'].mean()
    std_rate = device_stats['error_rate'].std()
    device_stats['z_score'] = (device_stats['error_rate'] - mean_rate) / (std_rate + 1e-6)
    
    # Flag failures (z-score > 3)
    device_stats['is_failure'] = device_stats['z_score'] > 3
    
    anomalies = device_stats[device_stats['is_failure']].copy()
    
    return anomalies


def detect_latency_induced_timeouts(
    df: pd.DataFrame,
    district_col: str = 'district',
    error_code_col: str = 'error_code',
    error_code_value: int = 561  # Request Expired
) -> pd.DataFrame:
    """
    Detect Latency-Induced Timeouts: Cluster of Error Code 561 in hill districts,
    indicating network round-trip time exceeds server threshold.
    """
    df = df.copy()
    
    if error_code_col not in df.columns:
        return pd.DataFrame()
    
    # Filter for error code 561
    error_561 = df[df[error_code_col] == error_code_value].copy()
    
    if error_561.empty:
        return pd.DataFrame()
    
    # Calculate error rate per district
    district_stats = df.groupby(district_col).agg({
        error_code_col: 'count'
    }).reset_index()
    district_stats.columns = [district_col, 'total_transactions']
    
    error_district = error_561.groupby(district_col).agg({
        error_code_col: 'count'
    }).reset_index()
    error_district.columns = [district_col, 'error_561_count']
    
    # Merge and calculate error rate
    district_stats = district_stats.merge(error_district, on=district_col, how='left')
    district_stats['error_561_count'] = district_stats['error_561_count'].fillna(0)
    district_stats['error_rate'] = district_stats['error_561_count'] / (district_stats['total_transactions'] + 1)
    
    # Calculate z-score
    mean_rate = district_stats['error_rate'].mean()
    std_rate = district_stats['error_rate'].std()
    district_stats['z_score'] = (district_stats['error_rate'] - mean_rate) / (std_rate + 1e-6)
    
    # Flag timeouts (z-score > 2)
    district_stats['is_timeout'] = district_stats['z_score'] > 2
    
    anomalies = district_stats[district_stats['is_timeout']].copy()
    
    return anomalies


def detect_census_anomaly_saturation(
    df: pd.DataFrame,
    district_col: str = 'district',
    activity_col: str = 'count',
    population_col: str = 'population'
) -> pd.DataFrame:
    """
    Detect Census Anomaly: Districts where active Aadhaar cards exceed
    2011 Census-based population projections (saturation >100%).
    """
    df = df.copy()
    
    # Calculate total Aadhaar per district
    district_stats = df.groupby(district_col).agg({
        activity_col: 'sum'
    }).reset_index()
    district_stats.columns = [district_col, 'total_aadhaar']
    
    # If population data available, calculate saturation
    if population_col in df.columns:
        district_pop = df.groupby(district_col)[population_col].first().reset_index()
        district_stats = district_stats.merge(district_pop, on=district_col, how='left')
        district_stats['saturation'] = district_stats['total_aadhaar'] / (district_stats[population_col] + 1)
        
        # Flag anomalies (>100% saturation)
        district_stats['is_anomaly'] = district_stats['saturation'] > 1.0
        
        anomalies = district_stats[district_stats['is_anomaly']].copy()
    else:
        # Without population data, flag districts with unusually high counts
        mean_count = district_stats['total_aadhaar'].mean()
        std_count = district_stats['total_aadhaar'].std()
        district_stats['z_score'] = (district_stats['total_aadhaar'] - mean_count) / (std_count + 1e-6)
        district_stats['is_anomaly'] = district_stats['z_score'] > 3
        
        anomalies = district_stats[district_stats['is_anomaly']].copy()
    
    return anomalies


def detect_deceased_id_persistence(
    df: pd.DataFrame,
    aadhaar_col: str = 'aadhaar_number',
    date_col: str = 'date',
    activity_col: str = 'count',
    deactivation_date_col: str = 'deactivation_date'
) -> pd.DataFrame:
    """
    Detect Deceased ID Persistence: Authentications from numbers that should
    have been deactivated following death, indicating CRS integration lag.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if deactivation_date_col not in df.columns:
        # Simulate: flag if activity continues after a certain date
        # In real implementation, this would cross-reference with CRS data
        return pd.DataFrame()
    
    df[deactivation_date_col] = pd.to_datetime(df[deactivation_date_col], errors='coerce')
    
    # Flag if activity occurs after deactivation date
    df['is_persistent'] = (
        (df[deactivation_date_col].notna()) &
        (df[date_col] > df[deactivation_date_col])
    )
    
    anomalies = df[df['is_persistent']].copy()
    
    return anomalies


def detect_voter_id_linking_spikes(
    df: pd.DataFrame,
    date_col: str = 'date',
    update_type_col: str = 'update_type',
    activity_col: str = 'count'
) -> pd.DataFrame:
    """
    Detect Voter ID Linking Spikes: Sudden bursts of update activity following
    election commission announcements, providing proxy for policy-driven stress.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year_month'] = df[date_col].dt.to_period('M')
    
    # Filter for voter ID related updates
    if update_type_col in df.columns:
        voter_updates = df[df[update_type_col].str.contains('voter|election', case=False, na=False)]
    else:
        voter_updates = df.copy()
    
    # Calculate monthly activity
    monthly = voter_updates.groupby('year_month').agg({
        activity_col: 'sum'
    }).reset_index()
    monthly = monthly.sort_values('year_month')
    
    # Calculate month-over-month change
    monthly['prev_month'] = monthly[activity_col].shift(1)
    monthly['mom_change'] = (monthly[activity_col] - monthly['prev_month']) / (monthly['prev_month'] + 1)
    
    # Calculate z-score
    mean_change = monthly['mom_change'].mean()
    std_change = monthly['mom_change'].std()
    monthly['z_score'] = (monthly['mom_change'] - mean_change) / (std_change + 1e-6)
    
    # Flag spikes (z-score > 3)
    monthly['is_spike'] = monthly['z_score'] > 3
    
    anomalies = monthly[monthly['is_spike']].copy()
    
    return anomalies


def get_category4_summary(enrolment_df: pd.DataFrame,
                          demographic_df: pd.DataFrame,
                          biometric_df: pd.DataFrame) -> Dict:
    """Get summary statistics for all Category IV anomalies"""
    
    summary = {
        'power_network_outage': 0,
        'device_reputation_failure': 0,
        'latency_induced_timeouts': 0,
        'census_anomaly_saturation': 0,
        'deceased_id_persistence': 0,
        'voter_id_linking_spikes': 0
    }
    
    try:
        outage = detect_power_network_outage_signature(enrolment_df)
        summary['power_network_outage'] = len(outage)
    except:
        pass
    
    try:
        device = detect_device_reputation_failure(enrolment_df)
        summary['device_reputation_failure'] = len(device)
    except:
        pass
    
    try:
        timeout = detect_latency_induced_timeouts(enrolment_df)
        summary['latency_induced_timeouts'] = len(timeout)
    except:
        pass
    
    try:
        census = detect_census_anomaly_saturation(enrolment_df)
        summary['census_anomaly_saturation'] = len(census)
    except:
        pass
    
    try:
        deceased = detect_deceased_id_persistence(enrolment_df)
        summary['deceased_id_persistence'] = len(deceased)
    except:
        pass
    
    try:
        voter = detect_voter_id_linking_spikes(demographic_df)
        summary['voter_id_linking_spikes'] = len(voter)
    except:
        pass
    
    return summary
