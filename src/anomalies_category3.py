"""Category III: Social and Behavioral Anomalies"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats

from src.config import ANOMALY_THRESHOLD_SIGMA


def detect_migration_trail_address_spike(
    df: pd.DataFrame,
    date_col: str = 'date',
    pincode_col: str = 'pincode',
    update_type_col: str = 'update_type',
    activity_col: str = 'count'
) -> pd.DataFrame:
    """
    Detect Migration Trail Address Spike: Massive address update volumes in
    urban Pincodes during March and November, correlating with seasonal migration.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    
    # Filter for address updates
    if update_type_col in df.columns:
        address_updates = df[df[update_type_col].str.contains('address', case=False, na=False)]
    else:
        address_updates = df.copy()
    
    # Focus on migration months (March=3, November=11)
    migration_months = address_updates[address_updates['month'].isin([3, 11])].copy()
    
    if migration_months.empty:
        return pd.DataFrame()
    
    # Group by pincode and month
    pincode_month = migration_months.groupby([pincode_col, 'month', 'year']).agg({
        activity_col: 'sum'
    }).reset_index()
    
    # Calculate average for non-migration months
    non_migration = address_updates[~address_updates['month'].isin([3, 11])]
    pincode_avg = non_migration.groupby(pincode_col).agg({
        activity_col: 'mean'
    }).reset_index()
    pincode_avg.columns = [pincode_col, 'avg_non_migration']
    
    # Merge and calculate spike ratio
    pincode_month = pincode_month.merge(pincode_avg, on=pincode_col, how='left')
    pincode_month['spike_ratio'] = pincode_month[activity_col] / (pincode_month['avg_non_migration'] + 1)
    
    # Flag spikes (>2x average)
    pincode_month['is_spike'] = pincode_month['spike_ratio'] > 2.0
    
    anomalies = pincode_month[pincode_month['is_spike']].copy()
    
    return anomalies


def detect_reverse_migration_patterns(
    df: pd.DataFrame,
    date_col: str = 'date',
    pincode_col: str = 'pincode',
    activity_col: str = 'count'
) -> pd.DataFrame:
    """
    Detect Reverse Migration Patterns: Anomalous address updates back to
    rural Pincodes during 2020-21, providing proxy for socio-economic shocks.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    
    # Focus on 2020-2021 period
    covid_period = df[df['year'].isin([2020, 2021])].copy()
    
    if covid_period.empty:
        return pd.DataFrame()
    
    # Calculate year-over-year change
    yearly = df.groupby([pincode_col, 'year']).agg({
        activity_col: 'sum'
    }).reset_index()
    
    yearly = yearly.sort_values([pincode_col, 'year'])
    yearly['prev_year'] = yearly.groupby(pincode_col)[activity_col].shift(1)
    yearly['yoy_change'] = (yearly[activity_col] - yearly['prev_year']) / (yearly['prev_year'] + 1)
    
    # Flag reverse migration (positive growth in 2020-21 after negative/stable trend)
    covid_years = yearly[yearly['year'].isin([2020, 2021])]
    pre_covid = yearly[yearly['year'] < 2020]
    
    if pre_covid.empty:
        return pd.DataFrame()
    
    pre_covid_avg = pre_covid.groupby(pincode_col).agg({
        'yoy_change': 'mean'
    }).reset_index()
    pre_covid_avg.columns = [pincode_col, 'pre_covid_trend']
    
    covid_years = covid_years.merge(pre_covid_avg, on=pincode_col, how='left')
    covid_years['is_reverse'] = (
        (covid_years['yoy_change'] > 0.2) &  # 20%+ growth
        (covid_years['pre_covid_trend'] <= 0)  # Was declining/stable
    )
    
    anomalies = covid_years[covid_years['is_reverse']].copy()
    
    return anomalies


def detect_baal_aadhaar_deactivation_wave(
    df: pd.DataFrame,
    age_col: str = 'age',
    date_col: str = 'date',
    deactivation_col: str = 'deactivation_count',
    activity_col: str = 'count'
) -> pd.DataFrame:
    """
    Detect Baal Aadhaar Deactivation Wave: High frequency of deactivations
    among 7-year-olds who never completed MBU-1.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter for age 7 (or 6-8 range)
    if age_col in df.columns:
        target_age = df[(df[age_col] >= 6) & (df[age_col] <= 8)]
    else:
        return pd.DataFrame()
    
    if deactivation_col not in df.columns:
        # Use activity_col as proxy if deactivation not available
        target_age[deactivation_col] = target_age.get(activity_col, 0) * 0.1
    
    # Group by age and calculate deactivation rate
    age_stats = target_age.groupby(age_col).agg({
        deactivation_col: ['sum', 'count']
    }).reset_index()
    age_stats.columns = [age_col, 'total_deactivations', 'record_count']
    age_stats['deactivation_rate'] = age_stats['total_deactivations'] / (age_stats['record_count'] + 1)
    
    # Flag if age 7 has significantly higher rate
    if 7 in age_stats[age_col].values:
        age7_rate = age_stats[age_stats[age_col] == 7]['deactivation_rate'].values[0]
        other_avg = age_stats[age_stats[age_col] != 7]['deactivation_rate'].mean()
        age_stats['is_wave'] = (age_stats[age_col] == 7) & (age7_rate > 1.5 * other_avg)
    else:
        age_stats['is_wave'] = False
    
    anomalies = age_stats[age_stats['is_wave']].copy()
    
    return anomalies


def detect_laborer_biometric_attrition(
    df: pd.DataFrame,
    district_col: str = 'district',
    error_code_col: str = 'error_code',
    error_code_value: int = 300  # Biometric Mismatch
) -> pd.DataFrame:
    """
    Detect Laborer Biometric Attrition: High Error Code 300 (Biometric Mismatch)
    rates in agricultural/industrial clusters.
    """
    df = df.copy()
    
    if error_code_col not in df.columns:
        return pd.DataFrame()
    
    # Filter for error code 300
    error_300 = df[df[error_code_col] == error_code_value].copy()
    
    if error_300.empty:
        return pd.DataFrame()
    
    # Calculate error rate per district
    district_stats = df.groupby(district_col).agg({
        error_code_col: 'count'
    }).reset_index()
    district_stats.columns = [district_col, 'total_transactions']
    
    error_district = error_300.groupby(district_col).agg({
        error_code_col: 'count'
    }).reset_index()
    error_district.columns = [district_col, 'error_300_count']
    
    # Merge and calculate error rate
    district_stats = district_stats.merge(error_district, on=district_col, how='left')
    district_stats['error_300_count'] = district_stats['error_300_count'].fillna(0)
    district_stats['error_rate'] = district_stats['error_300_count'] / (district_stats['total_transactions'] + 1)
    
    # Calculate z-score
    mean_rate = district_stats['error_rate'].mean()
    std_rate = district_stats['error_rate'].std()
    district_stats['z_score'] = (district_stats['error_rate'] - mean_rate) / (std_rate + 1e-6)
    
    # Flag high error rates (z-score > 2)
    district_stats['is_attrition'] = district_stats['z_score'] > 2
    
    anomalies = district_stats[district_stats['is_attrition']].copy()
    
    return anomalies


def detect_gendered_digital_divide(
    df: pd.DataFrame,
    gender_col: str = 'gender',
    update_method_col: str = 'update_method',
    district_col: str = 'district'
) -> pd.DataFrame:
    """
    Detect Gendered Digital Divide: Online updates are 80% male while
    center-based updates are 50/50, indicating digital literacy gap.
    """
    df = df.copy()
    
    if gender_col not in df.columns or update_method_col not in df.columns:
        return pd.DataFrame()
    
    # Normalize gender
    df[gender_col] = df[gender_col].str.upper().str.strip()
    
    # Filter for online vs center-based
    online = df[df[update_method_col].str.contains('online', case=False, na=False)]
    center = df[df[update_method_col].str.contains('center', case=False, na=False)]
    
    if online.empty or center.empty:
        return pd.DataFrame()
    
    # Calculate gender distribution per district
    online_gender = online.groupby([district_col, gender_col]).size().unstack(fill_value=0)
    online_gender['online_total'] = online_gender.sum(axis=1)
    male_col_online = 'M' if 'M' in online_gender.columns else online_gender.columns[0] if len(online_gender.columns) > 0 else None
    online_gender['online_male_pct'] = (online_gender[male_col_online] if male_col_online else 0) / (online_gender['online_total'] + 1)
    
    center_gender = center.groupby([district_col, gender_col]).size().unstack(fill_value=0)
    center_gender['center_total'] = center_gender.sum(axis=1)
    male_col_center = 'M' if 'M' in center_gender.columns else center_gender.columns[0] if len(center_gender.columns) > 0 else None
    center_gender['center_male_pct'] = (center_gender[male_col_center] if male_col_center else 0) / (center_gender['center_total'] + 1)
    
    # Merge
    comparison = online_gender[['online_male_pct']].merge(
        center_gender[['center_male_pct']], 
        left_index=True, 
        right_index=True, 
        how='inner'
    )
    
    # Flag divide (online >80% male AND center ~50%)
    comparison['is_divide'] = (
        (comparison['online_male_pct'] > 0.8) &
        (comparison['center_male_pct'] >= 0.4) &
        (comparison['center_male_pct'] <= 0.6)
    )
    
    anomalies = comparison[comparison['is_divide']].reset_index()
    
    return anomalies


def detect_elderly_iris_update_surge(
    df: pd.DataFrame,
    age_col: str = 'age',
    biometric_type_col: str = 'biometric_type',
    date_col: str = 'date',
    activity_col: str = 'count'
) -> pd.DataFrame:
    """
    Detect Elderly Iris Update Surge: Localized increase in iris-based updates
    following government push for Life Certificates.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter for elderly (age >= 60)
    if age_col in df.columns:
        elderly = df[df[age_col] >= 60].copy()
    else:
        return pd.DataFrame()
    
    # Filter for iris updates
    if biometric_type_col in df.columns:
        iris_updates = elderly[elderly[biometric_type_col].str.contains('iris', case=False, na=False)]
    else:
        return pd.DataFrame()
    
    # Calculate time trend
    iris_updates['year_month'] = iris_updates[date_col].dt.to_period('M')
    monthly = iris_updates.groupby('year_month').agg({
        activity_col: 'sum'
    }).reset_index()
    
    # Detect surge (significant increase)
    monthly = monthly.sort_values('year_month')
    monthly['prev_month'] = monthly[activity_col].shift(1)
    monthly['growth'] = (monthly[activity_col] - monthly['prev_month']) / (monthly['prev_month'] + 1)
    
    # Flag surges (>50% growth)
    monthly['is_surge'] = monthly['growth'] > 0.5
    
    anomalies = monthly[monthly['is_surge']].copy()
    
    return anomalies


def detect_tribal_enrolment_gaps(
    df: pd.DataFrame,
    district_col: str = 'district',
    activity_col: str = 'count',
    population_col: str = 'population'
) -> pd.DataFrame:
    """
    Detect Tribal Enrolment Gaps: Districts showing saturation <40% when
    national average is >90%.
    """
    df = df.copy()
    
    # Calculate enrolment per district
    district_stats = df.groupby(district_col).agg({
        activity_col: 'sum'
    }).reset_index()
    district_stats.columns = [district_col, 'total_enrolments']
    
    # If population data available, calculate saturation
    if population_col in df.columns:
        district_pop = df.groupby(district_col)[population_col].first().reset_index()
        district_stats = district_stats.merge(district_pop, on=district_col, how='left')
        district_stats['saturation'] = district_stats['total_enrolments'] / (district_stats[population_col] + 1)
    else:
        # Use relative ranking as proxy
        district_stats['saturation'] = district_stats['total_enrolments'] / district_stats['total_enrolments'].max()
    
    # Calculate national average (or dataset average)
    national_avg = district_stats['saturation'].mean()
    
    # Flag gaps (<40% saturation when national >90% or when significantly below average)
    district_stats['is_gap'] = (
        (district_stats['saturation'] < 0.4) &
        (national_avg > 0.9)
    ) | (
        (district_stats['saturation'] < national_avg * 0.5)  # Less than half of average
    )
    
    anomalies = district_stats[district_stats['is_gap']].copy()
    
    return anomalies


def detect_relational_identity_erosion(
    df: pd.DataFrame,
    enrolment_type_col: str = 'enrolment_type',
    date_col: str = 'date',
    activity_col: str = 'count'
) -> pd.DataFrame:
    """
    Detect Relational Identity Erosion: Sudden drop in Head of Family (HoF)
    based enrolments in urban areas, signaling family fragmentation.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    
    if enrolment_type_col not in df.columns:
        return pd.DataFrame()
    
    # Filter for HoF enrolments
    hof = df[df[enrolment_type_col].str.contains('head|hof|family', case=False, na=False)]
    
    if hof.empty:
        return pd.DataFrame()
    
    # Calculate yearly trend
    yearly = hof.groupby('year').agg({
        activity_col: 'sum'
    }).reset_index()
    yearly = yearly.sort_values('year')
    yearly['prev_year'] = yearly[activity_col].shift(1)
    yearly['yoy_change'] = (yearly[activity_col] - yearly['prev_year']) / (yearly['prev_year'] + 1)
    
    # Flag erosion (significant decline)
    yearly['is_erosion'] = yearly['yoy_change'] < -0.2  # 20%+ decline
    
    anomalies = yearly[yearly['is_erosion']].copy()
    
    return anomalies


def detect_dbt_seeding_disparities(
    df: pd.DataFrame,
    district_col: str = 'district',
    aadhaar_seeded_col: str = 'aadhaar_seeded',
    bank_seeded_col: str = 'bank_seeded'
) -> pd.DataFrame:
    """
    Detect DBT Seeding Disparities: Aadhaar enrolment 100% but bank account
    seeding (ABPS) only 60%, indicating JAM trinity integration friction.
    """
    df = df.copy()
    
    if aadhaar_seeded_col not in df.columns or bank_seeded_col not in df.columns:
        return pd.DataFrame()
    
    # Calculate seeding rates per district
    district_stats = df.groupby(district_col).agg({
        aadhaar_seeded_col: 'mean',
        bank_seeded_col: 'mean'
    }).reset_index()
    district_stats.columns = [district_col, 'aadhaar_rate', 'bank_rate']
    
    # Calculate gap
    district_stats['seeding_gap'] = district_stats['aadhaar_rate'] - district_stats['bank_rate']
    
    # Flag disparities (Aadhaar high but bank low)
    district_stats['is_disparity'] = (
        (district_stats['aadhaar_rate'] > 0.95) &  # >95% Aadhaar
        (district_stats['bank_rate'] < 0.7)  # <70% bank seeding
    )
    
    anomalies = district_stats[district_stats['is_disparity']].copy()
    
    return anomalies


def get_category3_summary(enrolment_df: pd.DataFrame,
                          demographic_df: pd.DataFrame,
                          biometric_df: pd.DataFrame) -> Dict:
    """Get summary statistics for all Category III anomalies"""
    
    summary = {
        'migration_trail_spike': 0,
        'reverse_migration': 0,
        'baal_aadhaar_deactivation': 0,
        'laborer_biometric_attrition': 0,
        'gendered_digital_divide': 0,
        'elderly_iris_surge': 0,
        'tribal_enrolment_gaps': 0,
        'relational_identity_erosion': 0,
        'dbt_seeding_disparities': 0
    }
    
    try:
        migration = detect_migration_trail_address_spike(demographic_df)
        summary['migration_trail_spike'] = len(migration)
    except:
        pass
    
    try:
        reverse = detect_reverse_migration_patterns(demographic_df)
        summary['reverse_migration'] = len(reverse)
    except:
        pass
    
    try:
        baal = detect_baal_aadhaar_deactivation_wave(enrolment_df)
        summary['baal_aadhaar_deactivation'] = len(baal)
    except:
        pass
    
    try:
        laborer = detect_laborer_biometric_attrition(biometric_df)
        summary['laborer_biometric_attrition'] = len(laborer)
    except:
        pass
    
    try:
        gender = detect_gendered_digital_divide(demographic_df)
        summary['gendered_digital_divide'] = len(gender)
    except:
        pass
    
    try:
        iris = detect_elderly_iris_update_surge(biometric_df)
        summary['elderly_iris_surge'] = len(iris)
    except:
        pass
    
    try:
        tribal = detect_tribal_enrolment_gaps(enrolment_df)
        summary['tribal_enrolment_gaps'] = len(tribal)
    except:
        pass
    
    try:
        erosion = detect_relational_identity_erosion(enrolment_df)
        summary['relational_identity_erosion'] = len(erosion)
    except:
        pass
    
    try:
        dbt = detect_dbt_seeding_disparities(enrolment_df)
        summary['dbt_seeding_disparities'] = len(dbt)
    except:
        pass
    
    return summary
