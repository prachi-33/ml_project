"""Visualization functions for anomaly detection results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_anomaly_summary_by_category(summary_dict: Dict) -> go.Figure:
    """Create bar chart showing anomaly counts by category"""
    
    categories = ['Category I\n(Admin/Operational)', 
                  'Category II\n(Data Integrity)', 
                  'Category III\n(Social/Behavioral)', 
                  'Category IV\n(Infrastructure)']
    
    counts = [
        sum(summary_dict.get('category1', {}).values()),
        sum(summary_dict.get('category2', {}).values()),
        sum(summary_dict.get('category3', {}).values()),
        sum(summary_dict.get('category4', {}).values())
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
            text=counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Total Anomalies Detected by Category',
        xaxis_title='Category',
        yaxis_title='Number of Anomalies',
        template='plotly_white'
    )
    
    return fig


def plot_category1_anomalies(anomalies_dict: Dict) -> go.Figure:
    """Visualize Category I anomalies"""
    
    anomaly_names = [
        'Pincode Activity Deserts',
        'Wait-Time Volatility',
        'Rejection Clusters',
        'Force Capture Saturation',
        'Operator Certification Lag',
        'Appointment Success Bimodal',
        'Haat Surge Anomaly'
    ]
    
    counts = [
        anomalies_dict.get('pincode_activity_deserts', 0),
        anomalies_dict.get('wait_time_volatility', 0),
        anomalies_dict.get('rejection_clusters', 0),
        anomalies_dict.get('force_capture_saturation', 0),
        anomalies_dict.get('operator_certification_lag', 0),
        anomalies_dict.get('appointment_success_bimodal', 0),
        anomalies_dict.get('haat_surge_anomaly', 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=anomaly_names,
            y=counts,
            marker_color='#FF6B6B',
            text=counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Category I: Administrative and Operational Bottlenecks',
        xaxis_title='Anomaly Type',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_category2_anomalies(anomalies_dict: Dict) -> go.Figure:
    """Visualize Category II anomalies"""
    
    anomaly_names = [
        'Age/DOB Discrepancy',
        'Gender-Photo Dissonance',
        'Multiple Enrolment Burst',
        'Pincode-Address Mismatch',
        'Mobile Number Churn',
        'Error Code 999',
        'Biometric Exception Overuse',
        'Transliteration Errors'
    ]
    
    counts = [
        anomalies_dict.get('age_dob_discrepancy', 0),
        anomalies_dict.get('gender_photo_dissonance', 0),
        anomalies_dict.get('multiple_enrolment_burst', 0),
        anomalies_dict.get('pincode_address_mismatch', 0),
        anomalies_dict.get('mobile_number_churn', 0),
        anomalies_dict.get('error_code_999_concentration', 0),
        anomalies_dict.get('biometric_exception_overuse', 0),
        anomalies_dict.get('transliteration_error_clusters', 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=anomaly_names,
            y=counts,
            marker_color='#4ECDC4',
            text=counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Category II: Data Integrity and Technical Deviations',
        xaxis_title='Anomaly Type',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_category3_anomalies(anomalies_dict: Dict) -> go.Figure:
    """Visualize Category III anomalies"""
    
    anomaly_names = [
        'Migration Trail Spike',
        'Reverse Migration',
        'Baal Aadhaar Deactivation',
        'Laborer Biometric Attrition',
        'Gendered Digital Divide',
        'Elderly Iris Surge',
        'Tribal Enrolment Gaps',
        'Relational Identity Erosion',
        'DBT Seeding Disparities'
    ]
    
    counts = [
        anomalies_dict.get('migration_trail_spike', 0),
        anomalies_dict.get('reverse_migration', 0),
        anomalies_dict.get('baal_aadhaar_deactivation', 0),
        anomalies_dict.get('laborer_biometric_attrition', 0),
        anomalies_dict.get('gendered_digital_divide', 0),
        anomalies_dict.get('elderly_iris_surge', 0),
        anomalies_dict.get('tribal_enrolment_gaps', 0),
        anomalies_dict.get('relational_identity_erosion', 0),
        anomalies_dict.get('dbt_seeding_disparities', 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=anomaly_names,
            y=counts,
            marker_color='#45B7D1',
            text=counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Category III: Social and Behavioral Anomalies',
        xaxis_title='Anomaly Type',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_category4_anomalies(anomalies_dict: Dict) -> go.Figure:
    """Visualize Category IV anomalies"""
    
    anomaly_names = [
        'Power/Network Outage',
        'Device Reputation Failure',
        'Latency-Induced Timeouts',
        'Census Anomaly (>100%)',
        'Deceased ID Persistence',
        'Voter ID Linking Spikes'
    ]
    
    counts = [
        anomalies_dict.get('power_network_outage', 0),
        anomalies_dict.get('device_reputation_failure', 0),
        anomalies_dict.get('latency_induced_timeouts', 0),
        anomalies_dict.get('census_anomaly_saturation', 0),
        anomalies_dict.get('deceased_id_persistence', 0),
        anomalies_dict.get('voter_id_linking_spikes', 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=anomaly_names,
            y=counts,
            marker_color='#FFA07A',
            text=counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Category IV: Technical Infrastructure and External Factors',
        xaxis_title='Anomaly Type',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_time_series_with_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    anomalies_df: Optional[pd.DataFrame] = None,
    title: str = "Time Series with Anomalies"
) -> go.Figure:
    """Plot time series with highlighted anomalies"""
    
    fig = go.Figure()
    
    # Main time series
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[value_col],
        mode='lines',
        name='Normal Activity',
        line=dict(color='#3498db', width=2)
    ))
    
    # Highlight anomalies if provided
    if anomalies_df is not None and not anomalies_df.empty:
        if date_col in anomalies_df.columns and value_col in anomalies_df.columns:
            fig.add_trace(go.Scatter(
                x=anomalies_df[date_col],
                y=anomalies_df[value_col],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=value_col,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_geographic_heatmap(
    df: pd.DataFrame,
    location_col: str,
    value_col: str,
    title: str = "Geographic Distribution"
) -> go.Figure:
    """Create heatmap of values by location"""
    
    location_stats = df.groupby(location_col)[value_col].sum().reset_index()
    location_stats.columns = [location_col, 'total']
    
    fig = go.Figure(data=go.Bar(
        x=location_stats[location_col],
        y=location_stats['total'],
        marker_color=location_stats['total'],
        marker_colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=location_col,
        yaxis_title='Total',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Create correlation heatmap"""
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        width=800,
        height=800
    )
    
    return fig


def create_anomaly_dashboard_summary(all_summaries: Dict) -> go.Figure:
    """Create comprehensive dashboard summary"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Category I: Admin/Operational',
            'Category II: Data Integrity',
            'Category III: Social/Behavioral',
            'Category IV: Infrastructure'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Category I
    cat1_names = ['Activity Deserts', 'Wait-Time Vol', 'Rejections', 'Force Capture', 
                  'Cert Lag', 'Bimodal', 'Haat Surge']
    cat1_counts = [all_summaries.get('category1', {}).get(k, 0) 
                   for k in ['pincode_activity_deserts', 'wait_time_volatility', 
                            'rejection_clusters', 'force_capture_saturation',
                            'operator_certification_lag', 'appointment_success_bimodal',
                            'haat_surge_anomaly']]
    
    fig.add_trace(
        go.Bar(x=cat1_names, y=cat1_counts, name='Category I', marker_color='#FF6B6B'),
        row=1, col=1
    )
    
    # Category II
    cat2_names = ['Age/DOB', 'Gender', 'MEB', 'Pincode', 'Mobile', 'Error 999', 
                  'Bio Exception', 'Translit']
    cat2_counts = [all_summaries.get('category2', {}).get(k, 0)
                   for k in ['age_dob_discrepancy', 'gender_photo_dissonance',
                            'multiple_enrolment_burst', 'pincode_address_mismatch',
                            'mobile_number_churn', 'error_code_999_concentration',
                            'biometric_exception_overuse', 'transliteration_error_clusters']]
    
    fig.add_trace(
        go.Bar(x=cat2_names, y=cat2_counts, name='Category II', marker_color='#4ECDC4'),
        row=1, col=2
    )
    
    # Category III
    cat3_names = ['Migration', 'Reverse', 'Baal', 'Laborer', 'Gender Divide',
                  'Iris', 'Tribal', 'Identity', 'DBT']
    cat3_counts = [all_summaries.get('category3', {}).get(k, 0)
                   for k in ['migration_trail_spike', 'reverse_migration',
                            'baal_aadhaar_deactivation', 'laborer_biometric_attrition',
                            'gendered_digital_divide', 'elderly_iris_surge',
                            'tribal_enrolment_gaps', 'relational_identity_erosion',
                            'dbt_seeding_disparities']]
    
    fig.add_trace(
        go.Bar(x=cat3_names, y=cat3_counts, name='Category III', marker_color='#45B7D1'),
        row=2, col=1
    )
    
    # Category IV
    cat4_names = ['Outage', 'Device', 'Timeout', 'Census', 'Deceased', 'Voter']
    cat4_counts = [all_summaries.get('category4', {}).get(k, 0)
                   for k in ['power_network_outage', 'device_reputation_failure',
                            'latency_induced_timeouts', 'census_anomaly_saturation',
                            'deceased_id_persistence', 'voter_id_linking_spikes']]
    
    fig.add_trace(
        go.Bar(x=cat4_names, y=cat4_counts, name='Category IV', marker_color='#FFA07A'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Comprehensive Anomaly Detection Dashboard",
        showlegend=False,
        height=800,
        template='plotly_white'
    )
    
    return fig
