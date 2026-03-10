# Aadhaar Analytics Dashboard - Anomaly Detection System

A comprehensive analytics platform for detecting anomalies, patterns, and trends in Aadhaar enrolment and update datasets. This project was developed for government hackathon analysis.

## Project Overview

This system analyzes three Aadhaar datasets:
1. **Aadhaar Enrolment Dataset** - Aggregated enrolment information
2. **Aadhaar Demographic Update Dataset** - Updates to demographic data
3. **Aadhaar Biometric Update Dataset** - Biometric update information

## Anomaly Detection Categories

### Category I: Administrative and Operational Bottlenecks
- Pincode-Level Activity Deserts
- Wait-Time Volatility Anomaly
- Center-Specific Rejection Clusters
- Force Capture Saturation
- Operator Certification Lag
- Bimodal Distribution of Appointment Success
- Haat Surge Anomaly

### Category II: Data Integrity and Technical Deviations
- Age/DOB Discrepancy
- Gender-Photo Dissonance
- Multiple Enrolment Burst (MEB)
- Pincode-Address Mismatch
- Mobile Number Churn Anomaly
- Error Code 999 Concentration
- Biometric Exception Overuse
- Transliteration Error Clusters

### Category III: Social and Behavioral Anomalies
- Migration Trail Address Spike
- Reverse Migration Patterns
- Baal Aadhaar Deactivation Wave
- Laborer Biometric Attrition
- Gendered Digital Divide
- Elderly Iris Update Surge
- Tribal Enrolment Gaps
- Relational Identity Erosion
- DBT Seeding Disparities

### Category IV: Technical Infrastructure and External Factors
- Power/Network Outage Signature
- Device Reputation Failure
- Latency-Induced Timeouts
- Census Anomaly (Saturation >100%)
- Deceased ID Persistence
- Voter ID Linking Spikes

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Data

Place your CSV files in the `data/raw/` folder:
- `data/raw/enrolment.csv`
- `data/raw/demographic_update.csv`
- `data/raw/biometric_update.csv`

See `setup.md` for detailed instructions.

### 3. Run the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser automatically.

## Project Structure

```
aadhar/
├── data/
│   └── raw/              # Place your CSV files here
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration settings
│   ├── data_loader.py    # Data loading utilities
│   ├── anomalies_category1.py  # Category I detection
│   ├── anomalies_category2.py  # Category II detection
│   ├── anomalies_category3.py  # Category III detection
│   ├── anomalies_category4.py  # Category IV detection
│   └── visualizations.py # Plotting functions
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Python dependencies
├── setup.md             # Setup instructions
├── streamlit_app.py     # Main dashboard application
└── README.md            # This file
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scipy** - Statistical analysis
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations
- **Streamlit** - Dashboard framework
- **Scikit-learn** - Machine learning utilities
- **Statsmodels** - Time-series analysis
- **PyOD** - Outlier detection

## Features

- **Comprehensive Anomaly Detection**: 30+ different anomaly types across 4 categories
- **Interactive Visualizations**: Plotly charts for all anomaly types
- **Real-time Analysis**: Fast detection algorithms with caching
- **Executive Summary**: High-level overview with key metrics
- **Detailed Reports**: Category-wise breakdowns with actionable insights
- **Conclusions & Recommendations**: Automated report generation

## Notes

- The system is designed to work with aggregated Aadhaar datasets
- Column names are automatically normalized (lowercase, underscores)
- Date columns are automatically detected and parsed
- The system gracefully handles missing data and column mismatches

## Data Privacy

**Important**: This system is designed for aggregated, anonymized data only. Never commit or upload actual Aadhaar data to public repositories.

## Support

For issues or questions, refer to the `setup.md` file or check the code documentation.

## 📄 License

This project is developed for government hackathon purposes.
