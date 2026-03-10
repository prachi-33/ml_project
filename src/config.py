"""Configuration file for data paths and analysis parameters"""

import pathlib

# Project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data can be provided either as a single CSV file OR as a folder of chunked CSVs.
# Newer downloads (or API exports) often come as `data/raw/<dataset_name>/*.csv`.
ENROLMENT_FILE = RAW_DATA_DIR / "enrolment.csv"
DEMOGRAPHIC_UPDATE_FILE = RAW_DATA_DIR / "demographic_update.csv"
BIOMETRIC_UPDATE_FILE = RAW_DATA_DIR / "biometric_update.csv"

ENROLMENT_DIR = RAW_DATA_DIR / "enrolment"
DEMOGRAPHIC_UPDATE_DIR = RAW_DATA_DIR / "demographic_update"
BIOMETRIC_UPDATE_DIR = RAW_DATA_DIR / "biometric_update"

# Analysis parameters
ANOMALY_THRESHOLD_SIGMA = 5  # For statistical anomaly detection
ACTIVITY_DROP_THRESHOLD = 0.1  # 90% drop indicates activity desert
FORCE_CAPTURE_THRESHOLD = 0.4  # 40% force capture threshold
BIOMETRIC_EXCEPTION_THRESHOLD = 0.05  # 5% baseline for biometric exceptions
MULTIPLE_ENROLMENT_THRESHOLD = 5  # 5+ enrolments in 90 days
MOBILE_CHURN_THRESHOLD = 20  # 20+ Aadhaar numbers per mobile

# Date columns to parse
DATE_COLUMNS = ['date', 'enrollment_date', 'update_date', 'timestamp']

# Age group mappings
AGE_GROUPS = {
    '0_5': '0-5 years',
    '5_17': '5-17 years',
    '18_plus': '18+ years'
}
