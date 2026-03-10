## Project Setup

This project is designed to analyse the three Aadhaar datasets (Enrolment, Demographic Update, Biometric Update) and expose insights through notebooks and a Streamlit dashboard.

### 1. Python environment

- **Create and activate a virtual environment** (recommended):
  - On Windows (PowerShell):
    - `python -m venv .venv`
    - `.venv\Scripts\activate`
- **Install dependencies**:
  - `pip install -r requirements.txt`

### 2. Where to place CSV files from your local system

Create the following folders in the project root (same level as `requirements.txt`):

- `data/`
  - `raw/`
    - This is where you will copy **all original CSV files** downloaded from the Aadhaar portal.

Your folder structure should look like:

- `aadhar/`
  - `requirements.txt`
  - `setup.md`
  - `data/`
    - `raw/`
      - `aadhaar_enrolment.csv` (or similar name)
      - `aadhaar_demographic_update.csv`
      - `aadhaar_biometric_update.csv`

You can provide the datasets in **either** of these formats:

#### Option A (single CSV per dataset)
Rename them as:

- `data/raw/enrolment.csv`
- `data/raw/demographic_update.csv`
- `data/raw/biometric_update.csv`

#### Option B (chunked CSVs in folders) (your current format)
If your data is exported in multiple parts, place them like:

- `data/raw/enrolment/*.csv`
- `data/raw/demographic_update/*.csv`
- `data/raw/biometric_update/*.csv`

Example:

- `data/raw/enrolment/api_data_aadhar_enrolment_0_500000.csv`
- `data/raw/enrolment/api_data_aadhar_enrolment_500000_1000000.csv`

> **Important**: Only copy data files here from your local system; do **not** commit/upload actual Aadhaar data anywhere public.

### 3. Running the Streamlit app

After placing the CSV files under `data/raw/` and installing dependencies:

1. Activate your virtual environment (if not already active).
2. From the project root (`aadhar` folder), run **one** of these:

- Preferred (guarantees correct Python environment on Windows):
  - `.venv\\Scripts\\python -m streamlit run streamlit_app.py`
- If `streamlit` is on PATH (only works if you activated the same `.venv`):
  - `streamlit run streamlit_app.py`

The app will load data from either the single CSV files **or** the chunk folders described in section 2.

You can change these paths later in `src/config.py` if needed.

### 3A. (Optional) AI-generated conclusions using OpenAI

The dashboard includes an optional section to generate conclusions/recommendations from **aggregated stats**.

To enable it, set your API key as an environment variable (recommended):

- PowerShell:
  - `$env:OPENAI_API_KEY = "YOUR_KEY_HERE"`

Then run:

- `.venv\\Scripts\\python -m streamlit run streamlit_app.py`

Alternatively you can paste the key into the app (password field). The key is **not** saved to disk by the app.

You can also create a local `.env` file (recommended for development) by copying `env.example` to `.env` and filling in:

- `OPENAI_API_KEY`
- (optional) `OPENAI_MODEL`

### 4. Notebooks

If you use Jupyter or VS Code notebooks, you can create a `notebooks/` folder at the project root and read data using relative paths like:

- `../data/raw/enrolment.csv`
- `../data/raw/demographic_update.csv`
- `../data/raw/biometric_update.csv`

No additional configuration is needed as long as the CSV files are in `data/raw/`.

### 5. Next steps

- Place the three CSVs into `data/raw/` as described above.
- Then you can start building or running:
  - EDA scripts/notebooks.
  - Time-series and clustering models.
  - The Streamlit dashboard (`streamlit_app.py`).
