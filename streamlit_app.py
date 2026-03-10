"""Aadhaar aggregated datasets dashboard (graphs + supported anomalies)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_all_data, preprocess_data
from src.llm_insights import DEFAULT_MODEL, generate_conclusions
from src.uidai_anomaly_proxies import (
    adult_bio_concentration_by_district,
    census_saturation_anomaly,
    detect_adult_enrolment_surges,
    detect_digital_divide,
    detect_enrolment_deserts,
    detect_mbu_backlog,
    detect_migration_trail_signature,
    detect_pincode_activity_blackouts,
    monthly_sum,
    top_outliers,
    zscore_over_time,
)

# Load local environment variables from .env (optional)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, the app will still work with real OS env vars.
    pass


LUCIDE_ICONS: dict[str, str] = {
    "activity": """
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M22 12h-4l-3 9-6-18-3 9H2"/>
</svg>
""",
    "bar-chart-3": """
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M3 3v18h18"/>
  <path d="M18 17V9"/>
  <path d="M13 17V5"/>
  <path d="M8 17v-3"/>
</svg>
""",
    "users": """
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
  <circle cx="9" cy="7" r="4"/>
  <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
  <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
</svg>
""",
    "fingerprint": """
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M2 12c0-5.5 4.5-10 10-10s10 4.5 10 10"/>
  <path d="M5 12c0-3.9 3.1-7 7-7s7 3.1 7 7"/>
  <path d="M8 12c0-2.2 1.8-4 4-4s4 1.8 4 4"/>
  <path d="M12 12v2"/>
  <path d="M16 14v2a4 4 0 0 1-4 4"/>
  <path d="M19 12v2a7 7 0 0 1-7 7"/>
  <path d="M22 12v2c0 5.5-4.5 10-10 10"/>
</svg>
""",
    "alert-triangle": """
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
  <path d="M12 9v4"/>
  <path d="M12 17h.01"/>
</svg>
""",
    "trending-up": """
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M22 7 13.5 15.5l-5-5L2 17"/>
  <path d="M16 7h6v6"/>
</svg>
""",
}


def lucide(name: str, *, color: str = "#2563eb", size: int = 18) -> str:
    svg = LUCIDE_ICONS.get(name)
    if not svg:
        return ""
    return svg.format(color=color, size=size)


def h2_with_icon(icon_name: str, text: str, *, color: str) -> None:
    st.markdown(
        f"""
<div style="display:flex; align-items:center; gap:10px; margin: 6px 0 10px 0;">
  {lucide(icon_name, color=color, size=20)}
  <span style="font-size: 1.4rem; font-weight: 650; line-height: 1.2;">{text}</span>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        enrolment, demo_update, biometric_update = load_all_data()
    except Exception as e:  # noqa: BLE001
        # Streamlit will surface this nicely in the UI.
        raise RuntimeError(
            "Failed to load CSVs from `data/raw/`. Ensure these exist:\n"
            "- data/raw/enrolment.csv\n"
            "- data/raw/demographic_update.csv\n"
            "- data/raw/biometric_update.csv\n"
            f"\nOriginal error: {e}"
        ) from e

    enrolment = preprocess_data(enrolment, "enrolment")
    demo_update = preprocess_data(demo_update, "demographic_update")
    biometric_update = preprocess_data(biometric_update, "biometric_update")
    return enrolment, demo_update, biometric_update


def _national_ts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date", as_index=False)["total"].sum().sort_values("date")


def _state_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["state", "year", "month"], as_index=False)["total"].sum()
    out["year_month"] = pd.to_datetime(out["year"].astype(str) + "-" + out["month"].astype(str) + "-01")
    return out.sort_values(["state", "year_month"])


def _heatmap_state_month(df: pd.DataFrame, title: str):
    m = df.groupby(["state", "year", "month"], as_index=False)["total"].sum()
    pivot = m.pivot_table(index="state", columns="month", values="total", aggfunc="sum", fill_value=0)
    # Month numbers 1..12 on x-axis
    fig = px.imshow(
        pivot,
        aspect="auto",
        labels=dict(x="Month", y="State", color="Total"),
        title=title,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=700)
    return fig


def _top_locations(df: pd.DataFrame, level: str, n: int = 15) -> pd.DataFrame:
    cols = [level]
    out = df.groupby(cols, as_index=False)["total"].sum().sort_values("total", ascending=False).head(n)
    return out


@st.cache_data
def detect_activity_deserts_monthly(enrolment: pd.DataFrame, min_prev: float = 500.0, drop_ratio: float = 0.1):
    """
    Public UIDAI enrolment download is aggregated; it has no operator/center fields.
    Here we implement a *proxy* for 30-day “activity deserts” using month-to-month drops.
    """
    m = enrolment.groupby(["pincode", "state", "district", "year", "month"], as_index=False)["total"].sum()
    m = m.sort_values(["pincode", "year", "month"])
    m["prev_month_total"] = m.groupby("pincode")["total"].shift(1)
    m["drop_ratio"] = m["total"] / (m["prev_month_total"] + 1e-9)
    deserts = m[(m["prev_month_total"] >= min_prev) & (m["drop_ratio"] <= drop_ratio)].copy()
    deserts["year_month"] = pd.to_datetime(deserts["year"].astype(str) + "-" + deserts["month"].astype(str) + "-01")
    return deserts.sort_values(["year_month", "drop_ratio"])


@st.cache_data
def detect_haat_weekday_surge(enrolment: pd.DataFrame, ratio_threshold: float = 2.0, min_mean: float = 20.0):
    """
    Detect “Haat surge” proxy: a pincode where one weekday’s average enrolment is much higher.
    """
    d = enrolment.groupby(["pincode", "state", "district", "date"], as_index=False)["total"].sum()
    d["weekday"] = d["date"].dt.day_name()

    by = d.groupby(["pincode", "weekday"], as_index=False)["total"].mean()
    overall = d.groupby("pincode", as_index=False)["total"].mean().rename(columns={"total": "mean_daily"})
    peak = by.loc[by.groupby("pincode")["total"].idxmax()].rename(columns={"total": "peak_weekday_mean"})

    out = peak.merge(overall, on="pincode", how="left")
    out["surge_ratio"] = out["peak_weekday_mean"] / (out["mean_daily"] + 1e-9)

    # Attach geo (mode)
    geo = d.groupby("pincode", as_index=False).agg({"state": "first", "district": "first"})
    out = out.merge(geo, on="pincode", how="left")

    return out[(out["mean_daily"] >= min_mean) & (out["surge_ratio"] >= ratio_threshold)].sort_values(
        "surge_ratio", ascending=False
    )


@st.cache_data
def detect_quarterly_drop_clusters(enrolment: pd.DataFrame, min_prev: float = 2000.0, drop_ratio: float = 0.2):
    """
    District-level sudden quarterly drop proxy (can indicate certification/admin lag).
    """
    q = enrolment.groupby(["state", "district", "year", "quarter"], as_index=False)["total"].sum()
    q = q.sort_values(["state", "district", "year", "quarter"])
    q["prev_q_total"] = q.groupby(["state", "district"])["total"].shift(1)
    q["drop_ratio"] = q["total"] / (q["prev_q_total"] + 1e-9)
    return q[(q["prev_q_total"] >= min_prev) & (q["drop_ratio"] <= drop_ratio)].sort_values("drop_ratio")


@st.cache_data
def detect_migration_spike_proxy(demo_update: pd.DataFrame):
    """
    Migration-trail proxy: compare address update surges in March/November.
    The public demographic update file here is already aggregated (no update-type split),
    so this is based on overall demographic update volume.
    """
    m = demo_update.groupby(["state", "year", "month"], as_index=False)["total"].sum()
    mig = m[m["month"].isin([3, 11])].groupby("state", as_index=False)["total"].mean().rename(columns={"total": "mig_mean"})
    non = m[~m["month"].isin([3, 11])].groupby("state", as_index=False)["total"].mean().rename(
        columns={"total": "non_mig_mean"}
    )
    out = mig.merge(non, on="state", how="outer").fillna(0)
    out["ratio_mig_to_non"] = np.where(out["non_mig_mean"] > 0, out["mig_mean"] / out["non_mig_mean"], np.nan)
    return out.sort_values(["ratio_mig_to_non", "mig_mean"], ascending=False)


def main():
    st.set_page_config(page_title="Aadhaar Trends & Anomalies", layout="wide")

    h2_with_icon("activity", "Aadhaar Enrolment & Updates — Trends, Heatmaps, and Anomaly Proxies", color="#2563eb")
    st.caption(
        "These are **aggregated** public datasets (date/state/district/pincode + age buckets). "
        "Some anomalies (tokens/rejections/error-codes/device issues) require center-level operational logs and "
        "cannot be computed from this download."
    )

    enrolment, demo_update, biometric_update = load_data()

    with st.sidebar:
        st.subheader("Data loaded")
        st.write(
            {
                "enrolment_rows": len(enrolment),
                "demographic_update_rows": len(demo_update),
                "biometric_update_rows": len(biometric_update),
            }
        )
        st.markdown("---")
        if st.button("Refresh (clear cache)"):
            st.cache_data.clear()
            st.rerun()

    tab_overview, tab_enrol, tab_demo, tab_bio, tab_anom, tab_conc = st.tabs(
        ["Overview", "Enrolment", "Demographic updates", "Biometric updates", "Anomalies", "Conclusions"]
    )

    with tab_overview:
        h2_with_icon("bar-chart-3", "Overview", color="#1d4ed8")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total enrolments (sum of age buckets)", int(enrolment["total"].sum()))
        c2.metric("Total demographic updates", int(demo_update["total"].sum()))
        c3.metric("Total biometric updates", int(biometric_update["total"].sum()))

        st.subheader("National trend (daily)")
        fig = px.line(_national_ts(enrolment), x="date", y="total", title="Enrolment (national)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(_national_ts(demo_update), x="date", y="total", title="Demographic updates (national)")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(_national_ts(biometric_update), x="date", y="total", title="Biometric updates (national)")
        st.plotly_chart(fig3, use_container_width=True)

    with tab_enrol:
        h2_with_icon("users", "Enrolment", color="#16a34a")
        st.subheader("State × Month heatmap")
        st.plotly_chart(_heatmap_state_month(enrolment, "Enrolment heatmap by State and Month"), use_container_width=True)

        st.subheader("Top states by enrolment")
        top_states = _top_locations(enrolment, "state", 20)
        st.plotly_chart(px.bar(top_states, x="state", y="total", title="Top 20 states"), use_container_width=True)

        st.subheader("Age bucket composition (national)")
        age_cols = [c for c in enrolment.columns if c.startswith("age_")]
        if age_cols:
            comp = enrolment[age_cols].sum().reset_index()
            comp.columns = ["age_bucket", "total"]
            st.plotly_chart(px.pie(comp, names="age_bucket", values="total", title="Share by age bucket"), use_container_width=True)
        else:
            st.info("Age bucket columns not found.")

    with tab_demo:
        h2_with_icon("users", "Demographic updates", color="#7c3aed")
        st.subheader("State × Month heatmap")
        st.plotly_chart(
            _heatmap_state_month(demo_update, "Demographic update heatmap by State and Month"), use_container_width=True
        )

        st.subheader("Top states by demographic updates")
        top_states = _top_locations(demo_update, "state", 20)
        st.plotly_chart(px.bar(top_states, x="state", y="total", title="Top 20 states"), use_container_width=True)

    with tab_bio:
        h2_with_icon("fingerprint", "Biometric updates", color="#ea580c")
        st.subheader("State × Month heatmap")
        st.plotly_chart(
            _heatmap_state_month(biometric_update, "Biometric update heatmap by State and Month"), use_container_width=True
        )

        st.subheader("Top states by biometric updates")
        top_states = _top_locations(biometric_update, "state", 20)
        st.plotly_chart(px.bar(top_states, x="state", y="total", title="Top 20 states"), use_container_width=True)

    with tab_anom:
        h2_with_icon("alert-triangle", "Anomalies (proxies from aggregated data)", color="#dc2626")
        st.subheader("Supported anomaly proxies from these aggregated CSVs")
        st.markdown(
            "- **Pincode activity deserts (monthly proxy)**: pincode drops by ≥90% vs previous month.\n"
            "- **Haat weekday surges**: one weekday’s mean activity ≥2× the pincode’s mean daily activity.\n"
            "- **Quarterly district drop clusters**: district drops sharply quarter-over-quarter.\n"
            "- **Migration spike (proxy, state-level)**: March/November vs other months in demographic updates."
        )

        c1, c2 = st.columns(2)
        with c1:
            min_prev = st.number_input("Activity desert: min previous-month total", min_value=0, value=500, step=100)
        with c2:
            drop_ratio = st.number_input("Activity desert: drop ratio threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

        deserts = detect_activity_deserts_monthly(enrolment, min_prev=float(min_prev), drop_ratio=float(drop_ratio))
        st.markdown("#### Pincode activity deserts (top 50)")
        st.plotly_chart(
            px.bar(
                deserts.groupby("state", as_index=False)["pincode"].nunique().sort_values("pincode", ascending=False).head(20),
                x="state",
                y="pincode",
                title="Number of affected pincodes by state (top 20)",
                labels={"pincode": "pincodes flagged"},
            ),
            use_container_width=True,
        )
        st.dataframe(deserts.head(50), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Haat weekday surges (top 50)")
        haat = detect_haat_weekday_surge(enrolment)
        st.plotly_chart(
            px.bar(
                haat.groupby("weekday", as_index=False)["pincode"].nunique().sort_values("pincode", ascending=False),
                x="weekday",
                y="pincode",
                title="How many pincodes peak on each weekday",
                labels={"pincode": "pincodes flagged"},
            ),
            use_container_width=True,
        )
        st.dataframe(haat.head(50), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Quarterly district drop clusters (top 50)")
        qdrops = detect_quarterly_drop_clusters(enrolment)
        st.plotly_chart(
            px.bar(
                qdrops.groupby("state", as_index=False)["district"].nunique().sort_values("district", ascending=False).head(20),
                x="state",
                y="district",
                title="Number of affected districts by state (top 20)",
                labels={"district": "districts flagged"},
            ),
            use_container_width=True,
        )
        st.dataframe(qdrops.head(50), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Migration spike proxy (state-level, demographic updates)")
        mig = detect_migration_spike_proxy(demo_update)
        st.plotly_chart(
            px.bar(mig.head(20), x="state", y="ratio_mig_to_non", title="March/Nov vs other months ratio (top 20)"),
            use_container_width=True,
        )
        st.dataframe(mig.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Requested anomalies (computed from available fields)")
        st.caption(
            "These sections implement your hackathon anomalies using only columns present in the three aggregated CSVs. "
            "Where an external baseline is required (Census 2011 population, institutional births), we show how to provide it."
        )

        st.markdown("#### Enrolment dataset anomalies")

        with st.expander("Adult enrolment surges (age_18_greater) — z-score by state"):
            zthr = st.slider("Z-score threshold (state-level)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
            surges = detect_adult_enrolment_surges(enrolment, z_threshold=float(zthr))
            # Always show top outliers so results exist even if threshold yields few
            top = top_outliers(surges, "z", n=25, ascending=False)
            st.plotly_chart(
                px.bar(top, x="state", y="z", title="Top 25 state-month adult enrolment surges (z-score)"),
                use_container_width=True,
            )
            st.dataframe(top, use_container_width=True)

        with st.expander("Enrolment deserts (0–5) — district high, pincode near zero"):
            min_dist = st.number_input("Min district under-5 enrolment (latest month)", min_value=0, value=500, step=100)
            max_pin = st.number_input("Max pincode under-5 enrolment (latest month)", min_value=0, value=5, step=1)
            deserts2 = detect_enrolment_deserts(
                enrolment, latest_only=True, min_district_under5=float(min_dist), max_pincode_under5=float(max_pin)
            )
            top = deserts2[deserts2["is_desert"]].head(50)
            st.plotly_chart(
                px.bar(
                    top.groupby("state", as_index=False)["pincode"].nunique().sort_values("pincode", ascending=False).head(20),
                    x="state",
                    y="pincode",
                    title="Under-5 enrolment deserts: affected pincodes by state (top 20)",
                    labels={"pincode": "pincodes flagged"},
                ),
                use_container_width=True,
            )
            st.dataframe(deserts2.head(50), use_container_width=True)

        with st.expander("Census saturation anomaly (>100%) — requires district population baseline"):
            st.write(
                "To compute this exactly, add a file like: `data/external/census_2011_district_population.csv` "
                "with columns: `state,district,population_2011`."
            )
            pop_path = st.text_input("Population CSV path", value="data/external/census_2011_district_population.csv")
            try:
                sat = census_saturation_anomaly(enrolment, population_2011_csv=Path(pop_path), threshold=1.0)
                st.plotly_chart(
                    px.bar(sat.head(20), x="district", y="saturation", color="state", title="Top 20 saturation values"),
                    use_container_width=True,
                )
                st.dataframe(sat.head(50), use_container_width=True)
            except Exception as e:
                st.info(f"Population baseline not loaded yet: {e}")

        st.markdown("---")
        st.markdown("#### Demographic vs Biometric update anomalies")

        with st.expander("MBU backlog proxy — demo_age_5_17 high but bio_age_5_17 low"):
            min_demo = st.number_input("Min demo_age_5_17 (latest month)", min_value=0, value=100, step=50)
            backlog = detect_mbu_backlog(demo_update, biometric_update, latest_only=True, min_demo=float(min_demo))
            top = backlog[backlog["is_backlog"]].head(50)
            st.plotly_chart(
                px.bar(
                    top.groupby("state", as_index=False)["pincode"].nunique().sort_values("pincode", ascending=False).head(20),
                    x="state",
                    y="pincode",
                    title="MBU backlog proxy: affected pincodes by state (top 20)",
                    labels={"pincode": "pincodes flagged"},
                ),
                use_container_width=True,
            )
            st.dataframe(backlog.head(50), use_container_width=True)

        with st.expander("Digital divide disparity — demo_age_17_ high while bio_age_17_ low"):
            min_demo_adult = st.number_input("Min demo_age_17_ (latest month)", min_value=0, value=100, step=50)
            gap = detect_digital_divide(demo_update, biometric_update, latest_only=True, min_demo_adult=float(min_demo_adult))
            top = gap[gap["is_gap"]].head(50)
            st.plotly_chart(
                px.bar(top.head(25), x="pincode", y="demo_to_bio_ratio", color="state", title="Top 25 pincodes by demo/bio ratio"),
                use_container_width=True,
            )
            st.dataframe(gap.head(50), use_container_width=True)

        with st.expander("Migration trail signature — seasonal spikes in demo_age_17_ (urban proxy + z-score)"):
            zthr2 = st.slider("Z-score threshold (pincode)", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
            spikes = detect_migration_trail_signature(demo_update, z_threshold=float(zthr2), top_urban_quantile=0.9)
            top = spikes[spikes["is_spike"]].head(50)
            if top.empty:
                top = top_outliers(spikes[spikes["is_urban_proxy"]], "z", n=50, ascending=False)
            st.plotly_chart(
                px.bar(top.head(25), x="pincode", y="z", color="state", title="Top 25 adult demographic spikes (z-score)"),
                use_container_width=True,
            )
            st.dataframe(top.head(50), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Cross-dataset anomalies")

        with st.expander("Pincode activity blackouts — synchronized drops across enrolment + demo + bio"):
            zthr3 = st.slider("Blackout threshold (z-score, more negative is stricter)", min_value=-6.0, max_value=-1.0, value=-3.0, step=0.5)
            blackouts = detect_pincode_activity_blackouts(enrolment, demo_update, biometric_update, z_threshold=float(zthr3))
            top = blackouts[blackouts["is_blackout"]].head(50)
            if top.empty:
                top = blackouts.head(50)
            st.plotly_chart(
                px.bar(top.head(25), x="pincode", y="blackout_score", color="state", title="Top 25 potential blackouts (min z across datasets)"),
                use_container_width=True,
            )
            st.dataframe(top, use_container_width=True)

        with st.expander("Adult biometric concentration (district) — proxy for rural exclusion risk"):
            topk = st.number_input("Top-k pincodes per district", min_value=1, value=5, step=1)
            conc = adult_bio_concentration_by_district(biometric_update, latest_only=True, top_k=int(topk))
            st.plotly_chart(
                px.bar(conc.head(20), x="district", y="top_share", color="state", title=f"Top 20 districts by top-{topk} pincode share"),
                use_container_width=True,
            )
            st.dataframe(conc.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Z-score scoring (burst vs administrative failure)")
        st.caption("For each pincode, compute z-score of monthly totals; \(z>3\) is a burst, \(z<-3\) is a failure signal.")

        z_choice = st.selectbox("Dataset for z-score", ["Enrolment total", "Demographic total", "Biometric total"])
        z_thr = st.slider("Threshold", min_value=1.0, max_value=6.0, value=3.0, step=0.5)

        if z_choice == "Enrolment total":
            zdf = monthly_sum(enrolment, ["total"], ["state", "district", "pincode"]).rename(columns={"total": "value"})
        elif z_choice == "Demographic total":
            zdf = monthly_sum(demo_update, ["total"], ["state", "district", "pincode"]).rename(columns={"total": "value"})
        else:
            zdf = monthly_sum(biometric_update, ["total"], ["state", "district", "pincode"]).rename(columns={"total": "value"})

        zdf = zscore_over_time(zdf, ["pincode"], "value")
        latest = zdf["year_month"].max()
        latest_rows = zdf[zdf["year_month"] == latest].copy()
        bursts = latest_rows[latest_rows["z"] >= float(z_thr)].sort_values("z", ascending=False).head(50)
        fails = latest_rows[latest_rows["z"] <= -float(z_thr)].sort_values("z", ascending=True).head(50)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top bursts (latest month)**")
            st.dataframe(bursts, use_container_width=True)
        with c2:
            st.markdown("**Top failures (latest month)**")
            st.dataframe(fails, use_container_width=True)

        st.subheader("Additional anomaly types that need operational logs")
        st.write(
            "Your public aggregated CSVs do **not** include center/device/error-code level fields. "
            "To compute the following anomaly types, you would need additional operational datasets."
        )

        requirements = pd.DataFrame(
            [
                {
                    "Anomaly": "Wait-time volatility",
                    "Why not computable from CSVs": "No token/packet workflow fields in aggregated public data.",
                    "Fields needed (examples)": "center_id, tokens_issued, packets_uploaded, timestamp",
                    "Closest proxy we show": "Activity deserts (monthly) + weekday surges + quarterly drops",
                },
                {
                    "Anomaly": "Center-specific rejection clusters",
                    "Why not computable from CSVs": "No rejection counts or rejection reasons.",
                    "Fields needed (examples)": "center_id, rejection_reason, rejected_count, total_packets",
                    "Closest proxy we show": "None (requires rejection metadata)",
                },
                {
                    "Anomaly": "Force-capture saturation",
                    "Why not computable from CSVs": "No fingerprint capture mode flags.",
                    "Fields needed (examples)": "pincode/center_id, force_capture_count, total_capture_count, age_bucket",
                    "Closest proxy we show": "None (requires capture-mode telemetry)",
                },
                {
                    "Anomaly": "Device error-code clusters (521/561/999/300)",
                    "Why not computable from CSVs": "No error_code field in public aggregated downloads.",
                    "Fields needed (examples)": "error_code, device_id, district, timestamp, request_latency_ms",
                    "Closest proxy we show": "None (requires transactional logs)",
                },
                {
                    "Anomaly": "Biometric exception overuse",
                    "Why not computable from CSVs": "No biometric-exception flag/counts (missing fingers/eyes).",
                    "Fields needed (examples)": "biometric_exception_flag/count, modality, pincode, age_bucket",
                    "Closest proxy we show": "Biometric update volume patterns (not exceptions)",
                },
                {
                    "Anomaly": "Deceased ID persistence",
                    "Why not computable from CSVs": "Requires linkage to Civil Registration System (CRS) / deactivation dates.",
                    "Fields needed (examples)": "aadhaar_id (or hashed), deactivation_date, auth_timestamp, auth_count",
                    "Closest proxy we show": "None (requires cross-system linkage)",
                },
            ]
        )

        st.dataframe(requirements, use_container_width=True, hide_index=True)

        with st.expander("If you can get operational logs, what to ask for"):
            st.markdown(
                "- Center-level daily operational summary (tokens, packets, uploads, failures)\n"
                "- Rejection reason counts by center\n"
                "- Device batch/model and device error codes\n"
                "- Biometric capture telemetry (force-capture, exception flags)\n"
                "- CRS/UIDAI deactivation linkage (de-identified)\n"
            )

    with tab_conc:
        h2_with_icon("trending-up", "Conclusions", color="#0f766e")
        st.subheader("Data-driven conclusions (based on your files)")

        deserts = detect_activity_deserts_monthly(enrolment)
        haat = detect_haat_weekday_surge(enrolment)
        qdrops = detect_quarterly_drop_clusters(enrolment)
        mig = detect_migration_spike_proxy(demo_update)

        top_mig_ratio = None
        if mig is not None and not mig.empty and "ratio_mig_to_non" in mig.columns:
            top_mig_ratio = mig.iloc[0]["ratio_mig_to_non"]

        st.markdown(
            f"- **Activity deserts**: flagged **{deserts['pincode'].nunique()} pincodes** across **{deserts['state'].nunique()} states**.\n"
            f"- **Haat weekday surges**: flagged **{haat['pincode'].nunique()} pincodes** (check staffing on peak weekdays).\n"
            f"- **Quarterly district drops**: flagged **{qdrops['district'].nunique()} districts** across **{qdrops['state'].nunique()} states**.\n"
            f"- **Migration spike proxy**: top state ratio is **{top_mig_ratio if top_mig_ratio is not None else 'N/A'}** (computed only when non-migration months exist)."
        )

        st.markdown("#### Suggested actions")
        st.markdown(
            "- **Operational**: investigate flagged pincodes/districts for connectivity/operator availability; plan backup connectivity.\n"
            "- **Staffing**: align staffing/outreach with weekday peaks (weekly market days).\n"
            "- **Planning**: use month heatmaps to schedule camps and monitor unexpected drops early."
        )

        st.markdown("---")
        st.subheader("AI-generated conclusions (optional)")
        st.caption(
            "This will send **aggregated statistics only** (counts, top outliers, date ranges) to the model. "
            "Do not use this feature if you are not allowed to share even aggregated outputs."
        )

        if "ai_report_md" not in st.session_state:
            st.session_state["ai_report_md"] = ""

        # Key via env / secrets / password input (never persisted)
        env_key = os.environ.get("OPENAI_API_KEY", "")
        secrets_key = ""
        try:
            secrets_key = st.secrets.get("OPENAI_API_KEY", "")  # type: ignore[attr-defined]
        except Exception:
            secrets_key = ""

        api_key = secrets_key or env_key
        api_key_input = st.text_input("OpenAI API key (optional)", value="", type="password")
        if api_key_input.strip():
            api_key = api_key_input.strip()

        model = st.text_input("Model", value=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
        include_examples = st.checkbox("Include top-10 examples per anomaly in the payload", value=True)

        c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 2])
        with c_btn1:
            generate_clicked = st.button("Generate AI conclusions", use_container_width=True)
        with c_btn2:
            clear_clicked = st.button("Clear AI conclusions", use_container_width=True)
        with c_btn3:
            st.download_button(
                "Download AI conclusions (markdown)",
                data=st.session_state.get("ai_report_md", "") or "",
                file_name="ai_conclusions.md",
                mime="text/markdown",
                disabled=not bool(st.session_state.get("ai_report_md", "").strip()),
                use_container_width=True,
            )

        if clear_clicked:
            st.session_state["ai_report_md"] = ""

        if generate_clicked:
            if not api_key:
                st.error("No API key found. Set `OPENAI_API_KEY` or paste a key above.")
            else:
                # Build a compact payload from current computed stats
                payload: dict = {
                    "dataset_info": {
                        "enrolment_rows": int(len(enrolment)),
                        "demographic_rows": int(len(demo_update)),
                        "biometric_rows": int(len(biometric_update)),
                        "date_range": {
                            "enrolment": [str(enrolment["date"].min()), str(enrolment["date"].max())],
                            "demographic": [str(demo_update["date"].min()), str(demo_update["date"].max())],
                            "biometric": [str(biometric_update["date"].min()), str(biometric_update["date"].max())],
                        },
                    },
                    "totals": {
                        "enrolment_total": int(enrolment["total"].sum()),
                        "demographic_total": int(demo_update["total"].sum()),
                        "biometric_total": int(biometric_update["total"].sum()),
                    },
                    "signals": {},
                }

                # Existing proxies used earlier in conclusions tab
                payload["signals"]["activity_deserts"] = {
                    "pincodes_flagged": int(deserts["pincode"].nunique()),
                    "states_affected": int(deserts["state"].nunique()),
                }
                payload["signals"]["haat_weekday_surges"] = {
                    "pincodes_flagged": int(haat["pincode"].nunique()),
                }
                payload["signals"]["quarterly_drops"] = {
                    "districts_flagged": int(qdrops["district"].nunique()),
                    "states_affected": int(qdrops["state"].nunique()),
                }

                # Additional requested anomalies (top outliers)
                surges = detect_adult_enrolment_surges(enrolment, z_threshold=3.0)
                mbu = detect_mbu_backlog(demo_update, biometric_update, latest_only=True, min_demo=100.0)
                divide = detect_digital_divide(demo_update, biometric_update, latest_only=True, min_demo_adult=100.0)
                blackouts = detect_pincode_activity_blackouts(enrolment, demo_update, biometric_update, z_threshold=-3.0)
                conc = adult_bio_concentration_by_district(biometric_update, latest_only=True, top_k=5)

                payload["signals"]["adult_enrolment_surge"] = {
                    "count_z_ge_3": int((surges["z"] >= 3.0).sum()),
                }
                payload["signals"]["mbu_backlog_proxy"] = {
                    "pincodes_meeting_min_demo": int(mbu["pincode"].nunique()),
                }
                payload["signals"]["digital_divide_proxy"] = {
                    "pincodes_meeting_min_demo_adult": int(divide["pincode"].nunique()),
                }
                payload["signals"]["blackout_proxy"] = {
                    "rows_flagged": int(blackouts["is_blackout"].sum()),
                    "pincodes_flagged": int(blackouts.loc[blackouts["is_blackout"], "pincode"].nunique()),
                }
                payload["signals"]["adult_bio_concentration"] = {
                    "top20_avg_top_share": float(conc.head(20)["top_share"].mean()) if not conc.empty else None,
                }

                if include_examples:
                    payload["examples"] = {
                        "adult_enrolment_surges_top10": top_outliers(surges, "z", n=10, ascending=False).to_dict("records"),
                        "mbu_backlog_top10": mbu.head(10).to_dict("records"),
                        "digital_divide_top10": divide.head(10).to_dict("records"),
                        "blackouts_top10": blackouts[blackouts["is_blackout"]].head(10).to_dict("records"),
                        "adult_bio_concentration_top10": conc.head(10).to_dict("records"),
                    }

                with st.spinner("Generating AI conclusions..."):
                    try:
                        text = generate_conclusions(api_key=api_key, payload=payload, model=model)
                        if not text.strip():
                            st.warning("Model returned empty text.")
                        else:
                            st.session_state["ai_report_md"] = text
                    except Exception as e:
                        st.error(f"AI generation failed: {e}")

        if st.session_state.get("ai_report_md", "").strip():
            st.markdown("#### AI conclusions and suggestions")
            st.markdown(st.session_state["ai_report_md"])


if __name__ == "__main__":
    main()
