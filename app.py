# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="AI Job Market Insights", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/ai_job_market_clean.csv")

df = load_data()
st.title("AI‑Powered Job Market Insights (Interactive)")
st.caption("Synthetic dataset with AI adoption & automation risk across roles/industries.")

# Sidebar filters
inds = sorted(df["industry"].dropna().unique())
adopts = ["Low", "Medium", "High"]
risks  = ["Low", "Medium", "High"]
sel_ind = st.sidebar.multiselect("Industry", inds, default=inds[:5])
sel_adopt = st.sidebar.multiselect("AI Adoption", adopts, default=adopts)
sel_risk  = st.sidebar.multiselect("Automation Risk", risks, default=risks)
min_sal, max_sal = st.sidebar.slider("Salary (USD)", 30000, 250000, (50000, 200000), step=5000)

mask = (
    df["industry"].isin(sel_ind) &
    df["ai_adoption_level"].isin(sel_adopt) &
    df["automation_risk"].isin(sel_risk) &
    df["salary_usd"].between(min_sal, max_sal)
)

dff = df[mask]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", len(dff))
c2.metric("Median Salary", f"${int(dff['salary_usd'].median()):,}" if len(dff) else "N/A")
c3.metric("Avg Adoption", f"{dff['adoption_score'].mean():.2f}" if len(dff) else "N/A")
c4.metric("Avg Risk", f"{dff['risk_score'].mean():.2f}" if len(dff) else "N/A")

# Industry salary vs adoption
ind_summary = (dff.groupby(["industry","ai_adoption_level"])
                 .agg(median_salary=("salary_usd","median"),
                      avg_adoption=("adoption_score","mean"),
                      count=("job_title","count"))
                 .reset_index())

fig1 = px.scatter(ind_summary, x="avg_adoption", y="median_salary",
                  color="industry", size="count",
                  labels={"avg_adoption":"Avg Adoption","median_salary":"Median Salary (USD)"},
                  title="Salary vs AI Adoption by Industry")
st.plotly_chart(fig1, use_container_width=True)

# Salary bands by adoption level
fig2 = px.histogram(dff, x="salary_band", color="ai_adoption_level", barmode="group",
                    title="Salary Bands by AI Adoption Level")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Sample (top 10)")
st.dataframe(dff.head(10))