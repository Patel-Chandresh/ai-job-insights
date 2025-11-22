# clean.py
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

RAW = Path("data/ai_job_market_insights.csv")
OUT = Path("data/ai_job_market_clean.csv")

def parse_skills(s: str):
    if pd.isna(s): 
        return []
    s = s.replace("|", ",")
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    df = pd.read_csv(RAW)

    # --- Normalize column names
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_"))

    # --- String trims
    str_cols = ["job_title","industry","company_size","location",
                "ai_adoption_level","automation_risk","required_skills",
                "remote_friendly","job_growth_projection"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # --- Salary clean + outlier filter
    if "salary_usd" in df.columns:
        df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")
        df = df[df["salary_usd"].between(10000, 500000)]  # tune if needed

    # --- Normalize adoption level to Low/Medium/High
    map_adopt = {
        "low":"Low","medium":"Medium","high":"High",
        "emerging":"Low","intermediate":"Medium","advanced":"High"
    }
    df["ai_adoption_level"] = (df["ai_adoption_level"]
                                .str.lower()
                                .map(map_adopt)
                                .fillna(df["ai_adoption_level"]))

    # --- Normalize risk categories
    map_risk = {"low":"Low","moderate":"Medium","medium":"Medium","high":"High"}
    df["automation_risk"] = (df["automation_risk"]
                              .str.lower()
                              .map(map_risk)
                              .fillna(df["automation_risk"]))

    # --- Company size normalization
    map_size = {"small":"Small","medium":"Medium","large":"Large",
                "startup":"Small","smb":"Medium","enterprise":"Large"}
    df["company_size"] = (df["company_size"]
                           .str.lower()
                           .map(map_size)
                           .fillna(df["company_size"]))

    # --- Location formatting
    df["location"] = df["location"].str.title()

    # --- Remote friendly normalization
    map_remote = {"yes":"Yes","y":"Yes","true":"Yes","no":"No","n":"No","false":"No"}
    df["remote_friendly"] = (df["remote_friendly"]
                              .str.lower()
                              .map(map_remote)
                              .fillna(df["remote_friendly"]))

    # --- Skills list + one-hot for top-25 skills
    df["skills_list"] = df["required_skills"].apply(parse_skills)
    skill_counts = Counter([s for sl in df["skills_list"] for s in sl])
    top_skills = [k for k, _ in skill_counts.most_common(25)]
    for sk in top_skills:
        col = f"skill_{sk.lower().replace(' ','_').replace('+','plus')}"
        df[col] = df["skills_list"].apply(lambda L: int(sk in L))

    # --- Ordinal scores (for charts/stats)
    adoption_ord = {"Low":1, "Medium":2, "High":3}
    risk_ord     = {"Low":1, "Medium":2, "High":3}
    growth_ord   = {"Declining":1, "Stable":2, "Growing":3}
    if "ai_adoption_level" in df.columns:
        df["adoption_score"] = df["ai_adoption_level"].map(adoption_ord)
    if "automation_risk" in df.columns:
        df["risk_score"]     = df["automation_risk"].map(risk_ord)
    if "job_growth_projection" in df.columns:
        df["growth_score"]   = df["job_growth_projection"].map(growth_ord)

    # --- Salary bands
    if "salary_usd" in df.columns:
        bins   = [0, 50000, 100000, 150000, 200000, np.inf]
        labels = ["<50k","50-100k","100-150k","150-200k",">200k"]
        df["salary_band"] = pd.cut(df["salary_usd"], bins=bins, labels=labels, right=True)

    # --- Dedup + save
    df = df.drop_duplicates()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Clean file written to: {OUT.resolve()} (rows: {len(df)})")

if __name__ == "__main__":
    main()