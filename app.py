import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Palo Alto Networks | Attrition Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
)

PRIMARY = "#FA582D"
DARK = "#141414"

CAT_COLS = [
    "BusinessTravel", "Department", "Education", "EducationField",
    "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel",
    "JobRole", "JobSatisfaction", "MaritalStatus", "OverTime",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "WorkLifeBalance",
]
NUM_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",
    "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]
ENGINEERED_COLS = [
    "IncomeToExperienceRatio", "PromotionDelayFlag",
    "EngagementScore", "WorkloadStressFlag",
]


# --------------------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Income-to-experience ratio
    df["IncomeToExperienceRatio"] = df["MonthlyIncome"] / (df["TotalWorkingYears"].replace(0, np.nan) + 1)
    df["IncomeToExperienceRatio"] = df["IncomeToExperienceRatio"].fillna(df["MonthlyIncome"])

    # Promotion delay indicator (1 if longer than 4 years since last promotion)
    df["PromotionDelayFlag"] = (df["YearsSinceLastPromotion"] > 4).astype(int)

    # Engagement composite score (average of satisfaction / involvement metrics)
    df["EngagementScore"] = df[
        ["EnvironmentSatisfaction", "JobSatisfaction", "JobInvolvement", "RelationshipSatisfaction"]
    ].mean(axis=1)

    # Workload stress flag (overtime + low work-life balance)
    df["WorkloadStressFlag"] = (
        (df["OverTime"] == "Yes") & (df["WorkLifeBalance"] <= 2)
    ).astype(int)

    return df


def risk_category(prob: float) -> str:
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


RISK_COLOR = {"Low Risk": "#21C55D", "Medium Risk": "#FFA500", "High Risk": "#FF4B4B"}


# --------------------------------------------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, model_choice: str, test_size: float, use_class_weight: bool):
    df = engineer_features(df)

    y = df["Attrition"]
    if y.dtype == object:
        y = y.map({"Yes": 1, "No": 0}).fillna(y)
    y = y.astype(int)

    feature_cols = CAT_COLS + NUM_COLS + ENGINEERED_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]

    cat_features = [c for c in CAT_COLS if c in feature_cols]
    num_features = [c for c in feature_cols if c not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    if model_choice == "Logistic Regression (baseline)":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if use_class_weight else None,
        )
    elif model_choice == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            class_weight="balanced" if use_class_weight else None,
        )
    else:  # Gradient Boosting
        clf = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42
        )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, zero_division=0),
        "Recall": recall_score(y_test, y_pred_test, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_test, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_test),
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    cm = confusion_matrix(y_test, y_pred_test)

    # Score the FULL dataset for the dashboard (risk scoring framework)
    full_proba = pipe.predict_proba(X)[:, 1]

    # Feature importance
    try:
        ohe_names = pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_features)
        all_names = num_features + list(ohe_names)
        if hasattr(pipe.named_steps["clf"], "feature_importances_"):
            importances = pipe.named_steps["clf"].feature_importances_
        elif hasattr(pipe.named_steps["clf"], "coef_"):
            importances = np.abs(pipe.named_steps["clf"].coef_[0])
        else:
            importances = np.zeros(len(all_names))
        fi = pd.DataFrame({"feature": all_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        fi = pd.DataFrame({"feature": [], "importance": []})

    return {
        "pipeline": pipe,
        "metrics": metrics,
        "roc": (fpr, tpr),
        "cm": cm,
        "full_proba": full_proba,
        "feature_importance": fi,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
    }


def reason_codes(row: pd.Series) -> list:
    """Simple, interpretable individual-level reason codes."""
    reasons = []
    if row.get("OverTime") == "Yes":
        reasons.append("Works overtime")
    if row.get("JobSatisfaction", 4) <= 2:
        reasons.append("Low job satisfaction")
    if row.get("EnvironmentSatisfaction", 4) <= 2:
        reasons.append("Low environment satisfaction")
    if row.get("WorkLifeBalance", 4) <= 2:
        reasons.append("Poor work-life balance")
    if row.get("YearsSinceLastPromotion", 0) > 4:
        reasons.append(f"No promotion in {int(row['YearsSinceLastPromotion'])} years")
    if row.get("MonthlyIncome", 99999) < 3000:
        reasons.append("Below-average monthly income")
    if row.get("NumCompaniesWorked", 0) >= 4:
        reasons.append("History of frequent job changes")
    if row.get("DistanceFromHome", 0) >= 15:
        reasons.append("Long commute distance")
    if row.get("RelationshipSatisfaction", 4) <= 2:
        reasons.append("Low relationship satisfaction")
    if not reasons:
        reasons.append("No major risk drivers detected")
    return reasons[:5]


# --------------------------------------------------------------------------------------
# SIDEBAR — DATA + MODEL CONTROLS
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Palo Alto Networks")
st.sidebar.caption("Attrition Prediction & Risk Scoring System")

st.sidebar.divider()
st.sidebar.subheader("1. Data Source")

default_path = os.path.join(os.path.dirname(__file__), "data", "Palo Alto Networks.csv")

if os.path.exists(default_path):
    raw_df = load_data(default_path)
else:
    st.sidebar.error("Dataset not found. Please add 'Palo Alto Networks.csv' to the app data folder.")
    st.stop()
    st.stop()

st.sidebar.success(f"Loaded {len(raw_df):,} employee records")

st.sidebar.subheader("2. Model Configuration")
model_choice = st.sidebar.selectbox(
    "Model",
    ["Gradient Boosting", "Random Forest", "Logistic Regression (baseline)"],
    index=0,
)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
use_class_weight = st.sidebar.checkbox("Handle class imbalance (balanced weights)", value=True)

with st.spinner("Training model..."):
    result = train_models(raw_df, model_choice, test_size, use_class_weight)

df_fe = engineer_features(raw_df)
df_fe["AttritionProbability"] = result["full_proba"]
df_fe["RiskCategory"] = df_fe["AttritionProbability"].apply(risk_category)
if "EmployeeID" not in df_fe.columns:
    df_fe.insert(0, "EmployeeID", [f"EMP{1000+i}" for i in range(len(df_fe))])

st.sidebar.divider()
st.sidebar.subheader("3. Filters")
dept_filter = st.sidebar.multiselect(
    "Department", sorted(df_fe["Department"].unique()), default=list(df_fe["Department"].unique())
)
role_filter = st.sidebar.multiselect(
    "Job Role", sorted(df_fe["JobRole"].unique()), default=list(df_fe["JobRole"].unique())
)
risk_threshold = st.sidebar.slider("Minimum attrition probability shown", 0.0, 1.0, 0.0, 0.05)

filtered = df_fe[
    df_fe["Department"].isin(dept_filter)
    & df_fe["JobRole"].isin(role_filter)
    & (df_fe["AttritionProbability"] >= risk_threshold)
]

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.title("🛡️ Employee Attrition Risk Intelligence")
st.caption(
    "Predictive decision intelligence for proactive, employee-centric workforce management — "
    f"Model: **{model_choice}**"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Attrition Risk Dashboard",
        "👤 Employee Risk Profile",
        "🏢 Department-Level Risk View",
        "🔍 Explainability Panel",
        "🧪 Model Performance",
    ]
)

# --------------------------------------------------------------------------------------
# TAB 1 — ATTRITION RISK DASHBOARD
# --------------------------------------------------------------------------------------
with tab1:
    high_risk_n = (filtered["RiskCategory"] == "High Risk").sum()
    med_risk_n = (filtered["RiskCategory"] == "Medium Risk").sum()
    low_risk_n = (filtered["RiskCategory"] == "Low Risk").sum()
    avg_prob = filtered["AttritionProbability"].mean() if len(filtered) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees (filtered)", f"{len(filtered):,}")
    c2.metric("High Risk (>60%)", f"{high_risk_n:,}", delta=f"{high_risk_n/max(len(filtered),1):.1%}")
    c3.metric("Medium Risk (30–60%)", f"{med_risk_n:,}")
    c4.metric("Avg. Attrition Probability", f"{avg_prob:.1%}")

    st.markdown("### Overall Risk Distribution")
    colA, colB = st.columns([1, 1.4])

    with colA:
        pie_df = filtered["RiskCategory"].value_counts().reset_index()
        pie_df.columns = ["RiskCategory", "Count"]
        fig_pie = px.pie(
            pie_df, names="RiskCategory", values="Count", hole=0.5,
            color="RiskCategory", color_discrete_map=RISK_COLOR,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, width='stretch')

    with colB:
        fig_hist = px.histogram(
            filtered, x="AttritionProbability", nbins=30,
            color_discrete_sequence=[PRIMARY],
            labels={"AttritionProbability": "Attrition Probability"},
        )
        fig_hist.add_vline(x=0.3, line_dash="dash", line_color="orange")
        fig_hist.add_vline(x=0.6, line_dash="dash", line_color="red")
        fig_hist.update_layout(
            title="Distribution of Attrition Probability Scores",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hist, width='stretch')

    st.markdown("### High-Risk Employee Table")
    hr_table = (
        filtered[filtered["RiskCategory"] == "High Risk"]
        .sort_values("AttritionProbability", ascending=False)
        [["EmployeeID", "Department", "JobRole", "Age", "MonthlyIncome",
          "OverTime", "YearsAtCompany", "AttritionProbability"]]
    )
    hr_table = hr_table.rename(columns={"AttritionProbability": "Risk Score"})
    hr_table["Risk Score"] = hr_table["Risk Score"].map(lambda x: f"{x:.1%}")
    st.dataframe(hr_table, width='stretch', height=320)

# --------------------------------------------------------------------------------------
# TAB 2 — EMPLOYEE RISK PROFILE
# --------------------------------------------------------------------------------------
with tab2:
    st.markdown("### Individual Employee Lookup")
    emp_id = st.selectbox("Select Employee ID", filtered["EmployeeID"].tolist())
    emp_row = df_fe[df_fe["EmployeeID"] == emp_id].iloc[0]

    prob = emp_row["AttritionProbability"]
    cat = emp_row["RiskCategory"]
    color = RISK_COLOR[cat]

    c1, c2 = st.columns([1, 2])
    with c1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "Attrition Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#123d24"},
                        {"range": [30, 60], "color": "#4d3a12"},
                        {"range": [60, 100], "color": "#4d1414"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, width='stretch')
        st.markdown(f"**Risk Category:** <span style='color:{color}; font-weight:700'>{cat}</span>", unsafe_allow_html=True)

    with c2:
        st.markdown("#### Employee Snapshot")
        info_cols = ["Department", "JobRole", "Age", "Gender", "MaritalStatus",
                     "MonthlyIncome", "YearsAtCompany", "OverTime", "JobSatisfaction",
                     "WorkLifeBalance", "DistanceFromHome"]
        info_cols = [c for c in info_cols if c in emp_row.index]
        info_df = pd.DataFrame({"Attribute": info_cols, "Value": [str(emp_row[c]) for c in info_cols]})
        st.dataframe(info_df, hide_index=True, width='stretch')

    st.markdown("#### Key Contributing Factors (Reason Codes)")
    for r in reason_codes(emp_row):
        st.markdown(f"- ⚠️ {r}")

# --------------------------------------------------------------------------------------
# TAB 3 — DEPARTMENT-LEVEL RISK VIEW
# --------------------------------------------------------------------------------------
with tab3:
    st.markdown("### Aggregated Risk by Department & Role")

    dept_agg = (
        filtered.groupby("Department")["AttritionProbability"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "AvgRiskScore", "count": "Employees"})
        .sort_values("AvgRiskScore", ascending=False)
    )
    fig_dept = px.bar(
        dept_agg, x="Department", y="AvgRiskScore", color="AvgRiskScore",
        color_continuous_scale=["#21C55D", "#FFA500", "#FF4B4B"],
        text=dept_agg["AvgRiskScore"].map(lambda x: f"{x:.1%}"),
        labels={"AvgRiskScore": "Avg. Attrition Probability"},
    )
    fig_dept.update_layout(margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig_dept, width='stretch')

    role_agg = (
        filtered.groupby(["Department", "JobRole"])["AttritionProbability"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "AvgRiskScore", "count": "Employees"})
        .sort_values("AvgRiskScore", ascending=False)
    )
    fig_role = px.treemap(
        role_agg, path=["Department", "JobRole"], values="Employees",
        color="AvgRiskScore", color_continuous_scale=["#21C55D", "#FFA500", "#FF4B4B"],
    )
    fig_role.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_role, width='stretch')

    st.markdown("#### Department / Role Risk Table")
    role_agg_display = role_agg.copy()
    role_agg_display["AvgRiskScore"] = role_agg_display["AvgRiskScore"].map(lambda x: f"{x:.1%}")
    st.dataframe(role_agg_display, width='stretch', hide_index=True)

# --------------------------------------------------------------------------------------
# TAB 4 — EXPLAINABILITY PANEL
# --------------------------------------------------------------------------------------
with tab4:
    st.markdown("### Global Feature Importance")
    fi = result["feature_importance"].head(15)
    if len(fi):
        fig_fi = px.bar(
            fi.sort_values("importance"), x="importance", y="feature", orientation="h",
            color_discrete_sequence=[PRIMARY],
        )
        fig_fi.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=500)
        st.plotly_chart(fig_fi, width='stretch')
    else:
        st.info("Feature importance is unavailable for this model configuration.")

    st.divider()
    st.markdown("### What-If Scenario Exploration")
    st.caption("Adjust an employee's attributes and see how the predicted attrition probability changes.")

    base_id = st.selectbox("Base employee for scenario", filtered["EmployeeID"].tolist(), key="whatif_emp")
    base_row = df_fe[df_fe["EmployeeID"] == base_id].iloc[0].copy()

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        new_overtime = st.selectbox("OverTime", ["Yes", "No"], index=0 if base_row["OverTime"] == "Yes" else 1)
        new_income = st.number_input("Monthly Income", value=int(base_row["MonthlyIncome"]), step=100)
    with wc2:
        new_wlb = st.slider("Work-Life Balance", 1, 4, int(base_row["WorkLifeBalance"]))
        new_jobsat = st.slider("Job Satisfaction", 1, 4, int(base_row["JobSatisfaction"]))
    with wc3:
        new_promo = st.slider("Years Since Last Promotion", 0, 15, int(base_row["YearsSinceLastPromotion"]))
        new_envsat = st.slider("Environment Satisfaction", 1, 4, int(base_row["EnvironmentSatisfaction"]))

    scenario_row = base_row.copy()
    scenario_row["OverTime"] = new_overtime
    scenario_row["MonthlyIncome"] = new_income
    scenario_row["WorkLifeBalance"] = new_wlb
    scenario_row["JobSatisfaction"] = new_jobsat
    scenario_row["YearsSinceLastPromotion"] = new_promo
    scenario_row["EnvironmentSatisfaction"] = new_envsat

    scenario_df = pd.DataFrame([scenario_row])
    scenario_df = engineer_features(scenario_df)
    scenario_X = scenario_df[result["feature_cols"]]
    new_prob = result["pipeline"].predict_proba(scenario_X)[0, 1]

    delta = new_prob - base_row["AttritionProbability"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Original Probability", f"{base_row['AttritionProbability']:.1%}")
    s2.metric("Scenario Probability", f"{new_prob:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
    s3.metric("New Risk Category", risk_category(new_prob))

# --------------------------------------------------------------------------------------
# TAB 5 — MODEL PERFORMANCE
# --------------------------------------------------------------------------------------
with tab5:
    st.markdown("### Model Evaluation Metrics")
    m = result["metrics"]
    cols = st.columns(5)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")

    colA, colB = st.columns(2)
    with colA:
        fpr, tpr = result["roc"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve", line=dict(color=PRIMARY, width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(
            title=f"ROC Curve (AUC = {m['ROC-AUC']:.3f})",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_roc, width='stretch')

    with colB:
        cm = result["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Oranges",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Stay", "Leave"], y=["Stay", "Leave"],
        )
        fig_cm.update_layout(title="Confusion Matrix (test set)", margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_cm, width='stretch')

    st.caption(
        "Risk Scoring Framework: Low Risk < 30% · Medium Risk 30–60% · High Risk > 60%. "
        "Class imbalance handled via balanced class weights (SMOTE-equivalent effect); "
        "stratified train/test split preserves attrition ratio."
    )

st.divider()
st.caption(
    "Unified Mentor · Machine Learning–Based Employee Attrition Prediction and Risk Scoring System — "
    "Palo Alto Networks HR Analytics"
)