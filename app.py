"""Streamlit dashboard for the Bank Customer Churn risk-scoring project.

Modules:
  1. Customer Churn Risk Calculator
  2. Probability Distribution Visualization
  3. Feature Importance Dashboard
  4. What-If Scenario Simulator
"""

import json
import os
import sys

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_pipeline import engineer_features  # noqa: E402

# NOTE: DYLD_LIBRARY_PATH (for the project-local libomp.dylib XGBoost needs)
# must be set before this process starts — Streamlit runs app.py inside its
# own process via exec(), so re-execing here would kill the server instead
# of restarting it. Launch via ./run_dashboard.sh, which sets it up front.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

st.set_page_config(
    page_title="Bank Customer Churn Risk Intelligence",
    page_icon="🏦",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "best_model_name.json")) as f:
        best_name = json.load(f)["best_model"]
    return pipeline, best_name


@st.cache_data
def load_test_predictions():
    return pd.read_csv(os.path.join(REPORTS_DIR, "test_predictions.csv"))


@st.cache_data
def load_metrics():
    return pd.read_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"))


@st.cache_data
def load_feature_importance():
    fi = pd.read_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"))
    try:
        shap_fi = pd.read_csv(os.path.join(REPORTS_DIR, "shap_importance.csv"))
    except FileNotFoundError:
        shap_fi = None
    return fi, shap_fi


def predict_proba_for_input(pipeline, raw_row: dict) -> float:
    df = pd.DataFrame([raw_row])
    df = engineer_features(df)
    proba = pipeline.predict_proba(df)[:, 1][0]
    return proba, df


def risk_band(p: float) -> tuple:
    if p >= 0.6:
        return "High Risk", "#dc2626"
    elif p >= 0.3:
        return "Medium Risk", "#d97706"
    else:
        return "Low Risk", "#16a34a"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
pipeline, best_name = load_model()
metrics_df = load_metrics()
best_row = metrics_df[metrics_df["model"] == best_name].iloc[0]

st.title("🏦 Bank Customer Churn Risk Intelligence")
st.caption(
    "Predictive churn scoring for retail banking customers — European Central Bank engagement"
)

with st.sidebar:
    st.header("Model in Production")
    st.metric("Champion Model", best_name)
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{best_row['roc_auc']:.3f}")
    c2.metric("F1-Score", f"{best_row['f1']:.3f}")
    c3, c4 = st.columns(2)
    c3.metric("Precision", f"{best_row['precision']:.3f}")
    c4.metric("Recall", f"{best_row['recall']:.3f}")
    st.divider()
    st.markdown(
        "**Modules**\n"
        "1. Risk Calculator\n"
        "2. Probability Distribution\n"
        "3. Feature Importance\n"
        "4. What-If Simulator"
    )

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧮 Risk Calculator",
        "📊 Probability Distribution",
        "🔍 Feature Importance",
        "🎛️ What-If Simulator",
    ]
)


# ---------------------------------------------------------------------------
# Tab 1: Customer Churn Risk Calculator
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Customer Churn Risk Calculator")
    st.write("Enter a customer's profile to generate a churn probability and risk flag.")

    with st.form("risk_calc_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_score = st.slider("Credit Score", 300, 900, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            age = st.slider("Age", 18, 92, 40)
            tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
            balance = st.number_input("Account Balance (€)", 0.0, 300000.0, 50000.0, step=1000.0)
        with col3:
            num_products = st.slider("Number of Products", 1, 4, 2)
            has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"]) == "Yes"
            is_active = st.selectbox("Is Active Member", ["Yes", "No"]) == "Yes"
            estimated_salary = st.number_input(
                "Estimated Salary (€)", 0.0, 300000.0, 100000.0, step=1000.0
            )
        threshold = st.slider(
            "Churn flag threshold", 0.05, 0.95, 0.5, step=0.05,
            help="Probability above this value is flagged as predicted churn",
        )
        submitted = st.form_submit_button("Calculate Churn Risk", type="primary")

    if submitted:
        raw_row = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": int(has_cr_card),
            "IsActiveMember": int(is_active),
            "EstimatedSalary": estimated_salary,
        }
        proba, _ = predict_proba_for_input(pipeline, raw_row)
        label, color = risk_band(proba)
        flag = "CHURN PREDICTED" if proba >= threshold else "RETAINED"

        st.session_state["baseline_inputs"] = raw_row
        st.session_state["baseline_proba"] = proba

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{proba:.1%}")
        c2.markdown(
            f"<div style='padding:0.75rem;border-radius:0.5rem;background:{color}22;"
            f"border:1px solid {color};text-align:center;font-weight:600;color:{color}'>"
            f"{label}</div>",
            unsafe_allow_html=True,
        )
        c3.metric("Prediction @ Threshold", flag)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                                {"range": [0, 30], "color": "rgba(22, 163, 74, 0.2)"},
                                {"range": [30, 60], "color": "rgba(217, 119, 6, 0.2)"},
                                {"range": [60, 100], "color": "rgba(220, 38, 38, 0.2)"},
                            ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "value": threshold * 100,
                    },
                },
                title={"text": "Churn Risk Score"},
            )
        )
        fig.update_layout(height=320, margin=dict(l=30, r=30, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "This profile is now set as the baseline for the **What-If Simulator** tab.",
            icon="🎛️",
        )


# ---------------------------------------------------------------------------
# Tab 2: Probability Distribution Visualization
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Churn Probability Distribution — Held-Out Test Set")
    test_preds = load_test_predictions()

    fig = px.histogram(
        test_preds,
        x="y_proba",
        color=test_preds["y_true"].map({0: "Retained", 1: "Churned"}),
        nbins=40,
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={"Retained": "#16a34a", "Churned": "#dc2626"},
        labels={"y_proba": "Predicted Churn Probability", "color": "Actual Outcome"},
    )
    fig.update_layout(height=450, legend_title_text="Actual Outcome")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model Comparison**")
        st.dataframe(
            metrics_df.set_index("model").style.format("{:.3f}").highlight_max(
                subset=["roc_auc", "f1"], color="#16a34a33"
            ),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Confusion Matrix (Champion Model)**")
        cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_best.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)

    st.markdown("**ROC Curves**")
    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    if os.path.exists(roc_path):
        st.image(roc_path, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Feature Importance Dashboard
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Feature Importance & Churn Drivers")
    fi, shap_fi = load_feature_importance()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model Feature Importance — {best_name}**")
        fig = px.bar(
            fi.head(12).iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if shap_fi is not None:
            st.markdown("**SHAP Mean |Impact| on Churn Probability**")
            fig = px.bar(
                shap_fi.head(12).iloc[::-1],
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="Purples",
            )
            fig.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**SHAP Summary Plot**")
    shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=False, width=750)

    st.markdown("**Partial Dependence — Top Drivers**")
    pdp_path = os.path.join(FIGURES_DIR, "partial_dependence.png")
    if os.path.exists(pdp_path):
        st.image(pdp_path, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: What-If Scenario Simulator
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("What-If Scenario Simulator")
    st.write(
        "Start from the customer set in the **Risk Calculator** tab (or the default profile "
        "below), then adjust engagement and product variables to see how churn probability "
        "would change — useful for testing retention interventions."
    )

    baseline = st.session_state.get(
        "baseline_inputs",
        {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 40,
            "Tenure": 5,
            "Balance": 50000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 100000.0,
        },
    )
    baseline_proba = st.session_state.get("baseline_proba")
    if baseline_proba is None:
        baseline_proba, _ = predict_proba_for_input(pipeline, baseline)

    st.markdown("**Baseline profile**")
    st.json(baseline, expanded=False)

    st.markdown("**Adjust engagement / product levers**")
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_num_products = st.slider(
            "Number of Products", 1, 4, int(baseline["NumOfProducts"]), key="sim_products"
        )
    with col2:
        sim_active = st.selectbox(
            "Is Active Member",
            ["Yes", "No"],
            index=0 if baseline["IsActiveMember"] == 1 else 1,
            key="sim_active",
        ) == "Yes"
    with col3:
        sim_balance = st.slider(
            "Account Balance (€)",
            0.0,
            300000.0,
            float(baseline["Balance"]),
            step=1000.0,
            key="sim_balance",
        )

    scenario = dict(baseline)
    scenario["NumOfProducts"] = sim_num_products
    scenario["IsActiveMember"] = int(sim_active)
    scenario["Balance"] = sim_balance

    scenario_proba, _ = predict_proba_for_input(pipeline, scenario)
    delta = scenario_proba - baseline_proba

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Probability", f"{baseline_proba:.1%}")
    c2.metric("Scenario Probability", f"{scenario_proba:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
    label, color = risk_band(scenario_proba)
    c3.markdown(
        f"<div style='padding:0.75rem;border-radius:0.5rem;background:{color}22;"
        f"border:1px solid {color};text-align:center;font-weight:600;color:{color}'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Baseline", "Scenario"],
            y=[baseline_proba, scenario_proba],
            marker_color=["#64748b", color],
            text=[f"{baseline_proba:.1%}", f"{scenario_proba:.1%}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat=".0%", title="Churn Probability"),
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    if delta < -0.02:
        st.success(
            f"This intervention reduces churn risk by {abs(delta):.1%}. "
            "Consider prioritizing this lever for at-risk customers with similar profiles.",
            icon="✅",
        )
    elif delta > 0.02:
        st.warning(f"This scenario increases churn risk by {delta:.1%}.", icon="⚠️")
    else:
        st.info("This scenario has minimal impact on churn probability.", icon="ℹ️")
