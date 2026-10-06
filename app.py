import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

MODEL_PATH = "credit_risk_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    """SHAP explainer for the XGBoost step (optional - app works without it)."""
    try:
        import shap
        clf = _model.named_steps["classifier"]
        return shap.TreeExplainer(clf)
    except Exception:
        return None


def gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Probability of Default"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 30], "color": "#86efac"},
                    {"range": [30, 60], "color": "#fde68a"},
                    {"range": [60, 100], "color": "#fca5a5"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def explain(model, explainer, row: pd.DataFrame) -> go.Figure | None:
    if explainer is None:
        return None
    pre = model.named_steps["preprocessor"]
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    X = pre.transform(row)
    if hasattr(X, "toarray"):
        X = X.toarray()
    sv = explainer.shap_values(pd.DataFrame(X, columns=names))[0]
    s = pd.Series(sv, index=names)
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(8)[::-1]
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=s.index,
            orientation="h",
            marker_color=["#ef4444" if v > 0 else "#22c55e" for v in s.values],
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Top factors (red = pushes toward default, green = toward safe)",
        xaxis_title="SHAP value (impact on model output)",
    )
    return fig


# ----------------------------------------------------------------- UI
st.title("💳 Credit Risk Prediction System")
st.caption("Enter applicant and loan details to estimate the probability of default.")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"`{MODEL_PATH}` not found. Run the 'save model' cell in your notebook "
        "and place the file in the same folder as app.py."
    )
    st.stop()

explainer = load_explainer(model)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.05,
        help="Applicants with default probability >= threshold are flagged as high risk.",
    )

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("👤 Applicant")
    person_age = st.number_input("Age", 18, 100, 30)
    person_income = st.number_input("Annual income", 1000, 10_000_000, 60_000, step=1000)
    person_home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    person_emp_length = st.number_input("Employment length (years)", 0.0, 60.0, 5.0, step=1.0)
    cb_person_default_on_file = st.selectbox(
        "Previous default on file?", ["N", "Y"],
        format_func=lambda v: "Yes" if v == "Y" else "No",
    )
    cb_person_cred_hist_length = st.number_input("Credit history length (years)", 0, 60, 5)

with col_b:
    st.subheader("🏦 Loan")
    loan_intent = st.selectbox(
        "Loan purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan amount", 500, 100_000, 10_000, step=500)
    loan_int_rate = st.number_input("Interest rate (%)", 0.0, 40.0, 11.0, step=0.1)
    loan_percent_income = loan_amnt / person_income
    st.metric("Loan as % of income (auto-calculated)", f"{loan_percent_income:.1%}")

if st.button("Predict risk", type="primary", use_container_width=True):
    row = pd.DataFrame(
        [{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": round(loan_percent_income, 2),
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }]
    )

    prob = float(model.predict_proba(row)[0, 1])

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.plotly_chart(gauge(prob), use_container_width=True)
        if prob >= threshold:
            st.error("⚠️ **HIGH RISK** - likely to default. Recommend rejecting or reviewing manually.")
        elif prob >= threshold * 0.6:
            st.warning("🟡 **MEDIUM RISK** - consider additional checks.")
        else:
            st.success("✅ **LOW RISK** - likely to repay. Recommend approval.")

    with right:
        fig = explain(model, explainer, row)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Install `shap` to see which factors drove this prediction.")

    with st.expander("Input sent to the model"):
        st.dataframe(row, use_container_width=True)
