from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_explainer import explain_flagged_case
from src.predict import FraudPredictor


st.set_page_config(
    page_title="CMS Fraud Detection",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    return FraudPredictor()


def risk_level(probability):
    if probability >= 0.80:
        return "Very High"
    if probability >= 0.60:
        return "High"
    if probability >= 0.40:
        return "Moderate"
    if probability >= 0.20:
        return "Low"
    return "Very Low"


def friendly_name(feature):
    names = {
        "ClaimCount": "claim volume",
        "UniqueBeneficiaries": "number of beneficiaries",
        "ReimbursementTotal": "total reimbursement",
        "ReimbursementPerClaim": "reimbursement per claim",
        "ClaimsPerBeneficiary": "claims per beneficiary",
        "InpatientClaimRatio": "inpatient claim ratio",
        "DiagnosisCountMean": "average diagnosis count",
        "ProcedureCountMean": "average procedure count",
        "ChronicConditionMean": "average chronic-condition count",
        "UniqueAttendingPhysicians": "attending-physician diversity",
        "DiagnosisDiversityRatio": "diagnosis diversity",
        "PhysicianDiversityRatio": "physician diversity",
    }
    return names.get(feature, feature.replace("_", " "))


def local_explanation(contributions):
    if contributions.empty:
        return ["No strong positive model contribution was identified."]
    return [
        f"Higher {friendly_name(feature)} increased the model's fraud-risk score."
        for feature in contributions.index
    ]


predictor = load_predictor()

st.title("CMS Provider Fraud Detection")
st.caption("Interactive dashboard · Logistic Regression · Optional LLM explanation")

with st.sidebar:
    st.header("Provider Profile")
    st.caption("Enter aggregated provider-level claim information.")

    claim_count = st.slider(
        "Claim count", 0, 2000, int(round(predictor.defaults["ClaimCount"]))
    )
    unique_beneficiaries = st.slider(
        "Unique beneficiaries",
        0,
        1000,
        int(round(predictor.defaults["UniqueBeneficiaries"])),
    )
    reimbursement_total = st.number_input(
        "Total reimbursement ($)",
        min_value=0.0,
        value=float(predictor.defaults["ReimbursementTotal"]),
        step=1000.0,
    )
    reimbursement_per_claim = st.number_input(
        "Reimbursement per claim ($)",
        min_value=0.0,
        value=float(predictor.defaults["ReimbursementPerClaim"]),
        step=50.0,
    )
    claims_per_beneficiary = st.slider(
        "Claims per beneficiary",
        0.0,
        10.0,
        float(predictor.defaults["ClaimsPerBeneficiary"]),
        0.1,
    )
    inpatient_ratio = st.slider(
        "Inpatient claim ratio",
        0.0,
        1.0,
        float(predictor.defaults["InpatientClaimRatio"]),
        0.01,
    )
    diagnosis_count = st.slider(
        "Average diagnosis count",
        0.0,
        10.0,
        float(predictor.defaults["DiagnosisCountMean"]),
        0.1,
    )
    procedure_count = st.slider(
        "Average procedure count",
        0.0,
        5.0,
        float(predictor.defaults["ProcedureCountMean"]),
        0.1,
    )
    chronic_count = st.slider(
        "Average chronic-condition count",
        0.0,
        12.0,
        float(predictor.defaults["ChronicConditionMean"]),
        0.1,
    )

    uploaded_file = st.file_uploader(
        "Or upload one provider row", type=["csv"]
    )

    predict_button = st.button(
        "Predict fraud risk", type="primary", width="stretch"
    )

st.success(
    "**Use Notice**  \n"
    "• Educational and analytics use only  \n"
    "• Not a final fraud determination, payment decision, or legal conclusion"
)

manual_values = {
    "ClaimCount": claim_count,
    "UniqueBeneficiaries": unique_beneficiaries,
    "ReimbursementTotal": reimbursement_total,
    "ReimbursementPerClaim": reimbursement_per_claim,
    "ClaimsPerBeneficiary": claims_per_beneficiary,
    "InpatientClaimRatio": inpatient_ratio,
    "DiagnosisCountMean": diagnosis_count,
    "ProcedureCountMean": procedure_count,
    "ChronicConditionMean": chronic_count,
}

input_df = (
    pd.read_csv(uploaded_file).head(1)
    if uploaded_file is not None
    else predictor.build_input(manual_values)
)

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Input Profile")
    profile_columns = [
        "ClaimCount",
        "UniqueBeneficiaries",
        "ReimbursementTotal",
        "ReimbursementPerClaim",
        "ClaimsPerBeneficiary",
        "InpatientClaimRatio",
        "DiagnosisCountMean",
        "ProcedureCountMean",
        "ChronicConditionMean",
    ]
    preview = predictor.prepare_input(input_df)[profile_columns]
    st.dataframe(preview, width="stretch", hide_index=True)

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric(
        "PR-AUC",
        f"{predictor.metrics.get('pr_auc', 0):.3f}",
    )
    metric2.metric(
        "ROC-AUC",
        f"{predictor.metrics.get('roc_auc', 0):.3f}",
    )
    metric3.metric("Model", "Logistic")

    st.subheader("Model Interpretation")
    st.info(
        "The model estimates provider-level fraud risk from claim volume, "
        "beneficiary counts, reimbursement patterns, inpatient utilization, "
        "diagnosis/procedure complexity, and related aggregated features."
    )

    st.subheader("Model Disclaimer")
    st.warning(
        "• This is an ML screening estimate, not proof of fraud.  \n"
        "• Results require claims, documentation, clinical, and compliance review.  \n"
        "• Do not use the prediction as the only basis for a real-world decision."
    )

    st.subheader("Production Enhancements")
    st.info(
        "Human review | Audit logs | Drift monitoring | Fairness checks | "
        "Access controls | PHI safeguards"
    )

with right:
    st.subheader("Prediction")

    if not predict_button:
        st.info("Adjust the profile and click **Predict fraud risk**.")
    else:
        result = predictor.predict(input_df)
        probability = result["probability"]
        contributions = predictor.explain_contributions(result["input"])

        st.metric("Fraud probability", f"{probability:.1%}")
        st.progress(float(probability))

        if result["is_fraud"]:
            st.error(f"Risk Level: {risk_level(probability)} · Potential Fraud")
        else:
            st.success(f"Risk Level: {risk_level(probability)} · Not Flagged")

        st.subheader("Fraud Risk Drivers")
        for explanation in local_explanation(contributions):
            st.markdown(f"- {explanation}")

        if result["is_fraud"]:
            st.subheader("LLM Explanation")
            with st.spinner("Generating grounded explanation..."):
                try:
                    llm_text = explain_flagged_case(
                        probability=probability,
                        contributions=contributions,
                        input_row=result["input"].iloc[0],
                    )
                except Exception as error:
                    llm_text = None
                    st.warning(f"LLM explanation unavailable: {error}")

            if llm_text:
                st.write(llm_text)
            else:
                st.info(
                    "The profile was flagged because the listed model factors "
                    "increased the Logistic Regression risk score. Add a valid "
                    "OPENAI_API_KEY to `.env` for a generated explanation."
                )
        else:
            st.subheader("Explanation")
            st.write(
                "The combined model signals remained below the configured "
                f"fraud threshold of {result['threshold']:.0%}."
            )

        with st.expander("View all model inputs"):
            st.dataframe(
                result["input"].T.rename(columns={0: "Value"}),
                width="stretch",
            )
