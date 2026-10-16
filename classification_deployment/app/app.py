"""Smart Appliance Identifier -- Streamlit front-end for the 23CSE301 classification capstone.

The user enters one electrical reading (aggregated power, voltage, current, real power). The app rebuilds the SAME
engineered features used in the notebook, scales them with the SAME fitted StandardScaler, and returns the trained
model's probability for each of the 12 appliances.

Everything the app needs is inside models/final_model.joblib (written by the last cells of classification.ipynb):
the fitted model, the fitted scaler, the feature order and the class names.

The deployed model uses ELECTRICAL features only. The notebook (PB10) shows that the recording hour is an artefact of
when the lab recorded each appliance, so the app deliberately does not ask for the time of day.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_model.joblib"

st.set_page_config(page_title="Smart Appliance Identifier", page_icon="⚡", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
model, scaler = bundle["model"], bundle["scaler"]
FEATURES, CLASSES = bundle["features"], bundle["class_names"]

# Real example readings taken from the dataset (full precision), one per appliance, so the app can be demoed
# without a meter. Order: aggregated power (W), voltage (V), current (A), real power (W)
PRESETS = {
    "6 Ton AC": (7464.275, 231.0, 35.63, 7023.909),
    "Coffee Machine": (2008.202, 232.0, 5.186, 1198.03),
    "Desktop Computer": (1658.378, 232.0, 0.865, 180.43),
    "Electric Kettle": (1678.908, 232.0, 6.507, 1494.698),
    "Hair Dryer": (2472.27, 231.0, 6.917, 1599.776),
    "Iron": (2196.17, 231.0, 6.558, 1499.718),
    "Microwave Oven": (1137.404, 231.0, 4.029, 885.219),
    "Refrigerator": (1149.259, 231.0, 1.285, 252.842),
    "TV": (1190.86, 232.0, 0.867, 190.63),
    "Toaster": (1972.512, 231.0, 3.465, 801.512),
    "Vacuum Cleaner": (1269.655, 231.0, 5.179, 1019.926),
    "Washing Machine": (1489.135, 232.0, 2.569, 507.632),
}
CUSTOM = "(enter my own reading)"


def engineer(agg, volt, cur, power) -> pd.DataFrame:
    """Rebuild the notebook's engineered features. Must stay identical to Section B3 of classification.ipynb."""
    apparent = volt * cur                                   # apparent_power = V x I
    power_factor = min(max(power / apparent, 0.0), 1.0)     # real / apparent, clipped to [0, 1]
    load_share = power / agg                                # appliance's share of the total load
    row = {
        "Aggregated Power": agg, "Voltage": volt, "Current": cur, "Power": power,
        "apparent_power": apparent, "power_factor": power_factor, "load_share": load_share,
    }
    return pd.DataFrame([row])[FEATURES]


def apply_preset():
    choice = st.session_state["preset"]
    if choice != CUSTOM:
        agg, volt, cur, power = PRESETS[choice]
        st.session_state.update(agg=agg, volt=volt, cur=cur, power=power)


for key, default in {"agg": 2000.0, "volt": 231.0, "cur": 5.000, "power": 1000.0}.items():
    st.session_state.setdefault(key, default)

st.title("⚡ Smart Appliance Identifier")
st.write("Enter one electrical reading and the model tells you **which household appliance** is drawing it "
         "(Non-Intrusive Load Monitoring).")

st.selectbox("Load a real example reading (optional)", [CUSTOM] + list(PRESETS), key="preset", on_change=apply_preset)

c1, c2 = st.columns(2)
with c1:
    st.number_input("Aggregated power (W)", min_value=0.0, max_value=20000.0, step=10.0, format="%.3f", key="agg",
                    help="Total power drawn by the whole monitored system.")
    st.number_input("Current (A)", min_value=0.0, max_value=100.0, step=0.001, format="%.3f", key="cur",
                    help="Enter all three decimals. TV and Desktop Computer differ only in the third decimal.")
with c2:
    st.number_input("Real power of the appliance (W)", min_value=0.0, max_value=20000.0, step=10.0, format="%.3f", key="power")
    st.number_input("Voltage (V)", min_value=0.0, max_value=300.0, step=1.0, format="%.1f", key="volt")

agg, volt, cur, power = (st.session_state[k] for k in ("agg", "volt", "cur", "power"))

if min(agg, volt, cur, power) <= 0:
    st.error("Aggregated power, voltage, current and real power must all be greater than zero.")
    st.stop()

if power > agg:
    st.warning("Real power is higher than the aggregated power. In the training data the aggregated load is never "
               "below the appliance's own power, so this reading is unusual.")
if power > volt * cur * 1.05:
    st.warning("Real power exceeds voltage × current, which is physically impossible (power factor > 1). "
               "Check the inputs; the power factor will be capped at 1.")

X_raw = engineer(agg, volt, cur, power)
X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=FEATURES)
proba = model.predict_proba(X_scaled)[0]
order = np.argsort(proba)[::-1]
top, runner_up = CLASSES[order[0]], CLASSES[order[1]]
conf = float(proba[order[0]])

st.divider()
st.subheader(f"Most likely: {top}")
st.metric("Model confidence", f"{conf:.1%}")

if {top, runner_up} == {"Iron", "Electric Kettle"} and proba[order[0]] + proba[order[1]] > 0.9:
    st.info("**Iron and Electric Kettle look identical in these measurements.** Both are ~1.5 kW resistive heating "
            "elements, so voltage, current and real power cannot tell them apart. Read this as "
            "*'one of the two'* — the model is effectively choosing between them.")
elif conf < 0.6:
    st.warning("Low confidence: this reading does not clearly match one appliance. It may be outside the 12 "
               "appliances the model was trained on, or the current may have been entered with too few decimals.")

top5 = pd.Series(proba[order[:5]], index=[CLASSES[i] for i in order[:5]], name="probability")
st.bar_chart(top5)

with st.expander("What the model actually sees (engineered features)"):
    st.dataframe(X_raw.round(4), hide_index=True)
    st.caption("apparent_power = V × I · power_factor = P ÷ (V × I) · load_share = P ÷ aggregated power")

with st.sidebar:
    st.header("About this model")
    st.write(f"**Model:** {bundle['model_name']}")
    m = bundle.get("test_metrics", {})
    if m:
        st.write(f"Test accuracy **{m.get('Accuracy', float('nan')):.3f}** · weighted F1 **{m.get('F1', float('nan')):.3f}** "
                 f"· ROC-AUC (OvR) **{m.get('ROC_AUC', float('nan')):.3f}**")
    st.caption(f"Trained with scikit-learn {bundle['sklearn_version']} · running {sklearn.__version__}")
    st.markdown("**Known limitations:** the model only knows 12 laboratory appliances and always answers with one of "
                "them. It cannot separate Iron from Electric Kettle using these measurements. Readings are single "
                "appliances measured one at a time, not a whole home with several appliances running.")
    st.markdown("**Data:** PMU-Based Residential Appliance Dataset (USPCASE, NUST), Mendeley Data, "
                "DOI 10.17632/fc2myy23yg.2, CC BY 4.0.")
    st.caption("Course project, 23CSE301 Machine Learning. Not a safety or billing tool.")
