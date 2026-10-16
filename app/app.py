from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

st.set_page_config(page_title="ML Studio", page_icon="✦", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #f7f8fc 0%, #eef2ff 100%);}
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 20px; background: #171b35; color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2.1rem; margin: 0 0 .35rem 0;}
.hero p {color: #d9def8; margin: 0;}
div[data-testid="stMetric"] {background: white; padding: 1rem; border: 1px solid #e7e9f2; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>✦ ML Studio</h1><p>One workspace. Three ways to discover what your data says.</p></div>', unsafe_allow_html=True)
mode = st.radio("Choose a task", ["Classification", "Regression", "Clustering"], horizontal=True, label_visibility="collapsed")
st.write("")

def load_model(filename):
    path = MODELS / filename
    if not path.exists():
        st.error(f"Model file not found: `models/{filename}`. Run the model export cell in the relevant notebook first.")
        st.stop()
    return joblib.load(path)

if mode == "Classification":
    st.subheader("⚡ Appliance identification")
    st.caption("Enter an electrical reading to find the most likely appliance.")
    bundle = load_model("final_model.joblib")
    model, scaler = bundle["model"], bundle["scaler"]
    features, classes = bundle["features"], bundle["class_names"]
    presets = {
        "Toaster": (1972.512, 231.0, 3.465, 801.512),
        "Iron": (2196.170, 231.0, 6.558, 1499.718),
        "TV": (1190.860, 232.0, 0.867, 190.630),
        "Coffee Machine": (2008.202, 232.0, 5.186, 1198.030),
        "Refrigerator": (1149.259, 231.0, 1.285, 252.842),
        "Custom reading": (2000.0, 231.0, 5.0, 1000.0),
    }
    choice = st.selectbox("Try an example", list(presets))
    agg0, volt0, cur0, power0 = presets[choice]
    c1,c2,c3,c4 = st.columns(4)
    agg = c1.number_input("Total power (W)", 0.001, 20000.0, float(agg0), step=10.0)
    volt = c2.number_input("Voltage (V)", 0.001, 300.0, float(volt0), step=1.0)
    cur = c3.number_input("Current (A)", 0.001, 100.0, float(cur0), step=0.001, format="%.3f")
    power = c4.number_input("Appliance power (W)", 0.001, 20000.0, float(power0), step=10.0)
    if st.button("Identify appliance", type="primary", use_container_width=True):
        apparent = volt * cur
        row = {"Aggregated Power":agg, "Voltage":volt, "Current":cur, "Power":power,
               "apparent_power":apparent, "power_factor":min(max(power/apparent,0.0),1.0),
               "load_share":power/agg}
        X = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
        probs = model.predict_proba(pd.DataFrame(scaler.transform(X), columns=features))[0]
        order = np.argsort(probs)[::-1]
        st.divider()
        a,b = st.columns([1,2])
        a.metric("Best match", str(classes[order[0]]))
        a.metric("Model confidence", f"{probs[order[0]]:.1%}")
        b.bar_chart(pd.Series(probs[order[:5]], index=[classes[i] for i in order[:5]]))
        if {classes[order[0]], classes[order[1]]} == {"Iron", "Electric Kettle"}:
            st.info("Iron and Electric Kettle can look very similar using electrical readings alone.")

elif mode == "Regression":
    st.subheader("🏡 Airbnb price estimator")
    st.caption("Estimate a listing's nightly price from its details.")
    bundle = load_model("regression_bundle.joblib")
    model = bundle["model"]
    features = bundle["features"]
    medians = bundle.get("medians", {})
    freq_map = bundle.get("neighbourhood_freq", {})
    category_columns = bundle.get("category_columns", {})
    numeric_defaults = bundle.get("numeric_defaults", {})
    categorical_fields = bundle.get("categorical_fields", {})
    with st.form("regression_form"):
        c1,c2,c3 = st.columns(3)
        raw = {}
        numeric_labels = [
            ("minimum_nights","Minimum nights",1,365),
            ("number_of_reviews","Number of reviews",0,10000),
            ("availability_365","Availability (days/year)",0,365),
            ("host_listings_count","Host listings count",1,1000),
            ("reviews_per_month","Reviews per month",0.0,100.0),
            ("calculated_host_listings_count","Calculated host listings",1,1000),
        ]
        for i,(key,label,lo,hi) in enumerate(numeric_labels):
            with [c1,c2,c3][i%3]:
                raw[key] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                          value=float(numeric_defaults.get(key, lo)), step=1.0)
        for key,label,options in [
            ("neighbourhood_group","Area",categorical_fields.get("neighbourhood_group",["Manhattan","Brooklyn","Queens","Bronx","Staten Island"])),
            ("room_type","Room type",categorical_fields.get("room_type",["Entire home/apt","Private room","Shared room"])),
            ("cancellation_policy","Cancellation policy",categorical_fields.get("cancellation_policy",["strict","moderate","flexible"])),
            ("instant_bookable","Instant bookable",categorical_fields.get("instant_bookable",["TRUE","FALSE"])),
            ("host_identity_verified","Host identity verified",categorical_fields.get("host_identity_verified",["verified","unconfirmed"])),
        ]:
            raw[key] = st.selectbox(label, list(options))
        neighbourhood = st.text_input("Neighbourhood", "Midtown")
        submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)
    if submitted:
        raw["review_intensity"] = raw["number_of_reviews"] / (raw["availability_365"] + 1)
        raw["is_multi_listing_host"] = int(raw.get("host_listings_count", raw.get("calculated_host_listings_count",1)) > 1)
        raw["neighbourhood_freq"] = float(freq_map.get(neighbourhood, 0.0))
        encoded = {k:v for k,v in raw.items() if isinstance(v,(int,float,np.number))}
        for field, selected in [(k,raw[k]) for k in categorical_fields if k in raw]:
            for col in category_columns.get(field, []):
                # notebook uses drop_first=True; saved columns are the exact dummy columns
                encoded[col] = int(col.endswith("_"+str(selected)) or col.endswith("_"+str(selected).replace(" ","_")))
        X = pd.DataFrame([{f: encoded.get(f, medians.get(f, 0.0)) for f in features}], columns=features)
        X = X.replace([np.inf,-np.inf],np.nan).fillna(pd.Series(medians)).fillna(0)
        prediction = float(model.predict(X)[0])
        st.divider()
        st.metric("Estimated nightly price", f"${max(0,prediction):,.0f}")
        st.caption("An estimate from the trained model; actual prices vary by listing and date.")

else:
    st.subheader("🧩 Customer segmentation")
    st.caption("Explore which customer segment a transaction profile belongs to.")
    bundle = load_model("clustering_bundle.joblib")
    scaler, model = bundle["scaler"], bundle["model"]
    features = bundle["features"]
    base_features = bundle.get("base_features", features)
    labels = bundle.get("cluster_names", {})
    defaults = bundle.get("defaults", {})
    with st.form("cluster_form"):
        c1,c2,c3 = st.columns(3)
        raw = {}
        fields = [
            ("Recency","Days since last transaction",0,3650),
            ("Frequency","Number of transactions",1,10000),
            ("Monetary","Total transaction value",0,10000000),
            ("AvgTransactionAmount","Average transaction amount",0,1000000),
            ("MaxTransactionAmount","Largest transaction amount",0,1000000),
            ("AvgAccountBalance","Average account balance",0,10000000),
            ("MaxAccountBalance","Maximum account balance",0,10000000),
            ("AmountPerTransaction","Amount per transaction",0,1000000),
            ("BalanceToSpendRatio","Balance-to-spend ratio",0,100000),
            ("Age","Customer age",18,100),
        ]
        for i,(key,label,lo,hi) in enumerate(fields):
            with [c1,c2,c3][i%3]:
                if key in features:
                    raw[key] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                               value=float(defaults.get(key, max(lo, 30 if key=="Age" else lo))), step=1.0)
        submitted = st.form_submit_button("Find customer segment", type="primary", use_container_width=True)
    if submitted:
        X_base = pd.DataFrame([{f: raw.get(f, defaults.get(f,0.0)) for f in base_features}], columns=base_features)
        scaled_all = scaler.transform(X_base)
        scaled_df = pd.DataFrame(scaled_all, columns=base_features)
        scaled = scaled_df.reindex(columns=features).to_numpy()
        cluster = int(model.predict(scaled)[0])
        st.divider()
        st.metric("Assigned segment", labels.get(str(cluster), f"Segment {cluster + 1}"))
        st.caption("Cluster numbers identify groups, not a ranking. Review the cluster profiles to understand each segment.")
