"""ML Studio -- one Streamlit app for the 23CSE301 capstone: Classification, Regression and Clustering.

Pick a task at the top of the page; only that page is shown (and only its model is loaded).

Models live in ../models/ :
    final_model.joblib        classification (appliance identifier, Random Forest, electrical features only)
    regression_bundle.joblib  regression     (Airbnb nightly price, Random Forest)
    clustering_bundle.joblib  clustering     (bank customer segments, K-Means with k = 4)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Studio", page_icon="✦", layout="centered")

MODELS = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource(show_spinner="Loading model...")
def load_bundle(name: str):
    return joblib.load(MODELS / name)


st.markdown(
    """
    <style>
    .hero{padding:1.1rem 1.4rem;border-radius:18px;margin:.2rem 0 1.2rem}
    .hero h2{margin:0;font-size:1.65rem}
    .hero p{margin:.3rem 0 0;font-size:1rem}
    .card{padding:1.3rem 1.5rem;border-radius:18px;margin:1rem 0}
    .card .label{font-size:.8rem;letter-spacing:.08em;font-weight:600;opacity:.75}
    .card .big{font-size:2.8rem;font-weight:700;line-height:1.15;margin:.15rem 0 .3rem}
    .card .note{font-size:.9rem;opacity:.85}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================================================
# PAGE 1 -- CLASSIFICATION: identify the appliance from one electrical reading
# =====================================================================================================
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


def engineer(agg, volt, cur, power, features) -> pd.DataFrame:
    """Rebuild the notebook's engineered features. Must stay identical to Section B3 of classification.ipynb."""
    apparent = volt * cur                                   # apparent_power = V x I
    power_factor = min(max(power / apparent, 0.0), 1.0)     # real / apparent, clipped to [0, 1]
    load_share = power / agg                                # appliance's share of the total load
    row = {
        "Aggregated Power": agg, "Voltage": volt, "Current": cur, "Power": power,
        "apparent_power": apparent, "power_factor": power_factor, "load_share": load_share,
    }
    return pd.DataFrame([row])[features]


def apply_preset():
    choice = st.session_state["preset"]
    if choice != CUSTOM:
        agg, volt, cur, power = PRESETS[choice]
        st.session_state.update(agg=agg, volt=volt, cur=cur, power=power)


def page_classification():
    bundle = load_bundle("final_model.joblib")
    model, scaler = bundle["model"], bundle["scaler"]
    FEATURES, CLASSES = bundle["features"], bundle["class_names"]

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
        return

    if power > agg:
        st.warning("Real power is higher than the aggregated power. In the training data the aggregated load is never "
                   "below the appliance's own power, so this reading is unusual.")
    if power > volt * cur * 1.05:
        st.warning("Real power exceeds voltage × current, which is physically impossible (power factor > 1). "
                   "Check the inputs; the power factor will be capped at 1.")

    X_raw = engineer(agg, volt, cur, power, FEATURES)
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


# =====================================================================================================
# PAGE 2 -- REGRESSION: Airbnb nightly price
# =====================================================================================================
# Approximate centre of each area (latitude, longitude). The model sees location only as lat/long.
AREAS = {
    "Midtown, Manhattan": (40.7549, -73.9840),
    "Upper West Side, Manhattan": (40.7870, -73.9754),
    "East Village, Manhattan": (40.7265, -73.9815),
    "Harlem, Manhattan": (40.8116, -73.9465),
    "Williamsburg, Brooklyn": (40.7081, -73.9571),
    "Park Slope, Brooklyn": (40.6710, -73.9814),
    "Bushwick, Brooklyn": (40.6944, -73.9213),
    "Astoria, Queens": (40.7644, -73.9235),
    "Long Island City, Queens": (40.7447, -73.9485),
    "Flushing, Queens": (40.7675, -73.8331),
    "Fordham, Bronx": (40.8610, -73.8905),
    "St. George, Staten Island": (40.6437, -74.0735),
}
PRICE_AXIS_MAX = 1250   # the training prices top out around 1,200


def page_regression():
    st.markdown(
        "<div class='hero' style='background:#fff1e6;border:1px solid #f3d2bd;color:#5a2d17'>"
        "<h2>🏡 Airbnb price studio</h2><p>Pick an area, describe the listing, see a nightly price.</p></div>",
        unsafe_allow_html=True,
    )
    rb = load_bundle("regression_bundle.joblib")
    model, feats, med = rb["model"], rb["features"], rb["medians"]

    area = st.selectbox("Where is the listing?", list(AREAS), key="reg_area")
    lat, lon = AREAS[area]
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=12, size=90, color="#e0763a", height=230)

    c1, c2 = st.columns(2)
    with c1:
        nights = st.slider("Minimum nights", 1, 9, int(med["minimum_nights"]), key="reg_nights")
        avail = st.slider("Days available per year", 0, 365, int(med["availability_365"]), key="reg_avail")
        year = st.slider("Year built", 2003, 2022, int(med["construction_year"]), key="reg_year")
        rating = st.slider("Guest rating (1–5)", 1, 5, int(med["review_rate_number"]), key="reg_rating")
    with c2:
        reviews = st.number_input("Number of reviews", 0, 1000, int(med["number_of_reviews"]), key="reg_reviews")
        per_month = st.number_input("Reviews per month", 0.0, 30.0, float(med["reviews_per_month"]), step=0.1, key="reg_pm")
        hosts = st.number_input("Listings by this host", 1, 350, int(med["host_listings_count"]), key="reg_hosts")

    # Same engineered features as the notebook. neighbourhood_freq could not be rebuilt from a typed name (the
    # lookup table was not saved with the model), so it is held at its training median.
    row = {
        "lat": lat, "long": lon, "construction_year": year, "minimum_nights": nights,
        "number_of_reviews": reviews, "reviews_per_month": per_month, "review_rate_number": rating,
        "host_listings_count": hosts, "availability_365": avail,
        "review_intensity": reviews / (avail + 1), "is_multi_listing_host": int(hosts > 1),
        "neighbourhood_freq": med["neighbourhood_freq"],
    }
    X = pd.DataFrame([row])[feats]
    estimate = float(model.predict(X)[0])

    # Spread of the 100 trees = how much the forest disagrees with itself
    per_tree = np.array([t.predict(X.to_numpy(dtype=np.float32))[0] for t in model.estimators_])
    low, high = np.percentile(per_tree, [10, 90])

    pct = lambda v: max(0.0, min(100.0, v / PRICE_AXIS_MAX * 100))
    st.markdown(
        f"""
        <div class='card' style='background:#fff1e6;border:1px solid #f3d2bd;color:#5a2d17'>
          <div class='label'>ESTIMATED PRICE PER NIGHT</div>
          <div class='big'>${max(estimate, 0):,.0f}</div>
          <div>Likely range <b>${max(low, 0):,.0f} – ${max(high, 0):,.0f}</b></div>
          <div style='position:relative;height:10px;background:#f3d2bd;border-radius:6px;margin:1.1rem 0 .3rem'>
            <div style='position:absolute;left:{pct(low):.1f}%;width:{max(pct(high) - pct(low), 1):.1f}%;height:10px;background:#e0763a;border-radius:6px'></div>
            <div style='position:absolute;left:{pct(estimate):.1f}%;top:-4px;width:4px;height:18px;background:#5a2d17;border-radius:2px'></div>
          </div>
          <div style='display:flex;justify-content:space-between;font-size:.75rem;opacity:.7'><span>$0</span><span>${PRICE_AXIS_MAX:,}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Illustrative estimate: in this public dataset, prices depend only weakly on listing details, "
               "so expect a wide range.")


# =====================================================================================================
# PAGE 3 -- CLUSTERING: bank customer segments (K-Means, k = 4)
# =====================================================================================================
LOG_FEATURES = ["Monetary", "AvgTransactionAmount", "MaxTransactionAmount", "AvgAccountBalance",
                "MaxAccountBalance", "AmountPerTransaction", "BalanceToSpendRatio"]
# (phrase when the segment's average is LOW, phrase when it is HIGH)
PHRASES = {
    "Recency": ("transacted recently", "has not transacted for a long time"),
    "Monetary": ("spends little", "spends a lot"),
    "AvgAccountBalance": ("keeps a low balance", "keeps a high balance"),
    "BalanceToSpendRatio": ("spends most of what they hold", "holds far more than they spend"),
    "Age": ("younger", "older"),
}


def to_model_space(recency, spend, balance, age, b):
    """Raw customer values -> the scaled 5-feature vector K-Means was fitted on (same steps as the notebook)."""
    base = {
        "Recency": recency, "Frequency": b["defaults"].get("Frequency", 1.0), "Monetary": spend,
        "AvgTransactionAmount": spend, "MaxTransactionAmount": spend,          # a single-purchase customer
        "AvgAccountBalance": balance, "MaxAccountBalance": balance,
        "AmountPerTransaction": spend, "BalanceToSpendRatio": balance / (spend + 1), "Age": age,
    }
    row = pd.DataFrame([base])[b["base_features"]]
    for col in LOG_FEATURES:                                   # 1) log-transform the skewed features
        row[col] = np.log1p(row[col])
    z = pd.DataFrame(b["scaler"].transform(row), columns=b["base_features"])   # 2) standardise (10 features)
    z = z.clip(-2.7, 2.7)                                      # 3) approximates the notebook's IQR capping
    return z[b["features"]].to_numpy()                         # 4) keep the 5 features K-Means uses


@st.cache_resource
def describe_segments():
    """Name every segment from its centre, so the labels follow the model if it is ever re-trained."""
    b = load_bundle("clustering_bundle.joblib")
    km, sc, feats = b["model"], b["scaler"], b["features"]
    centres = pd.DataFrame(km.cluster_centers_, columns=feats)               # in z-score units
    share = pd.Series(km.labels_).value_counts(normalize=True).sort_index()

    typical = {}
    for f in feats:                                                          # back to real units
        i = list(sc.feature_names_in_).index(f)
        val = centres[f] * sc.scale_[i] + sc.mean_[i]
        typical[f] = np.expm1(val) if f in LOG_FEATURES else val
    typical = pd.DataFrame(typical)

    def name_of(z):
        if z["Recency"] >= 1.0:
            return "Dormant customers"
        if z["Monetary"] >= 0.5:
            return "Big spenders"
        if z["AvgAccountBalance"] <= -1.0:
            return "Low-balance customers"
        if z["BalanceToSpendRatio"] >= 0.5 and z["Monetary"] <= -0.5:
            return "Careful savers"
        return None

    segments, used = {}, set()
    for k in range(len(centres)):
        z = centres.loc[k]
        name = name_of(z)
        if name is None or name in used:
            name = f"Segment {k + 1}"
        used.add(name)
        strong = z[[f for f in feats if f in PHRASES and abs(z[f]) >= 0.4]].abs().sort_values(ascending=False)
        bullets = [PHRASES[f][int(z[f] > 0)] for f in strong.index[:3]]
        segments[k] = {
            "name": name, "share": float(share.get(k, 0.0)), "bullets": bullets,
            "typical": typical.loc[k].to_dict(),
        }
    return segments


def page_clustering():
    st.markdown(
        "<div class='hero' style='background:#e6f5f1;border:1px solid #bfe3da;color:#14423a'>"
        "<h2>🧩 Customer segment finder</h2><p>Describe a bank customer and see which of four groups they belong to.</p></div>",
        unsafe_allow_html=True,
    )
    b = load_bundle("clustering_bundle.joblib")
    km, d = b["model"], b["defaults"]
    segments = describe_segments()

    c1, c2 = st.columns(2)
    with c1:
        recency = st.slider("Days since last transaction", 0, 365, int(d["Recency"]), key="cl_recency")
        age = st.slider("Age", 18, 90, int(round(d["Age"])), key="cl_age")
    with c2:
        spend = st.number_input("Total spend", 0.0, 10_000_000.0, float(d["Monetary"]), step=50.0, key="cl_spend")
        balance = st.number_input("Average account balance", 0.0, 100_000_000.0,
                                  float(round(d["AvgAccountBalance"])), step=500.0, key="cl_balance")

    z = to_model_space(recency, spend, balance, age, b)
    cluster = int(km.predict(z)[0])
    seg = segments[cluster]

    bullets = "".join(f"<li>{t[0].upper() + t[1:]}</li>" for t in seg["bullets"]) or "<li>Close to the average customer</li>"
    st.markdown(
        f"""
        <div class='card' style='background:#e6f5f1;border:1px solid #bfe3da;color:#14423a'>
          <div class='label'>THIS CUSTOMER BELONGS TO</div>
          <div class='big'>{seg['name']}</div>
          <div class='note'>{seg['share']:.0%} of customers are in this group</div>
          <ul style='margin:.8rem 0 0;padding-left:1.2rem'>{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # How close the customer is to each group (a rough guide, not a probability)
    dist = np.linalg.norm(km.cluster_centers_ - z, axis=1)
    closeness = np.exp(-dist) / np.exp(-dist).sum()
    st.caption("Closeness to each group")
    st.bar_chart(pd.Series(closeness, index=[segments[k]["name"] for k in range(len(closeness))], name="closeness"))

    with st.expander("Compare with the typical customer in this group"):
        t = seg["typical"]
        st.dataframe(
            pd.DataFrame({
                "": ["Days since last transaction", "Total spend", "Average account balance", "Age"],
                "This customer": [f"{recency:,.0f}", f"{spend:,.0f}", f"{balance:,.0f}", f"{age:,.0f}"],
                "Typical in group": [f"{t['Recency']:,.0f}", f"{t['Monetary']:,.0f}", f"{t['AvgAccountBalance']:,.0f}", f"{t['Age']:,.0f}"],
            }),
            hide_index=True, use_container_width=True,
        )


# =====================================================================================================
# PAGE SELECTOR
# =====================================================================================================
PAGES = {
    "⚡ Classification": page_classification,
    "🏡 Regression": page_regression,
    "🧩 Clustering": page_clustering,
}

st.markdown("<div style='font-weight:700;letter-spacing:.12em;opacity:.65;font-size:.85rem'>✦ ML STUDIO</div>",
            unsafe_allow_html=True)
choice = st.segmented_control("Task", list(PAGES), default=list(PAGES)[0], key="page", label_visibility="collapsed")
choice = choice or st.session_state.get("_last_page", list(PAGES)[0])     # un-clicking keeps the current page
st.session_state["_last_page"] = choice
st.write("")

PAGES[choice]()