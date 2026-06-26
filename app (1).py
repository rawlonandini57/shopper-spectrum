"""
Shopper Spectrum - Streamlit App
Run: streamlit run app.py

NOTE: Place the models/ folder in the same directory as this app.py file.
The dataset CSV is NOT needed to run this app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700; color: #1a1a2e;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center; color: #666;
        font-size: 1rem; margin-bottom: 2rem;
    }
    .segment-card {
        padding: 1.2rem 1.5rem; border-radius: 12px;
        margin-top: 1rem; font-size: 1.1rem; font-weight: 600;
    }
    .high-value  { background:#d4edda; color:#155724; border-left:5px solid #28a745; }
    .regular     { background:#cce5ff; color:#004085; border-left:5px solid #007bff; }
    .occasional  { background:#fff3cd; color:#856404; border-left:5px solid #ffc107; }
    .at-risk     { background:#f8d7da; color:#721c24; border-left:5px solid #dc3545; }
    .rec-item {
        background:#f8f9fa; padding:0.7rem 1rem;
        border-radius:8px; margin:0.4rem 0; border-left:4px solid #6c63ff;
    }
</style>
""", unsafe_allow_html=True)

# ── Find models folder ───────────────────────────────────────────────
# Look in same folder as app.py, then subfolder models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    required = ["kmeans_model.pkl", "scaler.pkl", "cluster_labels.pkl", "similarity_df.pkl"]
    missing  = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None, None, None, None, missing
    kmeans         = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    scaler         = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    cluster_labels = joblib.load(os.path.join(MODELS_DIR, "cluster_labels.pkl"))
    similarity_df  = pd.read_pickle(os.path.join(MODELS_DIR, "similarity_df.pkl"))
    return kmeans, scaler, cluster_labels, similarity_df, []

kmeans, scaler, cluster_labels, similarity_df, missing_files = load_models()

# ── Header ───────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Customer Segmentation & Product Recommendations</div>', unsafe_allow_html=True)
st.divider()

# ── Model check ──────────────────────────────────────────────────────
if missing_files:
    st.error(f"**Model files not found:** {', '.join(missing_files)}")
    st.info(f"**Expected location:** `{MODELS_DIR}`")
    st.markdown("""
    **Fix:**
    1. Download the 4 `.pkl` files provided to you
    2. Create a folder called `models` next to `app.py`
    3. Place all 4 files inside it
    4. Refresh this page
    """)
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 Navigation")
    page = st.radio("Choose Module", [
        "🎯 Product Recommendations",
        "👥 Customer Segmentation",
        "ℹ️ About"
    ])
    st.divider()
    st.success("✅ Models Loaded")
    st.markdown(f"- Products in DB: **{len(similarity_df)}**")
    st.markdown(f"- Customer Segments: **{len(set(cluster_labels.values()))}**")

# ════════════════════════════════════════════════════════════════════
# PAGE 1 – Product Recommendations
# ════════════════════════════════════════════════════════════════════
if page == "🎯 Product Recommendations":
    st.header("🎯 Product Recommendation Engine")
    st.markdown("Type a product name to get **top 5 similar product recommendations**.")

    col1, col2 = st.columns([3, 1])
    with col1:
        product_input = st.text_input(
            "🔍 Product Name",
            placeholder="e.g. WHITE HANGING HEART, METAL SIGN, LUNCH BAG ..."
        )
    with col2:
        top_n = st.selectbox("Top N", [3, 5, 10], index=1)

    with st.expander("📋 Browse available products (first 50)"):
        st.dataframe(
            pd.DataFrame(similarity_df.index[:50], columns=["Product Name"]),
            use_container_width=True, hide_index=True
        )

    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            matches = [p for p in similarity_df.index if product_input.upper() in p.upper()]
            if not matches:
                st.error(f"❌ No product found matching **'{product_input}'**. Try a different keyword.")
            else:
                matched = matches[0]
                st.success(f"✅ Matched: **{matched}**")

                similar = similarity_df[matched].sort_values(ascending=False).drop(index=matched)
                top = similar.head(top_n).reset_index()
                top.columns = ["Product", "Similarity Score"]
                top["Similarity Score"] = top["Similarity Score"].round(4)
                top.index = top.index + 1

                st.subheader(f"🏆 Top {top_n} Similar Products")
                for i, row in top.iterrows():
                    pct = int(row["Similarity Score"] * 100)
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    st.markdown(
                        f'<div class="rec-item">#{i} &nbsp; <b>{row["Product"]}</b>'
                        f'<br><small>Similarity: {bar} {pct}%</small></div>',
                        unsafe_allow_html=True
                    )
                st.divider()
                st.dataframe(top, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 2 – Customer Segmentation
# ════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.header("👥 Customer Segment Predictor")
    st.markdown("Enter a customer's **RFM values** to predict their segment.")

    with st.expander("📖 What is RFM?"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**R – Recency**\nDays since last purchase. Lower = more recent.")
        c2.markdown("**F – Frequency**\nNumber of unique orders. Higher = more loyal.")
        c3.markdown("**M – Monetary**\nTotal amount spent. Higher = more valuable.")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        recency   = st.number_input("📅 Recency (days)", min_value=1, max_value=1000, value=30)
    with col2:
        frequency = st.number_input("🔄 Frequency (orders)", min_value=1, max_value=500, value=5)
    with col3:
        monetary  = st.number_input("💰 Monetary (£ spent)", min_value=1.0, max_value=100000.0, value=500.0, step=50.0)

    if st.button("🔮 Predict Segment", type="primary", use_container_width=True):
        scaled   = scaler.transform([[recency, frequency, monetary]])
        cluster  = kmeans.predict(scaled)[0]
        segment  = cluster_labels.get(cluster, "Unknown")

        style_map = {
            "High-Value": ("high-value", "🌟", "Best customers! Focus on loyalty rewards and VIP offers."),
            "Regular":    ("regular",    "✅", "Steady buyers. Keep them engaged with regular promotions."),
            "Occasional": ("occasional", "⚠️", "Infrequent buyers. Try limited-time offers to increase visits."),
            "At-Risk":    ("at-risk",    "🚨", "Inactive customers. Send win-back emails with special discounts."),
        }
        css, icon, advice = style_map.get(segment, ("regular", "❓", ""))

        st.markdown(f"""
        <div class="segment-card {css}">
            {icon} Predicted Segment: <span style="font-size:1.4rem">&nbsp;{segment}</span>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"💡 **Business Insight:** {advice}")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Recency",   f"{recency} days")
        c2.metric("Frequency", f"{frequency} orders")
        c3.metric("Monetary",  f"£{monetary:,.2f}")

        st.divider()
        st.subheader("📋 All Segment Descriptions")
        st.dataframe(pd.DataFrame([
            {"Segment":"🌟 High-Value",  "Recency":"Low",  "Frequency":"High","Monetary":"High","Action":"Loyalty rewards, VIP offers"},
            {"Segment":"✅ Regular",     "Recency":"Low",  "Frequency":"Med", "Monetary":"Med", "Action":"Promotions, cross-sell"},
            {"Segment":"⚠️ Occasional", "Recency":"High", "Frequency":"Low", "Monetary":"Low", "Action":"Limited-time offers"},
            {"Segment":"🚨 At-Risk",     "Recency":"High", "Frequency":"Low", "Monetary":"Low", "Action":"Re-engagement campaigns"},
        ]), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 3 – About
# ════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    ### 🛒 Shopper Spectrum
    **Domain:** E-Commerce and Retail Analytics

    This project analyzes online retail transaction data to:
    - Segment customers using **RFM Analysis** and **KMeans Clustering**
    - Recommend similar products using **Item-Based Collaborative Filtering**

    ---
    ### 🔧 Tech Stack
    | Component | Technology |
    |---|---|
    | Data Processing | Pandas, NumPy |
    | Visualization | Matplotlib, Seaborn, Plotly |
    | Clustering | Scikit-Learn KMeans |
    | Recommendation | Cosine Similarity |
    | Web App | Streamlit |

    ---
    ### 👤 Customer Segments
    | Segment | Description |
    |---|---|
    | 🌟 High-Value | Recent, frequent, high spend |
    | ✅ Regular | Steady purchasers |
    | ⚠️ Occasional | Infrequent buyers |
    | 🚨 At-Risk | Long inactive, low engagement |
    """)
