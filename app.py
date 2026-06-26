"""
Shopper Spectrum - Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .segment-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .high-value  { background: #d4edda; color: #155724; border-left: 5px solid #28a745; }
    .regular     { background: #cce5ff; color: #004085; border-left: 5px solid #007bff; }
    .occasional  { background: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }
    .at-risk     { background: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
    .rec-item {
        background: #f8f9fa;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        border-left: 4px solid #6c63ff;
    }
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load saved models from the models/ directory."""
    models_path = "models"
    required = ["kmeans_model.pkl", "scaler.pkl", "cluster_labels.pkl", "similarity_df.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(models_path, f))]
    if missing:
        return None, None, None, None, missing

    kmeans         = joblib.load(f"{models_path}/kmeans_model.pkl")
    scaler         = joblib.load(f"{models_path}/scaler.pkl")
    cluster_labels = joblib.load(f"{models_path}/cluster_labels.pkl")
    similarity_df  = pd.read_pickle(f"{models_path}/similarity_df.pkl")
    return kmeans, scaler, cluster_labels, similarity_df, []

kmeans, scaler, cluster_labels, similarity_df, missing_files = load_models()

# ── Header ─────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Customer Segmentation & Product Recommendations</div>', unsafe_allow_html=True)
st.divider()

# ── Check models loaded ────────────────────────────────────────────
if missing_files:
    st.error(f"""
    ⚠️ **Model files not found:** {', '.join(missing_files)}

    Please run the Jupyter notebook first to train and save the models.
    1. Open `shopper_spectrum.ipynb` in VS Code
    2. Run all cells
    3. Come back and refresh this app
    """)
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.title("Navigation")
    page = st.radio("Choose Module", [
        "🎯 Product Recommendations",
        "👥 Customer Segmentation",
        "ℹ️ About"
    ])
    st.divider()
    st.markdown("**Models Loaded ✅**")
    st.markdown(f"- Products in DB: **{len(similarity_df)}**")
    st.markdown(f"- Segments: **{len(set(cluster_labels.values()))}**")

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 – Product Recommendations
# ═══════════════════════════════════════════════════════════════════
if page == "🎯 Product Recommendations":
    st.header("🎯 Product Recommendation Engine")
    st.markdown("Enter a product name to get **5 similar product recommendations** based on customer purchase patterns.")

    col1, col2 = st.columns([3, 1])
    with col1:
        product_input = st.text_input(
            "🔍 Product Name",
            placeholder="e.g. WHITE HANGING HEART, METAL SIGN, LUNCH BAG ...",
            help="You can type part of the product name"
        )
    with col2:
        top_n = st.selectbox("Top N", [3, 5, 10], index=1)

    # Show sample products
    with st.expander("📋 Browse available products (first 30)"):
        sample = pd.DataFrame(similarity_df.index[:30], columns=["Product Name"])
        st.dataframe(sample, use_container_width=True, hide_index=True)

    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            matches = [p for p in similarity_df.index if product_input.upper() in p.upper()]

            if not matches:
                st.error(f"❌ No product found matching **'{product_input}'**. Try a different keyword.")
            else:
                matched_product = matches[0]
                st.success(f"✅ Matched: **{matched_product}**")

                similar = similarity_df[matched_product].sort_values(ascending=False)
                similar = similar.drop(index=matched_product)
                top = similar.head(top_n).reset_index()
                top.columns = ["Product", "Similarity Score"]
                top["Similarity Score"] = top["Similarity Score"].round(4)
                top.index = top.index + 1  # start rank from 1

                st.subheader(f"🏆 Top {top_n} Similar Products")
                for i, row in top.iterrows():
                    score_pct = int(row["Similarity Score"] * 100)
                    bar = "█" * (score_pct // 5) + "░" * (20 - score_pct // 5)
                    st.markdown(
                        f'<div class="rec-item">#{i} &nbsp; <b>{row["Product"]}</b>'
                        f'<br><small>Similarity: {bar} {score_pct}%</small></div>',
                        unsafe_allow_html=True
                    )

                st.divider()
                st.dataframe(top, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 – Customer Segmentation
# ═══════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.header("👥 Customer Segment Predictor")
    st.markdown("Enter a customer's **RFM values** to predict their segment.")

    # Explain RFM
    with st.expander("📖 What is RFM?"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**R – Recency**\nHow many days since the customer's last purchase. Lower = more recent.")
        with c2:
            st.markdown("**F – Frequency**\nHow many unique orders the customer has made. Higher = more loyal.")
        with c3:
            st.markdown("**M – Monetary**\nTotal amount spent by the customer. Higher = more valuable.")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "📅 Recency (days since last purchase)",
            min_value=1, max_value=1000, value=30,
            help="1 = purchased yesterday, 365 = purchased a year ago"
        )
    with col2:
        frequency = st.number_input(
            "🔄 Frequency (number of orders)",
            min_value=1, max_value=500, value=5,
            help="Total number of unique invoices/orders"
        )
    with col3:
        monetary = st.number_input(
            "💰 Monetary (total spend £)",
            min_value=1.0, max_value=100000.0, value=500.0, step=50.0,
            help="Total amount spent across all orders"
        )

    if st.button("🔮 Predict Customer Segment", type="primary", use_container_width=True):
        input_data = np.array([[recency, frequency, monetary]])
        scaled_input = scaler.transform(input_data)
        cluster_num = kmeans.predict(scaled_input)[0]
        segment = cluster_labels.get(cluster_num, "Unknown")

        # Style based on segment
        style_map = {
            "High-Value": ("high-value",  "🌟", "This customer is highly valuable! They purchase frequently, recently, and spend a lot. Focus on loyalty rewards and premium offers."),
            "Regular":    ("regular",     "✅", "A steady and reliable customer. Keep them engaged with regular promotions and personalized recommendations."),
            "Occasional": ("occasional",  "⚠️", "This customer purchases occasionally. Try win-back campaigns and limited-time offers to increase engagement."),
            "At-Risk":    ("at-risk",     "🚨", "This customer hasn't purchased in a long time. Send re-engagement emails and special discounts to bring them back."),
        }
        css_class, icon, advice = style_map.get(segment, ("regular", "❓", ""))

        st.markdown(f"""
        <div class="segment-card {css_class}">
            {icon} Predicted Segment: <span style="font-size:1.4rem">{segment}</span>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"💡 **Business Insight:** {advice}")

        st.divider()
        st.subheader("📊 Input Summary")
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Recency", f"{recency} days")
        with summary_cols[1]:
            st.metric("Frequency", f"{frequency} orders")
        with summary_cols[2]:
            st.metric("Monetary", f"£{monetary:,.2f}")

        # Segment descriptions table
        st.divider()
        st.subheader("📋 All Segment Descriptions")
        seg_df = pd.DataFrame([
            {"Segment": "🌟 High-Value",  "Recency": "Low",  "Frequency": "High", "Monetary": "High", "Action": "Loyalty rewards, VIP offers"},
            {"Segment": "✅ Regular",     "Recency": "Low",  "Frequency": "Med",  "Monetary": "Med",  "Action": "Regular promotions, cross-sell"},
            {"Segment": "⚠️ Occasional", "Recency": "High", "Frequency": "Low",  "Monetary": "Low",  "Action": "Limited-time offers, incentives"},
            {"Segment": "🚨 At-Risk",     "Recency": "High", "Frequency": "Low",  "Monetary": "Low",  "Action": "Re-engagement email campaigns"},
        ])
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 – About
# ═══════════════════════════════════════════════════════════════════
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
    ### 📊 Dataset
    Online retail transaction data (2022–2023) with columns:
    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

    ---
    ### 👤 Customer Segments
    | Segment | Description |
    |---|---|
    | 🌟 High-Value | Recent, frequent, high spend |
    | ✅ Regular | Steady purchasers |
    | ⚠️ Occasional | Infrequent buyers |
    | 🚨 At-Risk | Long inactive, low engagement |
    """)
