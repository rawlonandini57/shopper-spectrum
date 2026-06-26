# 🛒 Shopper Spectrum
## Customer Segmentation & Product Recommendations in E-Commerce

---

## 📁 Project Structure

```
shopper_spectrum/
│
├── data/                        ← Dataset goes here (auto-created)
│   └── online_retail.xlsx
│
├── models/                      ← Saved ML models (auto-created by notebook)
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── cluster_labels.pkl
│   └── similarity_df.pkl
│
├── shopper_spectrum.ipynb       ← Main Jupyter Notebook (run this first)
├── app.py                       ← Streamlit Web App
├── data_download.py             ← Dataset downloader script
├── requirements.txt             ← Python dependencies
└── README.md                    ← This file
```

---

## ⚙️ Setup Instructions (Step by Step)

### Step 1 — Install Python & VS Code
- Make sure Python 3.8+ is installed: https://www.python.org/downloads/
- Install VS Code: https://code.visualstudio.com/
- Install the **Python** and **Jupyter** extensions in VS Code

---

### Step 2 — Open the project folder in VS Code
```
File → Open Folder → select the shopper_spectrum folder
```

---

### Step 3 — Open a Terminal in VS Code
```
Terminal → New Terminal
```

---

### Step 4 — Install all required libraries
```bash
pip install -r requirements.txt
```

---

### Step 5 — Download the Dataset
```bash
python data_download.py
```
This will download the dataset into the `data/` folder automatically.

---

### Step 6 — Run the Jupyter Notebook
- Open `shopper_spectrum.ipynb` in VS Code
- Click **"Run All"** (or press Shift+Enter on each cell)
- This will:
  - Clean and analyze the data
  - Build the RFM customer segments
  - Train the KMeans model
  - Build the product recommendation system
  - Save all models to the `models/` folder

---

### Step 7 — Launch the Streamlit App
```bash
streamlit run app.py
```
- A browser window will open automatically at http://localhost:8501
- Use the sidebar to switch between modules

---

## 🎯 App Features

### Module 1 — Product Recommendations
- Enter any product name (partial match supported)
- Get top 5 similar products based on purchase patterns

### Module 2 — Customer Segmentation
- Enter Recency, Frequency, Monetary values
- Predict which segment the customer belongs to
- Get actionable business insights

---

## 📊 Customer Segments

| Segment     | Recency | Frequency | Monetary | Strategy                     |
|-------------|---------|-----------|----------|------------------------------|
| High-Value  | Low     | High      | High     | Loyalty rewards, VIP offers  |
| Regular     | Low     | Medium    | Medium   | Promotions, cross-sell       |
| Occasional  | High    | Low       | Low      | Limited-time offers          |
| At-Risk     | High    | Very Low  | Very Low | Re-engagement campaigns      |

---

## 🛠 Tech Stack
- **Pandas, NumPy** — Data processing
- **Matplotlib, Seaborn, Plotly** — Visualizations
- **Scikit-Learn** — KMeans Clustering, StandardScaler, Cosine Similarity
- **Streamlit** — Web application
- **Joblib** — Model saving/loading
