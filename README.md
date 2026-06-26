# Shopper Spectrum
## Customer Segmentation & Product Recommendations in E-Commerce

---

## Project Structure

```
shopper_spectrum/
│
├── data/                        ← Put your dataset here
│   └── online_retail.csv        ← Copy your CSV file here
│
├── models/                      ← Auto-created when notebook runs
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── cluster_labels.pkl
│   └── similarity_df.pkl
│
├── shopper_spectrum.ipynb       ← Main Jupyter Notebook (run first)
├── app.py                       ← Streamlit Web App
├── data_download.py             ← Dataset location checker
├── requirements.txt             ← Python dependencies
└── README.md                    ← This file
```

---

## Setup Instructions (Step by Step)

### Step 1 — Install Python & VS Code
- Python 3.8+: https://www.python.org/downloads/
- VS Code: https://code.visualstudio.com/
- Install Python and Jupyter extensions in VS Code

### Step 2 — Open the project folder in VS Code
```
File → Open Folder → select the shopper_spectrum folder
```

### Step 3 — Copy your dataset
Copy `online_retail.csv` into the `data/` folder:
```
shopper_spectrum/
└── data/
    └── online_retail.csv   ← here
```

### Step 4 — Open Terminal and install libraries
```bash
pip install -r requirements.txt
```

### Step 5 — Verify dataset location
```bash
python data_download.py
```
Should print: "Dataset found at 'data/online_retail.csv'"

### Step 6 — Run the Jupyter Notebook
- Open `shopper_spectrum.ipynb`
- Click Run All (Ctrl+Shift+P → Run All)
- Wait for all cells to complete (~3-5 minutes)
- Last cell should print: "All models saved to /models!"

### Step 7 — Launch the Streamlit App
```bash
streamlit run app.py
```
Opens automatically at http://localhost:8501

---

## App Modules

| Module | Description |
|---|---|
| Product Recommender | Enter a product name → get top-5 similar products |
| Customer Segmentation | Enter RFM values → predict customer segment |

---

## Customer Segments

| Segment | Description | Action |
|---|---|---|
| High-Value | Recent, frequent, high spend | Loyalty rewards, VIP offers |
| Regular | Steady buyers | Promotions, cross-sell |
| Occasional | Rare purchases | Limited-time offers |
| At-Risk | Long inactive | Win-back campaigns |

---

## Tech Stack
- Pandas, NumPy — Data processing
- Matplotlib, Seaborn, Plotly — Visualisations
- Scikit-Learn — KMeans, StandardScaler, Cosine Similarity
- Streamlit — Web application
- Joblib — Model saving/loading
