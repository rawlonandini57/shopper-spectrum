"""
Dataset Setup Script
Since you have the dataset file (online_retail.csv), just run this script
to verify it is in the correct location.

Usage: python data_download.py
"""

import os

CSV_PATH = "data/online_retail.csv"

os.makedirs("data", exist_ok=True)

if os.path.exists(CSV_PATH):
    size_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
    print(f"Dataset found at '{CSV_PATH}' ({size_mb:.1f} MB)")
    print("You are ready to run the notebook!")
else:
    print("Dataset NOT found.")
    print(f"Please copy your 'online_retail.csv' file into the 'data/' folder.")
    print(f"Expected path: {CSV_PATH}")
