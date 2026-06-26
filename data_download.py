"""
Step 0: Download the dataset from Google Drive.
Run this script ONCE before opening the notebook.
Usage: python data_download.py
"""

import os
import gdown

FILE_ID = "1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-"
OUTPUT  = "data/online_retail.xlsx"

os.makedirs("data", exist_ok=True)

if os.path.exists(OUTPUT):
    print(f"✅ Dataset already exists at '{OUTPUT}'. Skipping download.")
else:
    print("⬇️  Downloading dataset from Google Drive ...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, OUTPUT, quiet=False)
    print(f"✅ Dataset saved to '{OUTPUT}'")
