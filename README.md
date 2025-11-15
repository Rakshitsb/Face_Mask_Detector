# Face Mask Detection — Streamlit App

A real-time face mask detection web app built with TensorFlow (MobileNetV2), OpenCV and Streamlit. Upload images or use your webcam for live detection.

**Author:** Rakshit Badiger

---

## Features
- ✅ Real-time webcam detection (WebRTC)
- ✅ Image upload detection with bounding boxes and confidence
- ✅ Multi-face detection
- ✅ Lightweight MobileNetV2-based model

---

## Repo structure
/ (root)
├─ app.py
├─ model.h5 # optional (or provide MODEL_URL)
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ assets/
│ └─ images/ # screenshots for README


---

## How to run locally

1. Clone repo:
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>


(Recommended) Create virtual environment:

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Add your model:

Option A (recommended for Streamlit Cloud): Put model.h5 in repo root.

Option B: Host model externally and set MODEL_URL environment variable to a direct download URL.

Run app:

streamlit run app.py


Open the URL (usually http://localhost:8501).