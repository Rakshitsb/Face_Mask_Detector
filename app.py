# app.py
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

IMG_SIZE = (128, 128)
LOCAL_MODEL_PATH = "model.h5"   # Put model.h5 in repo root for Streamlit Cloud
DRIVE_MODEL_PATH = "/content/drive/MyDrive/face_mask_detector.h5"  # used when running in Colab

# -------------------------
# UI Styles
# -------------------------
st.markdown(
    """
    <style>
    .main-header { font-size: 42px; font-weight: bold; text-align: center; color: #1f77b4; }
    .sub-header { font-size: 18px; text-align: center; color: #666; margin-bottom: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">😷 Real-Time Face Mask Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect face masks using webcam or upload images</p>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Model loading helper
# -------------------------
@st.cache_resource
def load_face_mask_model():
    # Priority:
    # 1) local model.h5 in repo root (recommended for Streamlit Cloud)
    # 2) environment variable MODEL_URL -> download at startup (for external hosting)
    # 3) known Colab Drive path (when running in Colab)
    if os.path.exists(LOCAL_MODEL_PATH):
        model_path = LOCAL_MODEL_PATH
    elif os.getenv("MODEL_URL"):
        # Attempt to download the model once and cache it
        import requests
        url = os.getenv("MODEL_URL")
        model_path = LOCAL_MODEL_PATH
        if not os.path.exists(model_path):
            with st.spinner("Downloading model from MODEL_URL..."):
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    elif os.path.exists(DRIVE_MODEL_PATH):
        model_path = DRIVE_MODEL_PATH
    else:
        raise FileNotFoundError(
            "No model found. Place model.h5 in repo root or set MODEL_URL env var or put model in Drive."
        )

    model = load_model(model_path)
    return model

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load models
with st.spinner("🔄 Loading AI models..."):
    try:
        model = load_face_mask_model()
        face_cascade = load_face_detector()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# -------------------------
# Video Transformer for real-time webcam
# -------------------------
class MaskDetectorTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.face_cascade = face_cascade
        self.mask_count = 0
        self.no_mask_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        self.mask_count = 0
        self.no_mask_count = 0

        for (x, y, w, h) in faces:
            face = img[y : y + h, x : x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, IMG_SIZE)
            face = img_to_array(face) / 255.0
            face = np.expand_dims(face, axis=0)

            pred = self.model.predict(face, verbose=0)[0][0]

            if pred > 0.5:
                label = "Mask"
                conf = pred * 100
                color = (0, 255, 0)
                self.mask_count += 1
            else:
                label = "No Mask"
                conf = (1 - pred) * 100
                color = (0, 0, 255)
                self.no_mask_count += 1

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            label_text = f"{label}: {conf:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x, y - 30), (x + tw + 10, y), color, -1)
            cv2.putText(img, label_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

# -------------------------
# Image upload detection function
# -------------------------
def detect_mask_in_image(image: Image.Image):
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    results = []

    for (x, y, w, h) in faces:
        face = img_bgr[y : y + h, x : x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMG_SIZE)
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0][0]

        if pred > 0.5:
            label = "Mask"
            confidence = pred * 100
            color = (0, 255, 0)
        else:
            label = "No Mask"
            confidence = (1 - pred) * 100
            color = (255, 0, 0)

        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
        label_text = f"{label}: {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_bgr, (x, y - 35), (x + tw + 10, y), color, -1)
        cv2.putText(img_bgr, label_text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        results.append({"label": label, "confidence": float(confidence)})

    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_result, results, len(faces)

# -------------------------
# Sidebar & Mode selection
# -------------------------
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Choose Detection Mode:", ["📸 Upload Image", "📹 Live Webcam"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    """
This app uses **Deep Learning** (MobileNetV2) to detect face masks in real-time.

**Features:**
- ✅ Real-time webcam detection
- ✅ Image upload detection
- ✅ Multiple face detection
"""
)

# -------------------------
# App Logic
# -------------------------
if mode == "📸 Upload Image":
    st.subheader("📸 Image Upload Mode")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Original Image")
            st.image(image, use_column_width=True)

        with st.spinner("🔍 Detecting faces and masks..."):
            result_img, detections, face_count = detect_mask_in_image(image)

        with col2:
            st.markdown("#### 🎯 Detection Result")
            st.image(result_img, use_column_width=True)

        st.markdown("---")
        st.subheader("📊 Detection Summary")

        if face_count == 0:
            st.warning("⚠️ No faces detected. Try another image.")
        else:
            mask_count = sum(1 for d in detections if d["label"] == "Mask")
            no_mask_count = face_count - mask_count
            c1, c2, c3 = st.columns(3)
            c1.metric("👥 Total Faces", face_count)
            c2.metric("✅ With Mask", mask_count)
            c3.metric("❌ Without Mask", no_mask_count)

            st.markdown("#### 🔍 Detailed Results")
            for i, detection in enumerate(detections, 1):
                if detection["label"] == "Mask":
                    st.success(f"**Person {i}:** {detection['label']} ({detection['confidence']:.1f}% confidence)")
                else:
                    st.error(f"**Person {i}:** {detection['label']} ({detection['confidence']:.1f}% confidence)")
    else:
        st.info("👆 Upload an image to get started")
        st.markdown("### 📋 How to use:")
        st.markdown(
            """
1. Click **'Browse files'** above
2. Select an image (JPG, JPEG, or PNG)
3. View detection results with bounding boxes
4. See confidence scores for each face
"""
        )

else:  # Live Webcam
    st.subheader("📹 Real-Time Webcam Mode")
    st.write("Click **START** to begin real-time face mask detection")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    left, right = st.columns([2, 1])
    with left:
        ctx = webrtc_streamer(
            key="face-mask-detection",
            video_transformer_factory=MaskDetectorTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with right:
        st.markdown("#### 📊 Live Stats")
        if ctx.video_transformer:
            st.metric("✅ Mask Detected", ctx.video_transformer.mask_count)
            st.metric("❌ No Mask Detected", ctx.video_transformer.no_mask_count)
        else:
            st.info("Start webcam to see stats")

        st.markdown("---")
        st.markdown("#### 💡 Tips")
        st.markdown(
            """
- Ensure good lighting
- Face camera directly
- Stay within frame
- Allow camera permissions
"""
        )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with TensorFlow, OpenCV, Streamlit & WebRTC | Deep Learning Project</p>",
    unsafe_allow_html=True,
)
st.markdown("<p style='text-align:center; color:gray; font-size:12px;'>Made by <strong>Rakshit Badiger</strong></p>", unsafe_allow_html=True)
