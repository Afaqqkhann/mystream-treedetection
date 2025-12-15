import streamlit as st
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import shutil

st.set_page_config(page_title="Tree Detection", layout="wide")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "best.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
st.success("✅ YOLOv8 Model Loaded")

st.title("🌳 Tree Detection (Images & Videos)")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

def reencode_video(input_video, output_video):
    cap = cv2.VideoCapture(str(input_video))

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ✅ universal codec
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

if st.button("Run Detection") and uploaded_file:
    input_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    project_path = os.path.join(OUTPUT_FOLDER, "results")
    os.makedirs(project_path, exist_ok=True)

    with st.spinner("Processing..."):
        model.predict(
            source=input_path,
            conf=0.25,
            save=True,
            project=project_path,
            name="run",
            verbose=False
        )

    output_dir = Path(project_path) / "run"
    output_files = list(output_dir.glob("*"))

    if not output_files:
        st.error("No output generated.")
        st.stop()

    raw_output = output_files[0]

    # IMAGE RESULT
    if uploaded_file.type.startswith("image"):
        st.image(str(raw_output), caption="Detected Image", use_column_width=True)
        with open(raw_output, "rb") as f:
            st.download_button("⬇ Download Image", f, file_name=raw_output.name)

    # VIDEO RESULT (FIXED)
    else:
        fixed_video = raw_output.with_suffix(".mp4")

        reencode_video(raw_output, fixed_video)

        st.video(str(fixed_video))

        with open(fixed_video, "rb") as f:
            st.download_button(
                "⬇ Download Video (MP4)",
                f,
                file_name=fixed_video.name,
                mime="video/mp4"
            )
