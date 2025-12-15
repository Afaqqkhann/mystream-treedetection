import streamlit as st
from ultralytics import YOLO
import os
from pathlib import Path

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

    if output_files:
        output_file = output_files[0]

        if uploaded_file.type.startswith("image"):
            st.image(str(output_file), caption="Detected Image", use_column_width=True)
        else:
            st.video(str(output_file))

        with open(output_file, "rb") as f:
            st.download_button(
                "⬇ Download Result",
                f,
                file_name=output_file.name
            )
    else:
        st.error("No output generated.")
