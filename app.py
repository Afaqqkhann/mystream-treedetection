import streamlit as st
from ultralytics import YOLO
import os
from pathlib import Path

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Tree Detection", layout="wide")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "best.pt"  # model is inside your repo
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
st.success("✅ YOLOv8 Model Loaded from Repo")

# =====================================
# UI
# =====================================
st.title("🌳 Tree Detection with YOLOv8")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
uploaded_folder = st.text_input("Or enter path to folder of images", "")

# =====================================
# RUN DETECTION
# =====================================
if st.button("Run Detection"):
    if uploaded_file:
        image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())
        source_path = image_path
    elif uploaded_folder and os.path.exists(uploaded_folder):
        source_path = uploaded_folder
    else:
        st.error("Please upload an image or provide a valid folder path.")
        st.stop()

    project_path = os.path.join(OUTPUT_FOLDER, "tree_results")
    os.makedirs(project_path, exist_ok=True)

    with st.spinner("Running YOLOv8 detection..."):
        results = model.predict(
            source=source_path,
            conf=0.25,
            save=True,
            project=project_path,
            name="run"
        )

    st.success("✅ Detection Complete!")

    if uploaded_file:
        result_img_path = Path(project_path) / "run" / uploaded_file.name
        if result_img_path.exists():
            st.image(str(result_img_path), caption="Detected Trees", use_column_width=True)
            with open(result_img_path, "rb") as f:
                st.download_button(
                    "⬇ Download Result Image",
                    f,
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/png"
                )
        else:
            st.warning("Result image not found. Check your model output path.")
    else:
        st.write(f"Results saved in folder: {os.path.join(project_path, 'run')}")
