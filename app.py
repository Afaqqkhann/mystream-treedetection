import streamlit as st
from ultralytics import YOLO
import os
from pathlib import Path

st.set_page_config(page_title="Tree Detection", layout="wide")

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "best.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
st.success("YOLOv8 Loaded (Low Memory Mode)")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if st.button("Run Detection") and uploaded_file:
    input_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Running detection..."):
        model.predict(
            source=input_path,
            conf=0.3,
            save=True,
            project=OUTPUT_FOLDER,
            name="run",
            vid_stride=2
        )

    output_dir = Path(OUTPUT_FOLDER) / "run"
    outputs = list(output_dir.glob("*"))

    if outputs:
        out = outputs[0]
        if out.suffix.lower() in [".jpg", ".png"]:
            st.image(str(out))
        else:
            st.video(str(out))

        with open(out, "rb") as f:
            st.download_button(
                "Download Result",
                f,
                file_name=out.name
            )
