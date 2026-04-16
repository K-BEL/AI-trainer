import streamlit as st
import PIL.Image
import numpy as np
import pathlib
from pathlib import Path
from ultralytics import YOLO
import torch
import tempfile
import os

# Platform compatibility fix for loading Windows-trained models on Mac/Linux
if os.name != 'nt':
    pathlib.WindowsPath = pathlib.PosixPath

# --- UI Config ---
st.set_page_config(
    page_title="Siin Trainer - Live Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium feel
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🚀 Siin Trainer: Live Demo")
st.markdown("Upload an image to see your trained model in action.")

# --- Sidebar ---
st.sidebar.header("Model Configuration")

# Model Source Selection
model_source = st.sidebar.radio(
    "Model Source",
    ["Local Checkpoints", "Upload Model (.pt)"],
    index=0
)

selected_checkpoint = None

if model_source == "Local Checkpoints":
    # Scan for checkpoints
    run_dirs = list(Path("runs").rglob("weights/*.pt"))
    checkpoint_paths = [str(p) for p in run_dirs]
    checkpoint_paths.insert(0, "yolov8n.pt") # Preset
    
    selected_checkpoint = st.sidebar.selectbox(
        "Select Model Checkpoint",
        checkpoint_paths,
        index=min(1, len(checkpoint_paths)-1) if len(checkpoint_paths) > 1 else 0
    )
else:
    uploaded_model = st.sidebar.file_uploader(
        "Upload a YOLO model", 
        type=["pt"],
        help="Upload a trained PyTorch model (.pt) file."
    )
    if uploaded_model:
        # Create a temporary file to store the uploaded model
        # We need a stable suffix so YOLO can identify it as a pt file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(uploaded_model.getbuffer())
            selected_checkpoint = f.name
    else:
        st.sidebar.info("Please upload a .pt file.")

st.sidebar.markdown("---")
st.sidebar.subheader("Inference Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Visualization Settings")
show_labels = st.sidebar.checkbox("Show Labels", value=True)
show_conf = st.sidebar.checkbox("Show Confidence", value=True)
line_width = st.sidebar.slider("Line Width", 1, 10, 2)

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(selected_checkpoint)

# --- Class Filtering ---
if model:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Class Filtering")
    all_classes = list(model.names.values())
    selected_classes = st.sidebar.multiselect(
        "Classes to Detect", 
        all_classes, 
        default=all_classes
    )
    class_ids = [k for k, v in model.names.items() if v in selected_classes]
else:
    class_ids = None

# --- Logic ---
if model:
    tab1, tab2 = st.tabs(["📸 Image Inference", "ℹ️ Model Info"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            source = st.radio("Source", ["Upload", "Camera"])
            
            input_image = None
            if source == "Upload":
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
                if uploaded_file:
                    input_image = PIL.Image.open(uploaded_file)
            else:
                camera_image = st.camera_input("Take a photo")
                if camera_image:
                    input_image = PIL.Image.open(camera_image)
            
            if input_image:
                st.image(input_image, width='stretch')

        with col2:
            st.subheader("Predictions")
            if input_image:
                with st.spinner("Analyzing image..."):
                    results = model.predict(
                        source=input_image, 
                        conf=conf_threshold, 
                        iou=iou_threshold,
                        classes=class_ids,
                        verbose=False
                    )
                    
                    # Get plotted image with custom settings
                    res_plotted = results[0].plot(
                        labels=show_labels, 
                        conf=show_conf, 
                        line_width=line_width
                    )
                    st.image(res_plotted, caption="Detected Objects", width='stretch')
                    
                    # Statistics
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.success(f"Found {len(boxes)} objects!")
                        
                        # Show detections in a table
                        det_data = []
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = model.names[cls_id]
                            conf = float(box.conf[0].item())
                            det_data.append({"Class": cls_name, "Confidence": f"{conf:.2%}"})
                        st.table(det_data)
                    else:
                        st.info("No objects detected with current thresholds.")
            else:
                st.info("Please provide an image to start.")

    with tab2:
        st.subheader("Model Metadata")
        
        # Display name (clean up temp names)
        display_name = selected_checkpoint
        if "tmp" in display_name:
            display_name = "Uploaded Model (Temporary)"
            
        st.json({
            "Checkpoint": display_name,
            "Classes": model.names,
            "Total Classes": len(model.names),
            "Task": model.task
        })

else:
    st.warning("Please select a valid model checkpoint to begin.")

st.markdown("---")
st.caption("Built with ❤️ by Siin Lab using Ultralytics & Streamlit.")
