import streamlit as st
from PIL import Image
import sys
import numpy as np

# Thêm đường dẫn để import pipeline
sys.path.append('/home/mtahackathon/Documents/SuperSimpleNet')

from yolo_screen_detect.pipeline import YOLOPipeline
from pipeline import DetectionPipeline

# Đường dẫn đến mô hình và cấu hình của bạn
screen_model_path = 'yolo_screen_detect/runs/detect/train12/weights/best.pt'
defect_model_path = "results/superSimpleNet/checkpoints/cracked_screen/cracked_screen/weights.pt"
defect_model_config = {
    "image_size": (540, 960),
    # Các cấu hình khác của mô hình defect
}

# Khởi tạo các pipeline
yolo_pipeline = YOLOPipeline(model_path=screen_model_path, conf_threshold=0.25)
defect_pipeline = DetectionPipeline(defect_model_path=defect_model_path, defect_model_config=defect_model_config)

# Giao diện Streamlit
st.title("Phone Production Line Simulation - Single Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Screen Detection with YOLO Pipeline
    st.subheader("Screen Detection Results")
    cropped_phone_image = yolo_pipeline.extract_phone_region(image)
    
    if cropped_phone_image:
        st.image(cropped_phone_image, caption="Cropped Phone Region", use_container_width=True)
        
        # Defect Detection with SuperSimpleNet Pipeline
        st.subheader("Anomaly Defect Detection Results")
        anomaly_map, anomaly_score, segmentation_mask = defect_pipeline.detect_defect(cropped_phone_image)
        st.write("Anomaly Score:", anomaly_score)
        
        # Chuẩn hóa `anomaly_map` để nằm trong khoảng [0, 1]
        anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))

        # Display anomaly map and segmentation mask
        st.subheader("Anomaly Map")
        st.image(anomaly_map, caption="Anomaly Map", use_container_width=True)

        st.subheader("Segmentation Mask")
        st.image(segmentation_mask * 255, caption="Segmentation Mask", use_container_width=True)  # Nhân với 255 để hiển thị đúng
    else:
        st.write("No phone detected in the image.")
