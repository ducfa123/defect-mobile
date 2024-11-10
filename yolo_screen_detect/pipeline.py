from ultralytics import YOLOv10  # Hoặc YOLOv8, tùy phiên bản bạn dùng
import cv2
from PIL import Image

class YOLOPipeline:
    def __init__(self, model_path, conf_threshold=0.25):
        # Load mô hình YOLO từ tệp model_path
        self.model = YOLOv10(model_path)  # Hoặc YOLOv8(model_path) nếu bạn dùng YOLOv8
        self.conf_threshold = conf_threshold

    def detect_objects(self, image):
        # Chạy phát hiện đối tượng trên ảnh
        results = self.model(image, conf=self.conf_threshold)
        return results

    def extract_phone_region(self, image):
        # Chạy phát hiện để lấy bounding box của điện thoại
        results = self.detect_objects(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Lấy tọa độ của bounding boxes

        if len(boxes) > 0:
            # Chọn bounding box đầu tiên (giả sử là điện thoại)
            x1, y1, x2, y2 = boxes[0]
            cropped_image = image.crop((x1, y1, x2, y2))
            return cropped_image
        else:
            return None  # Trả về None nếu không phát hiện được điện thoại
