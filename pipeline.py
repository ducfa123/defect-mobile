import torch
from PIL import Image
from torchvision import transforms
from model.supersimplenet import SuperSimpleNet

class DetectionPipeline:
    def __init__(self, defect_model_path, defect_model_config):
        # Load SuperSimpleNet cho phát hiện defect
        self.defect_model = SuperSimpleNet(image_size=defect_model_config["image_size"], config=defect_model_config)
        self.defect_model.load_state_dict(torch.load(defect_model_path, map_location=torch.device('cpu')), strict=False)
        self.defect_model.eval()  # Set to inference mode

        # Configuration for defect model
        self.defect_model_config = defect_model_config

    def detect_defect(self, cropped_image, threshold=1.2):
        # Load and preprocess image cho phát hiện defect
        transform = transforms.Compose([
            transforms.Resize(self.defect_model_config["image_size"]),
            transforms.ToTensor(),
        ])
        image_tensor = transform(cropped_image).unsqueeze(0)

        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.defect_model.to(device)
        input_tensor = image_tensor.to(device)

        # Run inference
        with torch.no_grad():
            anomaly_map, anomaly_score = self.defect_model(input_tensor)

        # Process results
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        anomaly_score = torch.sigmoid(anomaly_score).item()
        segmentation_mask = (anomaly_map > threshold).astype(int)

        return anomaly_map, anomaly_score, segmentation_mask
