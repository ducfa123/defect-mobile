import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model.supersimplenet import SuperSimpleNet

# Configuration
config = {
    "wandb_project": "icpr",
    "datasets_folder": "/home/mtahackathon/Documents/SuperrSimpleNet/SuperSimpleNet/datasets/MSD-US",
    "num_workers": 8,
    "setup_name": "superSimpleNet",
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "patch_size": 3,
    "noise": True,
    "perlin": True,
    "perlin_thr": 0.6,
    "no_anomaly": "empty",
    "bad": True,
    "overlap": True,
    "noise_std": 0.015,
    "image_size": (540, 960),
    "seed": 42,
    "batch": 32,
    "epochs": 300,
    "flips": False,
    "seg_lr": 0.0002,
    "dec_lr": 0.0002,
    "adapt_lr": 0.0001,
    "gamma": 0.4,
    "stop_grad": True,
    "clip_grad": False,
    "eval_step_size": 4,
}

# Initialize model
model = SuperSimpleNet(image_size=config["image_size"], config=config)

# Load weights from file
# weight_path = "results/superSimpleNet/checkpoints/cracked_screen/cracked_screen/weights.pt"
# weight_path = "results/superSimpleNet/checkpoints/mvtec/bottle/weights.pt"
weight_path = "/home/mtahackathon/Documents/SuperrSimpleNet/SuperSimpleNet/results/final_model.pth"
model.load_state_dict(
    torch.load(weight_path, map_location=torch.device("cpu")), strict=False
)
model.eval()  # Set model to inference mode

# Load and process image
image_path = "/home/mtahackathon/Documents/SuperrSimpleNet/SuperSimpleNet/datasets/MSD-US/test/stain/Sta_0005.jpg"
image = Image.open(image_path).convert("RGB")

# Keep track of original size
original_size = image.size  # original_size = (width, height)

# Define transform and apply
transform = transforms.Compose(
    [
        transforms.Resize(config["image_size"]),  # Resize theo (height, width)
        transforms.ToTensor(),
    ]
)

# transform = transforms.Compose([
#     transforms.Resize((config["image_size"][1], config["image_size"][0])),  # Resize theo (width, height)
#     transforms.ToTensor(),
# ])

image_tensor = transform(image)
input_tensor = image_tensor.unsqueeze(0)
# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Inference
with torch.no_grad():
    anomaly_map, anomaly_score = model(input_tensor)

# Get prediction results
anomaly_map = anomaly_map.squeeze()  # No need to detach if we're working on GPU tensors
anomaly_score = torch.sigmoid(anomaly_score).item()

# Ensure the anomaly map dimensions are correct (height, width)
if anomaly_map.shape != config["image_size"]:
    anomaly_map = anomaly_map.permute(1, 0)  # Swap dimensions if necessary

# Apply threshold to anomaly map to create a binary segmentation mask
threshold = 1.2  # You may need to adjust this value
segmentation_mask = (anomaly_map > threshold).type(
    torch.uint8
)  # Use torch operations instead of numpy


# Display original image, anomaly map, and segmentation result
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
print("Original image size:", original_size)  # Đây là (width, height)
print("Anomaly map shape:", anomaly_map.shape)  # Đây có thể là (height, width)

# Original image
axs[0].imshow(np.transpose(image_tensor.cpu().numpy(), (2, 1, 0)))
axs[0].set_title("Original Image")
axs[0].axis("off")

# Convert PyTorch tensor to NumPy array for visualization
anomaly_map_np = anomaly_map.cpu().numpy()
segmentation_mask_np = segmentation_mask.cpu().numpy()

# Anomaly map
axs[1].imshow(np.transpose(anomaly_map_np, (1, 0)), cmap="jet")
axs[1].set_title("Anomaly Map")
axs[1].axis("off")

# Segmentation result
axs[2].imshow(np.transpose(segmentation_mask_np, (1, 0)), cmap="gray")
axs[2].set_title("Segmentation Mask")
axs[2].axis("off")

plt.savefig("segmentation_result_fixed_gpu.png")
