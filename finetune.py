import os
import copy
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, seed_everything
from torchmetrics import AveragePrecision
from anomalib.utils.metrics import AUROC, AUPRO
from model.supersimplenet import SuperSimpleNet
from common.results_writer import ResultsWriter
from common.loss import focal_loss

from torchviz import make_dot


class MSDUSDataModule(LightningDataModule):
    def __init__(self, data_dir, image_size, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.Resize(self.image_size), transforms.ToTensor()]
        )

    def setup(self, stage=None):
        self.train_data = MSDDataset(
            Path(self.data_dir) / "train",
            ground_truth_dir=Path(self.data_dir) / "ground_truth",
            transform=self.transform,
        )
        self.test_data = MSDDataset(
            Path(self.data_dir) / "test",
            ground_truth_dir=Path(self.data_dir) / "ground_truth",
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )


class MSDDataset(Dataset):
    def __init__(self, root, ground_truth_dir=None, transform=None):
        self.root = root
        self.ground_truth_dir = ground_truth_dir  # Thư mục ground_truth để tìm mask
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for label in os.listdir(self.root):
            label_path = os.path.join(self.root, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    # Xác định mask cho mỗi ảnh
                    mask_name = Path(img_name).stem + ".png"
                    mask_path = os.path.join(self.ground_truth_dir, mask_name)
                    # Gán nhãn 1 nếu có mask, 0 nếu không có mask
                    label_num = 1 if os.path.exists(mask_path) else 0
                    samples.append(
                        (img_path, label_num, mask_path if label_num == 1 else None)
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Tạo mask nếu có, nếu không thì mask rỗng
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            if self.transform:
                mask = self.transform(mask)
            mask = (mask > 0).float()  # Chuyển đổi mask sang nhị phân 0 và 1
            mask = torch.clamp(
                mask, min=0, max=1
            )  # Đảm bảo mask chỉ có các giá trị trong phạm vi nhị phân

        else:
            mask = torch.zeros_like(img[:1])  # Mask rỗng nếu không có mask

        return {"image": img, "label": label, "mask": mask}


def train_and_eval(model, datamodule, config, device):
    optimizer, scheduler = model.get_optimizers()
    model.to(device)
    train_loader = datamodule.train_dataloader()
    best_val_score = 0  # Track the best validation score for checkpointing

    # Ensure results_save_path is a Path object
    results_save_path = Path(config["results_save_path"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}"
        ) as prog_bar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                image_batch = batch["image"].to(device)
                label = torch.tensor(
                    [int(lbl) for lbl in batch["label"]], device=device
                ).type(torch.float32)
                # Forward pass
                anomaly_map, score, _, _ = model.forward(image_batch)

                # Reshape score to match the shape of label
                if score.shape[0] == 2 * label.shape[0]:
                    score = score.view(label.shape[0], 2).mean(dim=1)

                if score.shape != label.shape:
                    score = score.view(label.shape)

                # Compute loss
                loss = focal_loss(torch.sigmoid(score), label)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Track total loss
                total_loss += loss.detach().cpu().item()
                prog_bar.set_postfix(loss=total_loss / (i + 1))
                prog_bar.update(1)

        # Step the scheduler
        scheduler.step()

    # Run validation test after training is complete
    val_results = test(model, datamodule, device, config)
    val_score = val_results["AP-det"] if "AP-det" in val_results else 0
    print(f"Final Validation AP-det: {val_score}")

    # Save the final model at the end of training
    torch.save(model.state_dict(), results_save_path / "final_model.pth")
    print("Final model saved.")

    return val_results  # Return the final validation results


@torch.no_grad()
def test(model, datamodule, device, config):
    model.eval()
    model.to(device)
    test_loader = datamodule.test_dataloader()

    # Initialize metrics (Bỏ qua AUPRO)
    image_metrics = {"I-AUROC": AUROC(), "AP-det": AveragePrecision(num_classes=1)}
    pixel_metrics = {"P-AUROC": AUROC()}  # Pixel-level AUROC

    # Storage for results
    results = {"score": [], "label": [], "anomaly_map": [], "gt_mask": []}

    for batch in tqdm(test_loader, desc="Testing"):
        image_batch = batch["image"].to(device)
        label = batch["label"].to(device)

        # Call forward and unpack only two outputs during evaluation
        anomaly_map, score = model.forward(image_batch)
        scores = torch.sigmoid(score).cpu()

        print("Scores:", scores)
        print("Labels:", label.cpu())

        # Save results for image-level metrics
        results["score"].append(scores)
        results["label"].append(label.cpu())

        # If masks are available in the test set, add them to the results
        if "mask" in batch:
            mask = batch["mask"].to(device)
            results["anomaly_map"].append(anomaly_map.cpu())
            results["gt_mask"].append(mask.cpu())

    # Concatenate results for metric calculation
    results["score"] = torch.cat(results["score"])
    results["label"] = torch.cat(results["label"])

    # Convert lists to tensors if they contain elements
    if results["anomaly_map"]:
        results["anomaly_map"] = torch.cat(results["anomaly_map"])
        results["gt_mask"] = torch.cat(results["gt_mask"])

        # Chuẩn hóa `gt_mask` thành nhị phân 0 và 1
        results["gt_mask"] = (results["gt_mask"] > 0).float()

    # Calculate image-level metrics
    print("Image-Level Metrics:")
    for name, metric in image_metrics.items():
        metric.update(results["score"], results["label"])
        print(f"{name}: {metric.compute().item()}")

    # Calculate pixel-level metrics if `anomaly_map` and `gt_mask` are available
    if "anomaly_map" in results and len(results["anomaly_map"]) > 0:
        valid_masks = (
            results["gt_mask"].sum(dim=(1, 2, 3)) > 0
        )  # Filter only non-zero masks
        if valid_masks.sum() > 0:  # Ensure there's at least one valid mask
            print("Pixel-Level Metrics:")
            for name, metric in pixel_metrics.items():
                metric.update(
                    results["anomaly_map"][valid_masks],
                    results["gt_mask"][valid_masks].float(),
                )
                print(f"{name}: {metric.compute().item()}")
        else:
            print("Pixel-Level Metrics: No valid masks available for calculation.")

    return results


def run_train(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = "MSD-US"
    config["name"] = "train_MSD_US"
    results_writer = ResultsWriter(metrics=["AP-det", "P-AUROC", "I-AUROC"])

    model = SuperSimpleNet(image_size=config["image_size"], config=config)

    datamodule = MSDUSDataModule(
        data_dir=Path(config["datasets_folder"]) / "MSD-US",
        image_size=config["image_size"],
        batch_size=config["batch"],
        num_workers=config["num_workers"],
    )
    datamodule.setup()

    results = train_and_eval(model, datamodule, config, device)
    results_writer.add_result(category="MSD-US", last=results)
    results_writer.save(
        Path(config["results_save_path"]) / config["setup_name"] / config["dataset"]
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "datasets_folder": "./datasets",
        "num_workers": 8,
        "setup_name": "superSimpleNet",
        "image_size": (540, 960),
        "batch": 4,
        "epochs": 100,
        "results_save_path": "./results",
        "adapt_lr": 0.0001,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "gamma": 0.4,
        "noise": True,
        "bad": True,
        "overlap": False,
        "perlin": True,
        "perlin_thr": 0.6,
        "no_anomaly": "empty",
    }
    run_train(device, config)


if __name__ == "__main__":
    main()
