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

class MSDUSDataModule(LightningDataModule):
    def __init__(self, data_dir, image_size, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def setup(self, stage=None):
        self.train_data = MSDDataset(Path(self.data_dir) / 'train', transform=self.transform)
        self.test_data = MSDDataset(Path(self.data_dir) / 'test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

class MSDDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for label in os.listdir(self.root):
            label_path = os.path.join(self.root, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": 1 if label != 'good' else 0}

def train_and_eval(model, datamodule, config, device):
    optimizer, scheduler = model.get_optimizers()
    model.to(device)
    train_loader = datamodule.train_dataloader()
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}") as prog_bar:
            for batch in train_loader:
                optimizer.zero_grad()
                image_batch = batch["image"].to(device)

                # best downsampling proposed by DestSeg
                mask = batch["mask"].to(device).type(torch.float32)
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(model.fh, model.fw),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )


                label = batch["label"].to(device).type(torch.float32)
                anomaly_map, score, _, _ = model.forward(image_batch)
                loss = focal_loss(torch.sigmoid(score), label)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().item()
                prog_bar.set_postfix(loss=total_loss / (batch + 1))
                prog_bar.update(1)
        scheduler.step()

    return test(model, datamodule, device, config)

@torch.no_grad()
def test(model, datamodule, device, config):
    model.eval()
    model.to(device)
    test_loader = datamodule.test_dataloader()
    results = {"score": [], "label": []}
    for batch in tqdm(test_loader, desc="Testing"):
        image_batch = batch["image"].to(device)
        anomaly_map, score = model.forward(image_batch)
        results["score"].append(torch.sigmoid(score).cpu())
        results["label"].append(batch["label"].cpu())
    return results

def run_finetune(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = "MSD-US"
    config["name"] = "finetune_MSD_US"
    results_writer = ResultsWriter(
        metrics=["AP-det", "P-AUROC", "I-AUROC"]
    )

    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    
    # Load pre-trained weights
    model.load_model(Path(config["pretrained_model_path"]))
    
    datamodule = MSDUSDataModule(
        data_dir=Path(config["datasets_folder"]) / "MSD-US",
        image_size=config["image_size"],
        batch_size=config["batch"],
        num_workers=config["num_workers"],
    )
    datamodule.setup()

    results = train_and_eval(model, datamodule, config, device)
    results_writer.add_result(category="MSD-US", last=results)
    results_writer.save(Path(config["results_save_path"]) / config["setup_name"] / config["dataset"])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "datasets_folder": "./datasets",
        "num_workers": 8,
        "setup_name": "superSimpleNet",
        "image_size": (540, 960),
        "batch": 32,
        "epochs": 50,
        "results_save_path": "./results",
        "pretrained_model_path": "./results/superSimpleNet/checkpoints/mvtec/wood/weights.pt",  # Specify the path to MVTec weights here
        "adapt_lr": 0.0001,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "gamma": 0.4,
        "noise": True,
    }
    run_finetune(device, config)

if __name__ == "__main__":
        main()