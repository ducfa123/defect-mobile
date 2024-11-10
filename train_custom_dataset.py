import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, seed_everything
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
from common.results_writer import ResultsWriter
from common.loss import focal_loss
from tqdm import tqdm
from torchmetrics import AveragePrecision, Metric
# from torchmetrics.classification import AUROC
from anomalib.utils.metrics import AUROC, AUPRO

LOG_WANDB = False

import copy
import json

if LOG_WANDB:
    import wandb

from datamodules import ksdd2, sensum
from datamodules.ksdd2 import KSDD2, NumSegmented
from datamodules.sensum import Sensum
from datamodules.mvtec import MVTec
from datamodules.visa import Visa

from common.visualizer import Visualizer


import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
from torchvision.transforms import GaussianBlur

from common.perlin_noise import rand_perlin_2d
from model.feature_extractor import FeatureExtractor


class SuperSimpleNet(nn.Module):
    """
    SuperSimpleNet model

    Args:
        image_size: tuple with image dims (h, w)
        config: dict with model properties
    """

    def __init__(self, image_size: tuple[int, int], config):
        super().__init__()
        self.image_size = image_size
        self.config = config
        self.feature_extractor = FeatureExtractor(
            backbone=config.get("backbone", "wide_resnet50_2"),
            layers=config.get("layers", ["layer2", "layer3"]),
            patch_size=config.get("patch_size", 3),
            image_size=image_size,
        )
        # feature channels, height and width
        fc, fh, fw = self.feature_extractor.feature_dim
        self.fh = fh
        self.fw = fw
        self.feature_adaptor = FeatureAdaptor(projection_dim=fc)
        self.discriminator = Discriminator(
            projection_dim=fc,
            hidden_dim=1024,
            feature_w=fw,
            feature_h=fh,
            config=config,
        )

        self.anomaly_generator = AnomalyGenerator(
            noise_mean=0,
            noise_std=config.get("noise_std", 0.015),
            feature_w=fw,
            feature_h=fh,
            f_dim=fc,
            config=config,
        )
        self.anomaly_map_generator = AnomalyMapGenerator(
            output_size=image_size, sigma=4
        )

    def forward(
        self,
        images: Tensor,
        mask: Tensor = None,
        label: Tensor = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # feature extraction, upscaling and neigh. aggregation
        # [B, F_dim, H, W]
        features = self.feature_extractor(images)
        adapted = self.feature_adaptor(features)

        if self.training:
            # add noise to features
            if self.config["noise"]:
                # also returns adjusted labels and masks
                final_features, mask, label = self.anomaly_generator(
                    adapted, mask, label
                )
            else:
                final_features = adapted

            anomaly_map, anomaly_score = self.discriminator(final_features)
            return anomaly_map, anomaly_score, mask, label
        else:
            anomaly_map, anomaly_score = self.discriminator(adapted)
            anomaly_map = self.anomaly_map_generator(anomaly_map)

            return anomaly_map, anomaly_score

    def get_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        seg_params, dec_params = self.discriminator.get_params()
        optim = AdamW(
            [
                {
                    "params": self.feature_adaptor.parameters(),
                    "lr": self.config["adapt_lr"],
                },
                {
                    "params": seg_params,
                    "lr": self.config["seg_lr"],
                    "weight_decay": 0.00001,
                },
                {
                    "params": dec_params,
                    "lr": self.config["dec_lr"],
                    "weight_decay": 0.00001,
                },
            ]
        )
        sched = MultiStepLR(
            optim,
            milestones=[self.config["epochs"] * 0.8, self.config["epochs"] * 0.9],
            gamma=self.config["gamma"],
        )

        return optim, sched

    def save_model(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        state_dict = self.state_dict()
        # exclude feat extractor since it's pretrained
        saving_state_dict = OrderedDict(
            {
                n: k
                for n, k in state_dict.items()
                if not n.startswith("feature_extractor")
            }
        )

        torch.save(saving_state_dict, path / "weights.pt")

    def load_model(self, path):
        print(f"Loading model: {path}")
        self.load_state_dict(torch.load(path), strict=False)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)


class FeatureAdaptor(nn.Module):
    def __init__(self, projection_dim: int):
        super().__init__()
        # linear layer equivalent
        self.projection = nn.Conv2d(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=1,
            stride=1,
        )
        self.apply(init_weights)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)


def _conv_block(in_chanels, out_chanels, kernel_size, padding="same"):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_chanels,
            out_channels=out_chanels,
            kernel_size=kernel_size,
            padding=padding,
        ),
        nn.BatchNorm2d(out_chanels),
        nn.ReLU(inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(
        self, projection_dim: int, hidden_dim: int, feature_w, feature_h, config
    ):
        super().__init__()
        self.fw = feature_w
        self.fh = feature_h
        self.stop_grad = config.get("stop_grad", False)

        # 1x1 conv - linear layer equivalent
        self.seg = nn.Sequential(
            nn.Conv2d(
                in_channels=projection_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.dec_head = _conv_block(
            in_chanels=projection_dim + 1, out_chanels=128, kernel_size=5
        )

        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc_score = nn.Linear(in_features=128 * 2 + 2, out_features=1)

        self.apply(init_weights)

    def get_params(self):
        seg_params = self.seg.parameters()
        dec_params = list(self.dec_head.parameters()) + list(self.fc_score.parameters())
        return seg_params, dec_params

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        # get anomaly map from seg head
        map = self.seg(input)

        map_dec_copy = map
        if self.stop_grad:
            map_dec_copy = map_dec_copy.detach()
        # dec conv layer takes feat + map
        mask_cat = torch.cat((input, map_dec_copy), dim=1)
        dec_out = self.dec_head(mask_cat)

        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        map_max = self.map_max_pool(map)
        if self.stop_grad:
            map_max = map_max.detach()

        map_avg = self.map_avg_pool(map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        # final dec layer: conv channel max and avg and map max and avg
        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(
            dim=(2, 3)
        )
        score = self.fc_score(dec_cat).squeeze(dim=1)

        return map, score


class AnomalyGenerator(nn.Module):
    def __init__(
        self,
        noise_mean: float,
        noise_std: float,
        feature_h: int,
        feature_w: int,
        f_dim: int,
        config: dict,
        perlin_range: tuple[int, int] = (0, 6),
    ):
        super().__init__()

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.min_perlin_scale = perlin_range[0]
        self.max_perlin_scale = perlin_range[1]

        self.height = feature_h
        self.width = feature_w
        self.f_dim = f_dim

        self.config = config

        self.perlin_height = self.next_power_2(self.height)
        self.perlin_width = self.next_power_2(self.width)

    @staticmethod
    def next_power_2(num):
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches) -> Tensor:
        """
        Generate 2d perlin noise masks with dims [b, 1, self.h, self.w]

        Args:
            batches: number of batches (different masks)

        Returns:
            tensor with b perlin binarized masks
        """
        perlin = []
        for _ in range(batches):
            perlin_scalex = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )
            perlin_scaley = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )

            perlin_noise = rand_perlin_2d(
                (self.perlin_height, self.perlin_width), (perlin_scalex, perlin_scaley)
            )
            # original is power of 2 scale, so fit to our size
            perlin_noise = F.interpolate(
                perlin_noise.reshape(1, 1, self.perlin_height, self.perlin_width),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            threshold = self.config["perlin_thr"]
            # binarize
            perlin_thr = torch.where(perlin_noise > threshold, 1, 0)

            chance_anomaly = torch.rand(1).numpy()[0]
            if chance_anomaly > 0.5:
                if self.config["no_anomaly"] == "full":
                    # entire image is anomaly
                    perlin_thr = torch.ones_like(perlin_thr)
                elif self.config["no_anomaly"] == "empty":
                    # no anomaly
                    perlin_thr = torch.zeros_like(perlin_thr)
                # if none -> don't add

            perlin.append(perlin_thr)
        return torch.cat(perlin)

    def forward(
        self, input: Tensor, mask: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = mask.shape

        # duplicate input, mask, and labels
        input = torch.cat((input, input))
        mask = torch.cat((mask, mask))
        labels = labels.repeat(2, 1)  # Sử dụng repeat để nhân đôi mà giữ nguyên kích thước hợp lý


        noise = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=input.shape,
            device=input.device,
            requires_grad=False,
        )

        # mask indicating which regions will have noise applied
        noise_mask = torch.ones(
            b * 2, 1, h, w, device=input.device, requires_grad=False
        )

        if not self.config["bad"]:
            masking_labels = labels.reshape(b * 2, 1, 1, 1)
            noise_mask = noise_mask * (1 - masking_labels)

        if not self.config["overlap"]:
            noise_mask = noise_mask * (1 - mask)

        if self.config["perlin"]:
            perlin_mask = self.generate_perlin(b * 2).to(input.device)
            noise_mask = noise_mask * perlin_mask
        else:
            noise_mask[:b, ...] = 0

        # update gt mask
        mask = mask + noise_mask
        mask = torch.where(mask > 0, 1, 0)

        # make new labels. 1 if any part of mask is 1, 0 otherwise
        new_anomalous = mask.reshape(input.size(0), -1).any(dim=1).type(torch.float32)


        # Ensure labels have the same size as new_anomalous
        if labels.dim() > 1:
            labels = labels.view(-1)

        if labels.size(0) != new_anomalous.size(0):
            labels = labels[:new_anomalous.size(0)]


        # Perform addition after ensuring the sizes match
        labels = labels + new_anomalous

        # binarize
        labels = torch.where(labels > 0, 1, 0)

        # apply masked noise
        perturbed = input + noise * noise_mask

        return perturbed, mask, labels



class AnomalyMapGenerator(nn.Module):
    def __init__(self, output_size: tuple[int, int], sigma: float):
        super().__init__()
        self.size = output_size
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=4)

    def forward(self, input: Tensor) -> Tensor:
        # upscale & smooth
        anomaly_map = F.interpolate(input, size=self.size, mode="bilinear")
        anomaly_map = self.blur(anomaly_map)
        return anomaly_map
# Class CrackedScreenDataset to read data from XML files
class CrackedScreenDataset(Dataset):
    def __init__(self, image_files, mask_files=None, transform=None, mode='train'):
        """
        Args:
            image_files (list): Danh sách các đường dẫn đến ảnh.
            mask_files (list, optional): Danh sách các đường dẫn đến mask (chỉ dùng cho tập test).
            transform (callable, optional): Các phép biến đổi áp dụng lên ảnh.
            mode (str): 'train' hoặc 'test', xác định chế độ của dataset.
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform
        self.mode = mode

        if mode not in ['train', 'test']:
            raise ValueError("Mode must be either 'train' hoặc 'test'")

        # Nếu ở chế độ test, cần có các mask để đánh giá
        if mode == 'test' and mask_files is None:
            raise ValueError("Cần cung cấp mask_files cho chế độ test")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load ảnh
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Áp dụng phép biến đổi lên ảnh nếu có
        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            # Tập train chỉ có nhãn 'good' (không có mask)
            target = {'label': 0}  # Nhãn 0 cho ảnh 'good'
        elif self.mode == 'test':
            # Load mask tương ứng
            mask_path = self.mask_files[idx]
            mask = Image.open(mask_path).convert('L')

            # Chuyển đổi mask thành tensor nếu có phép biến đổi
            if self.transform:
                mask = self.transform(mask)

            target = {'label': 1, 'mask': mask}  # Nhãn 1 cho ảnh có lỗi

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)

    # Tiền xử lý hình ảnh
    images = torch.stack([torch.tensor(image) for image in images])

    # Tiền xử lý các mask nếu có
    masks = None
    if any('mask' in target for target in targets):
        masks = torch.stack([target['mask'] if 'mask' in target else torch.zeros_like(targets[0]['mask']) for target in targets])

    # Tiền xử lý nhãn (labels)
    labels = torch.tensor([target['label'] for target in targets])

    # Trả về kết quả
    if masks is not None:
        return images, {"masks": masks, "labels": labels}
    else:
        return images, {"labels": labels}


from sklearn.model_selection import train_test_split

# Sửa lại lớp DataModule để sử dụng `collate_fn` tùy chỉnh
# Sửa lại lớp DataModule để sử dụng `collate_fn` tùy chỉnh
class CrackedScreenDataModule(LightningDataModule):
    def __init__(self, dataset_path, image_size, train_batch_size, eval_batch_size, num_workers):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        # Load the list of image files from training folder (good)
        train_folder = os.path.join(self.dataset_path, 'train', 'good')
        train_image_files = sorted([os.path.join(train_folder, f) for f in os.listdir(train_folder)])

        # Load the list of image and mask files from testing folders (oil, scratch, stain, ground_truth)
        test_image_files = []
        defect_folders = ["oil", "scratch", "stain"]
        for defect_folder in defect_folders:
            folder_path = os.path.join(self.dataset_path, 'test', defect_folder)
            image_files = sorted(os.listdir(folder_path))
            test_image_files.extend([os.path.join(folder_path, f) for f in image_files])

        # Load mask files from `ground_truth`
        mask_folder = os.path.join(self.dataset_path, 'test', 'ground_truth')
        test_mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder)])

        # Ensure that image and mask files match
        assert len(test_image_files) == len(test_mask_files), "Number of test images and masks must be the same"

        # Split the dataset into training and testing sets
        train_mask_files = [None] * len(train_image_files)  # No masks for training images

        # Create the train and test datasets
        self.train_dataset = CrackedScreenDataset(train_image_files, train_mask_files, self.transform, mode='train')
        self.test_dataset = CrackedScreenDataset(test_image_files, test_mask_files, self.transform, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=collate_fn)

def train(
    model: SuperSimpleNet,
    epochs: int,
    datamodule: CrackedScreenDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    th: float = 0.5,
    clip_grad: bool = True,
    eval_step_size: int = 4,
):
    model.to(device)
    optimizer, scheduler = model.get_optimizers()

    model.train()
    train_loader = datamodule.train_dataloader()
    results = None  # Khởi tạo results

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(
            total=len(train_loader),
            desc=str(epoch) + "/" + str(epochs),
            miniters=int(1),
            unit="batch",
        ) as prog_bar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                image_batch, target = batch
                image_batch = image_batch.to(device)

                # Extract the mask from the target if available
                if 'mask' in target:
                    mask = target['mask'].to(device).type(torch.float32)

                    # Đảm bảo rằng `mask` có 4 chiều (N, C, H, W)
                    if mask.dim() == 3:  # Nếu mask có dạng (N, H, W)
                        mask = mask.unsqueeze(1)  # Thêm chiều kênh để có dạng (N, 1, H, W)
                else:
                    # Nếu không có mask (tức là tập train), tạo mask bằng 0
                    mask = torch.zeros((image_batch.size(0), 1, model.fh, model.fw), device=device)

                # Resize mask
                mask = F.interpolate(
                    mask,
                    size=(model.fh, model.fw),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )

                label = target['labels'].to(device).type(torch.float32)

                # Forward pass
                try:
                    anomaly_map, score, mask, label = model.forward(
                        image_batch, mask, label
                    )

                    # Adjust label dimensions if necessary
                    if label.dim() > 1 and label.size(1) != 1:
                        print(f"Warning: Adjusting label size from {label.size()} to {(label.size(0), 1)}")
                        label = label[:, :1]  # Adjust to match expected shape

                except RuntimeError as e:
                    # In ra thông tin nếu gặp lỗi
                    print(f"RuntimeError during forward pass: {str(e)}")
                    print(f"Mask size: {mask.size()}")
                    print(f"Label size: {label.size()}")
                    raise e

                # Adjusted truncated L1 loss
                normal_scores = anomaly_map[mask == 0]
                anomalous_scores = anomaly_map[mask > 0]
                true_loss = torch.clip(normal_scores + th, min=0)
                fake_loss = torch.clip(-anomalous_scores + th, min=0)

                if len(true_loss):
                    true_loss = true_loss.mean()
                else:
                    true_loss = torch.tensor(0.0, device=device)

                if len(fake_loss):
                    fake_loss = fake_loss.mean()
                else:
                    fake_loss = torch.tensor(0.0, device=device)

                # Resize mask to match anomaly_map size
                mask_resized = F.interpolate(mask, size=anomaly_map.shape[-2:], mode="bilinear", align_corners=False)
                mask_resized = mask_resized.view(-1)

                # Flatten anomaly_map and mask for focal loss computation
                anomaly_map_flat = anomaly_map.view(-1)
                mask_flat = mask_resized.view(-1)

                # Flatten score and label for focal loss computation
                score_flat = score.view(-1)
                label_flat = label.view(-1)

                # Ensure score and label have the same size
                min_length = min(score_flat.size(0), label_flat.size(0))
                score_flat = score_flat[:min_length]
                label_flat = label_flat[:min_length]

                # Compute focal loss
                loss = (
                    true_loss
                    + fake_loss
                    + focal_loss(torch.sigmoid(anomaly_map_flat), mask_flat)
                    + focal_loss(torch.sigmoid(score_flat), label_flat)
                )

                loss.backward()
                if clip_grad:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                else:
                    norm = None

                optimizer.step()

                total_loss += loss.detach().cpu().item()

                output = {
                    "batch_loss": np.round(loss.data.cpu().detach().numpy(), 5),
                    "avg_loss": np.round(total_loss / (i + 1), 5),
                    "norm": norm,
                }

                prog_bar.set_postfix(**output)
                prog_bar.update(1)

            if (epoch + 1) % eval_step_size == 0:
                results = test(
                    model=model,
                    datamodule=datamodule,
                    device=device,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                    normalize=True,
                )
                if LOG_WANDB:
                    wandb.log({**results, **output})
            else:
                if LOG_WANDB:
                    wandb.log(output)
        scheduler.step()

    return results

@torch.no_grad()
def test(
    model: SuperSimpleNet,
    datamodule: CrackedScreenDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    normalize: bool = True,
    image_save_path: Path = None,
    score_save_path: Path = None,
):
    model.to(device)
    model.eval()

    # for anomaly map max as image score
    seg_image_metrics = {}

    for m_name, metric in image_metrics.items():
        metric.cpu()
        metric.reset()

        seg_image_metrics[f"seg-{m_name}"] = copy.deepcopy(metric)

    for metric in pixel_metrics.values():
        metric.cpu()
        metric.reset()

    test_loader = datamodule.test_dataloader()
    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
    }

    # Kiểm tra sự đa dạng về nhãn trong tập kiểm tra
    all_labels = []

    for batch in tqdm(test_loader, position=0, leave=True):
        if len(test_loader) == 0:
            print("Warning: No data available in test dataloader.")
            return results

        # Kiểm tra kiểu của `batch` và truy cập phù hợp
        if isinstance(batch, dict):
            # Nếu `batch` là từ điển, truy cập như cũ
            image_batch = batch["image"].to(device)
            mask_batch = batch["mask"]
            label_batch = batch["label"]
            image_path_batch = batch["image_path"]
            mask_path_batch = batch["mask_path"]
        elif isinstance(batch, tuple) and len(batch) == 2:
            # Nếu `batch` là tuple với 2 phần tử, giả sử nó là (images, targets)
            image_batch, targets = batch
            image_batch = image_batch.to(device)

            # Truy cập vào các thông tin khác từ targets
            if "mask" in targets:
                mask_batch = targets["mask"]
                label_batch = targets["labels"]
                image_path_batch = targets.get("image_path", [""] * len(label_batch))  # Thêm giá trị mặc định nếu không có
                mask_path_batch = targets.get("mask_path", [""] * len(label_batch))  # Thêm giá trị mặc định nếu không có
            else:
                # Trường hợp không có mask
                mask_batch = torch.zeros((image_batch.size(0), 1, model.fh, model.fw), device=device)
                label_batch = targets["labels"]
                image_path_batch = [""] * len(label_batch)  # Thêm giá trị mặc định nếu không có
                mask_path_batch = [""] * len(label_batch)  # Thêm giá trị mặc định nếu không có
        else:
            raise TypeError("Unexpected batch type or number of elements: {}".format(type(batch)))

        # Kiểm tra dữ liệu trống hoặc không hợp lệ trong batch
        if len(image_batch) == 0:
            print("Warning: Empty batch, skipping.")
            continue

        # Xác nhận lại logic của nhãn dương và âm trong batch
        unique_labels = torch.unique(label_batch).tolist()
        print(f"Unique labels in current batch: {unique_labels}")
        all_labels.extend(unique_labels)

        # Kiểm tra hình ảnh và mask có phù hợp không
        print(f"Image batch shape: {image_batch.shape}")
        print(f"Mask batch shape: {mask_batch.shape}")

        # Thực hiện forward pass
        anomaly_map, anomaly_score = model.forward(image_batch)

        # Lấy kết quả và chuyển về CPU
        anomaly_map = anomaly_map.detach().cpu()
        anomaly_score = anomaly_score.detach().cpu()

        # Padding gt_mask để đảm bảo tất cả các mask có cùng kích thước
        padded_masks = []
        max_height = max(mask.shape[-2] for mask in mask_batch)
        max_width = max(mask.shape[-1] for mask in mask_batch)

        for mask in mask_batch:
            # Nếu mask chỉ có 2 chiều (H, W), thêm 1 chiều thành (1, H, W)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            
            # Pad mask để tất cả các mask có cùng kích thước
            pad_height = max_height - mask.shape[-2]
            pad_width = max_width - mask.shape[-1]
            padded_mask = F.pad(mask, (0, pad_width, 0, pad_height))
            padded_masks.append(padded_mask)

        # Ghép các mask đã pad thành một tensor duy nhất
        gt_mask = torch.stack(padded_masks)

        # Cập nhật kết quả
        results["anomaly_map"].append(anomaly_map)
        results["gt_mask"].append(gt_mask)
        results["score"].append(torch.sigmoid(anomaly_score))
        results["seg_score"].append(anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values)
        results["label"].append(label_batch.detach().cpu())
        results["image_path"].extend(image_path_batch)
        results["mask_path"].extend(mask_path_batch)

    # Kết hợp kết quả từ các batch
    results["anomaly_map"] = torch.cat(results["anomaly_map"])
    results["score"] = torch.cat(results["score"])
    results["seg_score"] = torch.cat(results["seg_score"])
    results["gt_mask"] = torch.cat(results["gt_mask"])
    results["label"] = torch.cat(results["label"])

    # Kiểm tra sự đa dạng về nhãn trong toàn bộ tập kiểm tra
    unique_labels_in_test_set = set(all_labels)
    print(f"Unique labels in test set: {unique_labels_in_test_set}")

    # Normalization
    if normalize:
        results["anomaly_map"] = (
            results["anomaly_map"] - results["anomaly_map"].flatten().min()
        ) / (
            results["anomaly_map"].flatten().max() - results["anomaly_map"].flatten().min()
        )
        results["score"] = (results["score"] - results["score"].min()) / (
            results["score"].max() - results["score"].min()
        )
        results["seg_score"] = (results["seg_score"] - results["seg_score"].min()) / (
            results["seg_score"].max() - results["seg_score"].min()
        )

    # Tính toán metrics cho kết quả ảnh
    results_dict = {}
    for name, metric in image_metrics.items():
        if metric.preds is not None and metric.target is not None and len(metric.preds) > 0:
            metric_result = metric.to(device).compute()
            if isinstance(metric_result, tuple):
                # Nếu kết quả là tuple, xử lý từng phần tử của nó
                for idx, value in enumerate(metric_result):
                    results_dict[f"{name}_{idx}"] = value.item()
            else:
                results_dict[name] = metric_result.item()
        else:
            print(f"Warning: No samples for metric '{name}' to compute.")
        metric.to("cpu")

    for name, metric in seg_image_metrics.items():
        if metric.preds is not None and metric.target is not None and len(metric.preds) > 0:
            metric_result = metric.to(device).compute()
            if isinstance(metric_result, tuple):
                # Nếu kết quả là tuple, xử lý từng phần tử của nó
                for idx, value in enumerate(metric_result):
                    results_dict[f"{name}_{idx}"] = value.item()
            else:
                results_dict[name] = metric_result.item()
        else:
            print(f"Warning: No samples for metric '{name}' to compute.")
        metric.to("cpu")

    # Tính toán metrics cho kết quả pixel
    for name, metric in pixel_metrics.items():
        try:
            # Tránh nan ở các giai đoạn sớm
            am = results["anomaly_map"]
            am[am != am] = 0
            results["anomaly_map"] = am

            if metric.preds is not None and metric.target is not None and len(metric.preds) > 0:
                metric_result = metric.to(device).compute()
                if isinstance(metric_result, tuple):
                    # Nếu kết quả là tuple, xử lý từng phần tử của nó
                    for idx, value in enumerate(metric_result):
                        results_dict[f"{name}_{idx}"] = value.item()
                else:
                    results_dict[name] = metric_result.item()
            else:
                print(f"Warning: No samples for metric '{name}' to compute.")
        except RuntimeError:
            # Trường hợp xảy ra lỗi với AUPRO, bỏ qua và gán giá trị 0
            results_dict[name] = 0
        metric.to("cpu")

    for name, value in results_dict.items():
        print(f"{name}: {value} ", end="")
    print()


    # Lưu kết quả điểm số nếu cần
    score_dict = {}
    if score_save_path:
        # Lưu cả seg_score và score vào json
        for img_path, score, seg_score, label in zip(
            results["image_path"],
            results["score"],
            results["seg_score"],
            results["label"],
        ):
            img_path = Path(img_path)

            anomaly_type = img_path.parent.name
            if anomaly_type not in score_dict:
                score_dict[anomaly_type] = {"good": {}, "bad": {}}

            # Phân loại kết quả thành "good" hoặc "bad"
            kind = "bad" if label == 1 else "good"

            score_dict[anomaly_type][kind][img_path.stem] = {
                "score": score.item(),
                "seg_score": seg_score.item(),
            }

        score_save_path.mkdir(exist_ok=True, parents=True)
        with open(score_save_path / "scores.json", "w") as f:
            json.dump(score_dict, f)

    return results_dict

def train_and_eval(model, datamodule, config, device):
    if LOG_WANDB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(project=config["wandb_project"], config=config, name=config["name"])

    image_metrics = {
        "I-AUROC": AUROC(task="binary"),
        "AP-det": AveragePrecision(num_classes=1,task="binary"),
    }
    pixel_metrics = {
        "P-AUROC": AUROC(task="binary"),
        "AUPRO": AUPRO(),
        "AP-loc": AveragePrecision(num_classes=1,task="binary"),
    }

    train(
        model=model,
        epochs=config["epochs"],
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        clip_grad=config["clip_grad"],
        eval_step_size=config["eval_step_size"],
    )
    if LOG_WANDB:
        wandb.finish()

    try:
        model.save_model(
            Path(config["results_save_path"])
            / config["setup_name"]
            / "checkpoints"
            / config["dataset"]
            / config["category"],
        )
    except Exception as e:
        print("Error saving checkpoint" + str(e))

    results = test(
        model=model,
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        normalize=True,
        image_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "visual"
        / config["dataset"]
        / config["category"],
        score_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "scores"
        / config["dataset"]
        / config["category"],
    )

    return results


def main_cracked_screen(device, config):
    config = config.copy()
    config["dataset"] = "cracked_screen"
    config["category"] = "cracked_screen"
    config["name"] = f"cracked_screen_{config['setup_name']}"

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
        ]
    )

    # Deterministic
    seed_everything(config["seed"], workers=True)
    with torch.backends.cudnn.flags(deterministic=True, benchmark=False):
        model = SuperSimpleNet(image_size=config["image_size"], config=config)

        datamodule = CrackedScreenDataModule(
            dataset_path=config["datasets_folder"],
            image_size=config["image_size"],
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
        )


        datamodule.setup()

        # Check if data is loaded properly
        print(f"Number of training samples: {len(datamodule.train_dataset)}")
        print(f"Number of test samples: {len(datamodule.test_dataset)}")

        # Check sample images and annotations
        sample_image, sample_target = datamodule.train_dataset[0]
        print(f"Sample image size: {sample_image.size()}")
        print(f"Sample target: {sample_target}")

        print("Starting training...")
        results = train_and_eval(
            model=model, datamodule=datamodule, config=config, device=device
        )
        print("results",results)
        if results is not None:
            results_writer.add_result(
                category="cracked_screen",
                last=results,
            )
            results_writer.save(
                Path(config["results_save_path"]) / config["setup_name"] / config["dataset"]
            )
        print("Training completed.")

# Example call to run the training process
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "overlap": True,  # makes no difference, just faster if false to avoid computation
        "noise_std": 0.015,
        "image_size": (540, 960),
        "seed": 42,
        "batch": 2,
        "epochs": 300,
        "flips": False,  # makes no difference, just faster if false to avoid computation
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": True,
        "clip_grad": False,
        "eval_step_size": 4,
        "results_save_path": Path("./results"),
    }

    print("Initializing training...")
    main_cracked_screen(device=device, config=config)
    print("All processes completed.")

if __name__ == "__main__":
    main()