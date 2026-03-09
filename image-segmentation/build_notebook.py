#!/usr/bin/env python3
"""import json
import os

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def md(source):"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from build_utils import md, code, write_notebook


def L(*lines):
    """Build an nbformat source list: all lines except the last get a trailing newline."""
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


cells = []

# ── 1. Title Banner ───────────────────────────────────────────────────────
cells.append(md(L(
    "# Image Segmentation Masterclass: U-Net+",
    "",
    "![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)"
    " ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)"
    " ![License](https://img.shields.io/badge/License-MIT-green.svg)"
    " ![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)",
    "",
    "---"
)))

# ── 2. TL;DR ──────────────────────────────────────────────────────────────
cells.append(md(L(
    "## TL;DR",
    "",
    "This notebook is a **complete walkthrough** of image segmentation for Kaggle competitions.",
    "We build a **U-Net from scratch**, explore **U-Net++**, **DeepLabV3+**, and **SegFormer**,",
    "implement five loss functions, train with mixed-precision & gradient accumulation,",
    "and cover competition-winning post-processing tricks.",
    "",
    "> **Key Takeaway:** Mastering segmentation is less about picking the right architecture",
    "> and more about *smart data augmentation*, *loss function design*, and *post-processing*."
)))

# ── 3. Table of Contents ──────────────────────────────────────────────────
cells.append(md(L(
    "## Table of Contents",
    "",
    "1. [Setup & Imports](#setup)",
    "2. [Segmentation Taxonomy](#taxonomy)",
    "3. [Data Preparation](#data-prep)",
    "4. [U-Net from Scratch](#unet)",
    "5. [Loss Functions](#losses)",
    "6. [Training Pipeline](#training)",
    "7. [Advanced Architectures](#advanced)",
    "8. [Post-Processing](#postproc)",
    "9. [Evaluation & Visualization](#eval)",
    "10. [Competition Tips](#tips)",
    "11. [Further Reading](#reading)",
    "",
    "---"
)))

# ── 4. Setup & Imports ────────────────────────────────────────────────────
cells.append(md(L(
    '<a id="setup"></a>',
    "## 1. Setup & Imports"
)))

cells.append(code(L(
    "# ── Install dependencies (Kaggle-friendly) ──────────────────────────",
    "import subprocess, sys",
    "",
    "def _pip(*pkgs):",
    "    for p in pkgs:",
    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])",
    "",
    "_pip('segmentation-models-pytorch', 'albumentations>=1.3', 'opencv-python-headless')"
)))

cells.append(code(L(
    "import json, os, math, random, warnings",
    "from pathlib import Path",
    "",
    "import numpy as np",
    "import cv2",
    "import matplotlib.pyplot as plt",
    "import matplotlib.patches as mpatches",
    "",
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "from torch.utils.data import Dataset, DataLoader",
    "from torch.cuda.amp import autocast, GradScaler",
    "",
    "import torchvision.transforms.functional as TF",
    "import albumentations as A",
    "from albumentations.pytorch import ToTensorV2",
    "",
    "import segmentation_models_pytorch as smp",
    "",
    "warnings.filterwarnings('ignore')",
    "plt.rcParams['figure.figsize'] = (12, 6)",
    "",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print(f'Device: {DEVICE}')",
    "print(f'PyTorch: {torch.__version__}')"
)))

# ── 5. Segmentation Taxonomy ──────────────────────────────────────────────
cells.append(md(L(
    '<a id="taxonomy"></a>',
    "## 2. Segmentation Taxonomy",
    "",
    "There are **three** major flavours of image segmentation:",
    "",
    "| Type | Goal | Output |",
    "|------|------|--------|",
    "| **Semantic** | Label every pixel with a class | Class map (H x W) |",
    "| **Instance** | Separate individual objects | Class + Instance ID |",
    "| **Panoptic** | Semantic + Instance combined | Unified map |",
    "",
    "```",
    "Semantic              Instance              Panoptic",
    "+-----------+         +-----------+         +-----------+",
    "| A A B B B |         | 1 1 2 2 2 |         |A1 A1 B1 B1|",
    "| A A B B B |         | 1 1 2 2 2 |         |A1 A1 B1 B1|",
    "| C C C C C |         | 3 3 3 3 3 |         |C1 C1 C1 C1|",
    "+-----------+         +-----------+         +-----------+",
    "```",
    "",
    "> **Key Takeaway:** Most Kaggle competitions (Vesuvius Challenge, SenNet + HOA, HuBMAP)",
    "> use **binary or multi-class semantic segmentation**. That is our focus."
)))

# ── 6. Data Preparation ───────────────────────────────────────────────────
cells.append(md(L(
    '<a id="data-prep"></a>',
    "## 3. Data Preparation",
    "",
    "We generate a **synthetic dataset** of images with circles and rectangles",
    "so this notebook is fully self-contained -- no external downloads required."
)))

cells.append(code(L(
    "def generate_synthetic_sample(size=256, num_shapes=5, seed=None):",
    "    '''Create a synthetic RGB image and its binary segmentation mask.'''",
    "    if seed is not None:",
    "        random.seed(seed)",
    "        np.random.seed(seed)",
    "",
    "    img = np.zeros((size, size, 3), dtype=np.uint8)",
    "    mask = np.zeros((size, size), dtype=np.uint8)",
    "",
    "    # background gradient",
    "    for y in range(size):",
    "        img[y, :, 0] = int(30 + 40 * y / size)",
    "        img[y, :, 1] = int(20 + 30 * y / size)",
    "        img[y, :, 2] = int(50 + 50 * y / size)",
    "",
    "    for _ in range(num_shapes):",
    "        shape_type = random.choice(['circle', 'rectangle'])",
    "        color = tuple(random.randint(80, 255) for _ in range(3))",
    "",
    "        if shape_type == 'circle':",
    "            cx = random.randint(30, size - 30)",
    "            cy = random.randint(30, size - 30)",
    "            r = random.randint(15, 50)",
    "            cv2.circle(img, (cx, cy), r, color, -1)",
    "            cv2.circle(mask, (cx, cy), r, 1, -1)",
    "        else:",
    "            x1 = random.randint(10, size - 60)",
    "            y1 = random.randint(10, size - 60)",
    "            x2 = x1 + random.randint(20, 80)",
    "            y2 = y1 + random.randint(20, 80)",
    "            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)",
    "            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)",
    "",
    "    return img, mask",
    "",
    "",
    "# Quick preview",
    "fig, axes = plt.subplots(2, 4, figsize=(14, 7))",
    "for i in range(4):",
    "    img, msk = generate_synthetic_sample(seed=i)",
    "    axes[0, i].imshow(img)",
    "    axes[0, i].set_title(f'Image {i}')",
    "    axes[0, i].axis('off')",
    "    axes[1, i].imshow(msk, cmap='gray')",
    "    axes[1, i].set_title(f'Mask {i}')",
    "    axes[1, i].axis('off')",
    "plt.suptitle('Synthetic Segmentation Data', fontsize=14, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

cells.append(md(L(
    "### Custom PyTorch Dataset"
)))

cells.append(code(L(
    "class SyntheticSegDataset(Dataset):",
    "    '''On-the-fly synthetic segmentation dataset.'''",
    "",
    "    def __init__(self, length=200, size=256, transform=None):",
    "        self.length = length",
    "        self.size = size",
    "        self.transform = transform",
    "",
    "    def __len__(self):",
    "        return self.length",
    "",
    "    def __getitem__(self, idx):",
    "        img, mask = generate_synthetic_sample(self.size, seed=idx)",
    "        if self.transform:",
    "            augmented = self.transform(image=img, mask=mask)",
    "            img = augmented['image']          # (C, H, W) float tensor",
    "            mask = augmented['mask']           # (H, W) uint8",
    "        else:",
    "            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0",
    "            mask = torch.from_numpy(mask)",
    "        return img, mask.float().unsqueeze(0)  # (1, H, W)"
)))

cells.append(md(L(
    "### Albumentations Pipeline"
)))

cells.append(code(L(
    "train_transform = A.Compose([",
    "    A.HorizontalFlip(p=0.5),",
    "    A.VerticalFlip(p=0.5),",
    "    A.RandomRotate90(p=0.5),",
    "    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),",
    "    A.OneOf([",
    "        A.GaussNoise(var_limit=(10, 50)),",
    "        A.GaussianBlur(blur_limit=3),",
    "    ], p=0.3),",
    "    A.RandomBrightnessContrast(p=0.3),",
    "    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),",
    "    ToTensorV2(),",
    "])",
    "",
    "val_transform = A.Compose([",
    "    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),",
    "    ToTensorV2(),",
    "])",
    "",
    "train_ds = SyntheticSegDataset(length=400, transform=train_transform)",
    "val_ds   = SyntheticSegDataset(length=100, transform=val_transform)",
    "",
    "train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0, pin_memory=True)",
    "val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0, pin_memory=True)",
    "",
    "imgs, masks = next(iter(train_loader))",
    "print(f'Image batch : {imgs.shape}  dtype={imgs.dtype}')",
    "print(f'Mask batch  : {masks.shape}  dtype={masks.dtype}')"
)))

# ── 7. U-Net from Scratch ─────────────────────────────────────────────────
cells.append(md(L(
    '<a id="unet"></a>',
    "## 4. U-Net from Scratch",
    "",
    "The U-Net architecture consists of:",
    "- **Encoder** (contracting path): Repeated 3x3 conv blocks + 2x2 max-pool",
    "- **Bottleneck**: Deepest feature representation",
    "- **Decoder** (expanding path): 2x2 up-conv + skip-connection concatenation + 3x3 conv blocks",
    "",
    "```",
    "Input (3,256,256)",
    "  |",
    "  v",
    "[Enc1] 64 --> skip1 ----+",
    "  |                      |",
    "[Enc2] 128 -> skip2 --+  |",
    "  |                    |  |",
    "[Enc3] 256 -> skip3 +  |  |",
    "  |                  |  |  |",
    "[Enc4] 512 -> sk4 +  |  |  |",
    "  |               |  |  |  |",
    "[Bottleneck] 1024 |  |  |  |",
    "  |               |  |  |  |",
    "[Dec4] 512 <------+  |  |  |",
    "  |                  |  |  |",
    "[Dec3] 256 <---------+  |  |",
    "  |                     |  |",
    "[Dec2] 128 <------------+  |",
    "  |                        |",
    "[Dec1] 64 <----------------+",
    "  |",
    "  v",
    "Output (1,256,256)",
    "```"
)))

cells.append(code(L(
    "class ConvBlock(nn.Module):",
    "    '''Two 3x3 conv layers with BatchNorm and ReLU.'''",
    "",
    "    def __init__(self, in_ch, out_ch):",
    "        super().__init__()",
    "        self.block = nn.Sequential(",
    "            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),",
    "            nn.BatchNorm2d(out_ch),",
    "            nn.ReLU(inplace=True),",
    "            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),",
    "            nn.BatchNorm2d(out_ch),",
    "            nn.ReLU(inplace=True),",
    "        )",
    "",
    "    def forward(self, x):",
    "        return self.block(x)",
    "",
    "",
    "class Encoder(nn.Module):",
    "    '''Encoder: 4 ConvBlocks with max-pooling between them.'''",
    "",
    "    def __init__(self, channels=(3, 64, 128, 256, 512)):",
    "        super().__init__()",
    "        self.blocks = nn.ModuleList()",
    "        self.pools  = nn.ModuleList()",
    "        for i in range(len(channels) - 1):",
    "            self.blocks.append(ConvBlock(channels[i], channels[i + 1]))",
    "            if i < len(channels) - 2:",
    "                self.pools.append(nn.MaxPool2d(2))",
    "",
    "    def forward(self, x):",
    "        skips = []",
    "        for i, block in enumerate(self.blocks):",
    "            x = block(x)",
    "            skips.append(x)",
    "            if i < len(self.pools):",
    "                x = self.pools[i](x)",
    "        return skips",
    "",
    "",
    "class Decoder(nn.Module):",
    "    '''Decoder: Upsample + skip concatenation + ConvBlock.'''",
    "",
    "    def __init__(self, channels=(1024, 512, 256, 128, 64)):",
    "        super().__init__()",
    "        self.ups    = nn.ModuleList()",
    "        self.blocks = nn.ModuleList()",
    "        for i in range(len(channels) - 1):",
    "            self.ups.append(",
    "                nn.ConvTranspose2d(channels[i], channels[i + 1], 2, stride=2)",
    "            )",
    "            self.blocks.append(ConvBlock(channels[i], channels[i + 1]))",
    "",
    "    def forward(self, x, skips):",
    "        for i in range(len(self.ups)):",
    "            x = self.ups[i](x)",
    "            skip = skips[i]",
    "            # handle size mismatch from odd dimensions",
    "            if x.shape != skip.shape:",
    "                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)",
    "            x = torch.cat([skip, x], dim=1)",
    "            x = self.blocks[i](x)",
    "        return x",
    "",
    "",
    "class UNet(nn.Module):",
    "    '''U-Net for binary segmentation.'''",
    "",
    "    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):",
    "        super().__init__()",
    "        self.encoder    = Encoder(channels=(in_ch, *features))",
    "        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)",
    "        self.decoder    = Decoder(channels=(features[-1] * 2, *reversed(features)))",
    "        self.head       = nn.Conv2d(features[0], out_ch, 1)",
    "",
    "    def forward(self, x):",
    "        skips = self.encoder(x)",
    "        x = self.bottleneck(nn.MaxPool2d(2)(skips[-1]))",
    "        x = self.decoder(x, list(reversed(skips)))",
    "        return self.head(x)",
    "",
    "",
    "# Quick sanity check",
    "model = UNet().to(DEVICE)",
    "dummy = torch.randn(2, 3, 256, 256, device=DEVICE)",
    "out   = model(dummy)",
    "print(f'Input:  {dummy.shape}')",
    "print(f'Output: {out.shape}')",
    "params = sum(p.numel() for p in model.parameters())",
    "print(f'Parameters: {params:,}')"
)))

# ── 8. Loss Functions ─────────────────────────────────────────────────────
cells.append(md(L(
    '<a id="losses"></a>',
    "## 5. Loss Functions",
    "",
    "Segmentation losses must handle **class imbalance** (foreground is often tiny).",
    "",
    "| Loss | Handles Imbalance | Smooth Gradient | Notes |",
    "|------|:-----------------:|:---------------:|-------|",
    "| BCE | No | Yes | Baseline |",
    "| Dice | Yes | Moderate | Directly optimises F1 |",
    "| Focal | Yes | Yes | Down-weights easy pixels |",
    "| Tversky | Yes | Yes | Tunable FP/FN penalty |",
    "| Combined | Yes | Yes | BCE + Dice (best default) |"
)))

cells.append(code(L(
    "class DiceLoss(nn.Module):",
    "    def __init__(self, smooth=1.0):",
    "        super().__init__()",
    "        self.smooth = smooth",
    "",
    "    def forward(self, logits, targets):",
    "        probs = torch.sigmoid(logits)",
    "        pflat = probs.view(-1)",
    "        tflat = targets.view(-1)",
    "        intersection = (pflat * tflat).sum()",
    "        return 1 - (2. * intersection + self.smooth) / (pflat.sum() + tflat.sum() + self.smooth)",
    "",
    "",
    "class FocalLoss(nn.Module):",
    "    def __init__(self, alpha=0.25, gamma=2.0):",
    "        super().__init__()",
    "        self.alpha = alpha",
    "        self.gamma = gamma",
    "",
    "    def forward(self, logits, targets):",
    "        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')",
    "        pt = torch.exp(-bce)",
    "        focal = self.alpha * (1 - pt) ** self.gamma * bce",
    "        return focal.mean()",
    "",
    "",
    "class TverskyLoss(nn.Module):",
    "    '''Tversky loss with tunable alpha/beta for FP/FN trade-off.'''",
    "    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):",
    "        super().__init__()",
    "        self.alpha = alpha",
    "        self.beta = beta",
    "        self.smooth = smooth",
    "",
    "    def forward(self, logits, targets):",
    "        probs = torch.sigmoid(logits)",
    "        pflat = probs.view(-1)",
    "        tflat = targets.view(-1)",
    "        tp = (pflat * tflat).sum()",
    "        fp = (pflat * (1 - tflat)).sum()",
    "        fn = ((1 - pflat) * tflat).sum()",
    "        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)",
    "        return 1 - tversky",
    "",
    "",
    "class CombinedLoss(nn.Module):",
    "    '''BCE + Dice -- robust default for segmentation.'''",
    "    def __init__(self, bce_weight=0.5, dice_weight=0.5):",
    "        super().__init__()",
    "        self.bce_weight  = bce_weight",
    "        self.dice_weight = dice_weight",
    "        self.dice = DiceLoss()",
    "",
    "    def forward(self, logits, targets):",
    "        bce = F.binary_cross_entropy_with_logits(logits, targets)",
    "        dice = self.dice(logits, targets)",
    "        return self.bce_weight * bce + self.dice_weight * dice",
    "",
    "",
    "# Quick test",
    "pred = torch.randn(2, 1, 64, 64)",
    "tgt  = torch.randint(0, 2, (2, 1, 64, 64)).float()",
    "for name, fn in [('BCE', nn.BCEWithLogitsLoss()), ('Dice', DiceLoss()),",
    "                  ('Focal', FocalLoss()), ('Tversky', TverskyLoss()),",
    "                  ('Combined', CombinedLoss())]:",
    "    print(f'{name:10s}: {fn(pred, tgt):.4f}')"
)))

# ── 9. Training Pipeline ──────────────────────────────────────────────────
cells.append(md(L(
    '<a id="training"></a>',
    "## 6. Training Pipeline",
    "",
    "Production-grade training loop with:",
    "- **Mixed precision** (torch.cuda.amp)",
    "- **Gradient accumulation** (simulates larger batch sizes)",
    "- **CosineAnnealingWarmRestarts** scheduler",
    "- **Early stopping** via best-Dice checkpointing"
)))

cells.append(code(L(
    "def compute_dice(preds, targets, threshold=0.5):",
    "    '''Compute Dice score for a batch.'''",
    "    preds = (torch.sigmoid(preds) > threshold).float()",
    "    pflat = preds.view(-1)",
    "    tflat = targets.view(-1)",
    "    inter = (pflat * tflat).sum()",
    "    return (2. * inter + 1e-6) / (pflat.sum() + tflat.sum() + 1e-6)"
)))

cells.append(code(L(
    "def train_one_epoch(model, loader, criterion, optimizer, scaler, device,",
    "                    accumulation_steps=1):",
    "    model.train()",
    "    running_loss = 0.0",
    "    running_dice = 0.0",
    "    optimizer.zero_grad()",
    "",
    "    for i, (images, masks) in enumerate(loader):",
    "        images = images.to(device, non_blocking=True)",
    "        masks  = masks.to(device, non_blocking=True)",
    "",
    "        with autocast(enabled=(device.type == 'cuda')):",
    "            logits = model(images)",
    "            loss = criterion(logits, masks) / accumulation_steps",
    "",
    "        scaler.scale(loss).backward()",
    "",
    "        if (i + 1) % accumulation_steps == 0:",
    "            scaler.step(optimizer)",
    "            scaler.update()",
    "            optimizer.zero_grad()",
    "",
    "        running_loss += loss.item() * accumulation_steps",
    "        running_dice += compute_dice(logits, masks).item()",
    "",
    "    n = len(loader)",
    "    return running_loss / n, running_dice / n",
    "",
    "",
    "@torch.no_grad()",
    "def validate(model, loader, criterion, device):",
    "    model.eval()",
    "    running_loss = 0.0",
    "    running_dice = 0.0",
    "",
    "    for images, masks in loader:",
    "        images = images.to(device, non_blocking=True)",
    "        masks  = masks.to(device, non_blocking=True)",
    "",
    "        with autocast(enabled=(device.type == 'cuda')):",
    "            logits = model(images)",
    "            loss = criterion(logits, masks)",
    "",
    "        running_loss += loss.item()",
    "        running_dice += compute_dice(logits, masks).item()",
    "",
    "    n = len(loader)",
    "    return running_loss / n, running_dice / n"
)))

cells.append(code(L(
    "# ── Hyperparameters ─────────────────────────────────────────────────",
    "NUM_EPOCHS   = 5   # keep low for demo; use 30-80 in competition",
    "LR           = 1e-3",
    "ACCUM_STEPS  = 2",
    "",
    "model     = UNet().to(DEVICE)",
    "criterion = CombinedLoss()",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)",
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)",
    "scaler    = GradScaler(enabled=(DEVICE.type == 'cuda'))",
    "",
    "best_dice = 0.0",
    "history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}",
    "",
    "for epoch in range(NUM_EPOCHS):",
    "    tl, td = train_one_epoch(model, train_loader, criterion, optimizer, scaler,",
    "                              DEVICE, ACCUM_STEPS)",
    "    vl, vd = validate(model, val_loader, criterion, DEVICE)",
    "    scheduler.step()",
    "",
    "    history['train_loss'].append(tl)",
    "    history['val_loss'].append(vl)",
    "    history['train_dice'].append(td)",
    "    history['val_dice'].append(vd)",
    "",
    "    tag = ''",
    "    if vd > best_dice:",
    "        best_dice = vd",
    "        tag = ' ** best **'",
    "",
    "    print(f'Epoch {epoch+1}/{NUM_EPOCHS} | '",
    "          f'TrLoss={tl:.4f} TrDice={td:.4f} | '",
    "          f'VaLoss={vl:.4f} VaDice={vd:.4f}{tag}')",
    "",
    "print(f'\\nBest validation Dice: {best_dice:.4f}')"
)))

cells.append(md(L(
    "### Training Curves"
)))

cells.append(code(L(
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))",
    "",
    "ax1.plot(history['train_loss'], label='Train Loss', marker='o')",
    "ax1.plot(history['val_loss'],   label='Val Loss',   marker='s')",
    "ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')",
    "ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)",
    "",
    "ax2.plot(history['train_dice'], label='Train Dice', marker='o')",
    "ax2.plot(history['val_dice'],   label='Val Dice',   marker='s')",
    "ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')",
    "ax2.set_title('Dice Score Curves'); ax2.legend(); ax2.grid(True, alpha=0.3)",
    "",
    "plt.suptitle('Training History', fontsize=14, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

# ── 10. Advanced Architectures ────────────────────────────────────────────
cells.append(md(L(
    '<a id="advanced"></a>',
    "## 7. Advanced Architectures",
    "",
    "The `segmentation_models_pytorch` library gives us access to **pre-trained encoders**",
    "with state-of-the-art decoder heads.",
    "",
    "| Architecture | Key Idea | When to Use |",
    "|-------------|----------|-------------|",
    "| **U-Net++** | Dense skip connections | Better feature fusion |",
    "| **DeepLabV3+** | Atrous Spatial Pyramid Pooling | Multi-scale context |",
    "| **FPN** | Feature Pyramid Network | Multi-scale objects |",
    "| **SegFormer** | Transformer encoder | Large datasets, high resolution |"
)))

cells.append(code(L(
    "# ── U-Net++ with EfficientNet-B3 encoder ─────────────────────────────",
    "unetpp = smp.UnetPlusPlus(",
    "    encoder_name='efficientnet-b3',",
    "    encoder_weights='imagenet',",
    "    in_channels=3,",
    "    classes=1,",
    "    activation=None,",
    ")",
    "print(f'U-Net++ params: {sum(p.numel() for p in unetpp.parameters()):,}')"
)))

cells.append(code(L(
    "# ── DeepLabV3+ with ResNet-50 encoder ────────────────────────────────",
    "deeplab = smp.DeepLabV3Plus(",
    "    encoder_name='resnet50',",
    "    encoder_weights='imagenet',",
    "    in_channels=3,",
    "    classes=1,",
    "    activation=None,",
    ")",
    "print(f'DeepLabV3+ params: {sum(p.numel() for p in deeplab.parameters()):,}')"
)))

cells.append(code(L(
    "# ── FPN with MobileNetV2 encoder (lightweight) ───────────────────────",
    "fpn_model = smp.FPN(",
    "    encoder_name='mobilenet_v2',",
    "    encoder_weights='imagenet',",
    "    in_channels=3,",
    "    classes=1,",
    "    activation=None,",
    ")",
    "print(f'FPN (MobileNetV2) params: {sum(p.numel() for p in fpn_model.parameters()):,}')",
    "",
    "# Sanity check forward pass",
    "with torch.no_grad():",
    "    test_in  = torch.randn(1, 3, 256, 256)",
    "    test_out = fpn_model(test_in)",
    "    print(f'FPN output shape: {test_out.shape}')"
)))

# ── 11. Post-Processing ───────────────────────────────────────────────────
cells.append(md(L(
    '<a id="postproc"></a>',
    "## 8. Post-Processing",
    "",
    "Raw model outputs need post-processing for competition submissions:",
    "- **Threshold optimization** (not always 0.5!)",
    "- **Test-Time Augmentation (TTA)**",
    "- **Morphological operations** (opening, closing, remove small objects)"
)))

cells.append(code(L(
    "def find_best_threshold(model, loader, device, thresholds=None):",
    "    '''Sweep thresholds to maximise Dice on validation set.'''",
    "    if thresholds is None:",
    "        thresholds = np.arange(0.1, 0.91, 0.05)",
    "",
    "    model.eval()",
    "    all_preds, all_masks = [], []",
    "",
    "    with torch.no_grad():",
    "        for images, masks in loader:",
    "            images = images.to(device)",
    "            logits = model(images)",
    "            all_preds.append(torch.sigmoid(logits).cpu())",
    "            all_masks.append(masks.cpu())",
    "",
    "    preds = torch.cat(all_preds)",
    "    masks = torch.cat(all_masks)",
    "",
    "    best_th, best_dice = 0.5, 0.0",
    "    results = []",
    "    for th in thresholds:",
    "        binary = (preds > th).float()",
    "        pflat = binary.view(-1)",
    "        tflat = masks.view(-1)",
    "        inter = (pflat * tflat).sum()",
    "        dice = (2. * inter + 1e-6) / (pflat.sum() + tflat.sum() + 1e-6)",
    "        results.append((th, dice.item()))",
    "        if dice > best_dice:",
    "            best_dice = dice.item()",
    "            best_th = th",
    "",
    "    return best_th, best_dice, results",
    "",
    "",
    "best_th, best_dice, curve = find_best_threshold(model, val_loader, DEVICE)",
    "print(f'Best threshold: {best_th:.2f}  Dice: {best_dice:.4f}')",
    "",
    "# Plot threshold sweep",
    "ths, dices = zip(*curve)",
    "plt.figure(figsize=(8, 4))",
    "plt.plot(ths, dices, marker='o')",
    "plt.axvline(best_th, color='r', linestyle='--', label=f'Best={best_th:.2f}')",
    "plt.xlabel('Threshold'); plt.ylabel('Dice Score')",
    "plt.title('Threshold Optimization')",
    "plt.legend(); plt.grid(True, alpha=0.3)",
    "plt.show()"
)))

cells.append(md(L(
    "### Test-Time Augmentation (TTA)"
)))

cells.append(code(L(
    "def tta_predict(model, image, device):",
    "    '''Apply TTA: original + hflip + vflip + hflip+vflip, average predictions.'''",
    "    model.eval()",
    "    augmented = [",
    "        image,",
    "        torch.flip(image, dims=[-1]),       # horizontal flip",
    "        torch.flip(image, dims=[-2]),       # vertical flip",
    "        torch.flip(image, dims=[-1, -2]),   # both flips",
    "    ]",
    "",
    "    preds = []",
    "    with torch.no_grad():",
    "        for aug_img in augmented:",
    "            logit = model(aug_img.unsqueeze(0).to(device))",
    "            pred  = torch.sigmoid(logit).cpu()",
    "            preds.append(pred)",
    "",
    "    # Reverse augmentations",
    "    preds[1] = torch.flip(preds[1], dims=[-1])",
    "    preds[2] = torch.flip(preds[2], dims=[-2])",
    "    preds[3] = torch.flip(preds[3], dims=[-1, -2])",
    "",
    "    return torch.stack(preds).mean(dim=0).squeeze()",
    "",
    "",
    "# Demo: compare regular vs TTA prediction",
    "sample_img, sample_mask = val_ds[0]",
    "with torch.no_grad():",
    "    regular = torch.sigmoid(model(sample_img.unsqueeze(0).to(DEVICE))).cpu().squeeze()",
    "tta_pred = tta_predict(model, sample_img, DEVICE)",
    "",
    "print(f'Regular pred range: [{regular.min():.3f}, {regular.max():.3f}]')",
    "print(f'TTA pred range:     [{tta_pred.min():.3f}, {tta_pred.max():.3f}]')"
)))

cells.append(md(L(
    "### Morphological Post-Processing"
)))

cells.append(code(L(
    "def morphological_postprocess(mask, min_area=100, kernel_size=5):",
    "    '''Apply morphological operations to clean up predictions.'''",
    "    mask_uint8 = (mask * 255).astype(np.uint8)",
    "    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))",
    "",
    "    # Close small holes",
    "    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)",
    "    # Open to remove small noise",
    "    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)",
    "",
    "    # Remove small connected components",
    "    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)",
    "    cleaned = np.zeros_like(opened)",
    "    for i in range(1, num_labels):",
    "        if stats[i, cv2.CC_STAT_AREA] >= min_area:",
    "            cleaned[labels == i] = 255",
    "",
    "    return (cleaned / 255).astype(np.float32)",
    "",
    "",
    "# Demo",
    "raw_pred = (regular.numpy() > best_th).astype(np.float32)",
    "cleaned  = morphological_postprocess(raw_pred)",
    "",
    "fig, axes = plt.subplots(1, 3, figsize=(14, 4))",
    "axes[0].imshow(sample_mask.squeeze(), cmap='gray')",
    "axes[0].set_title('Ground Truth')",
    "axes[1].imshow(raw_pred, cmap='gray')",
    "axes[1].set_title('Raw Prediction')",
    "axes[2].imshow(cleaned, cmap='gray')",
    "axes[2].set_title('After Morphology')",
    "for ax in axes: ax.axis('off')",
    "plt.suptitle('Morphological Post-Processing', fontsize=13, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

# ── 12. Evaluation & Visualization ────────────────────────────────────────
cells.append(md(L(
    '<a id="eval"></a>',
    "## 9. Evaluation & Visualization",
    "",
    "Standard segmentation metrics:",
    "- **IoU (Jaccard):** Intersection / Union",
    "- **Dice (F1):** 2 * Intersection / (|Pred| + |Target|)",
    "- **Pixel Accuracy:** Correct pixels / Total pixels"
)))

cells.append(code(L(
    "def compute_metrics(pred, target, threshold=0.5):",
    "    '''Compute IoU, Dice, and pixel accuracy.'''",
    "    pred_bin = (pred > threshold).float()",
    "    tp = (pred_bin * target).sum()",
    "    fp = (pred_bin * (1 - target)).sum()",
    "    fn = ((1 - pred_bin) * target).sum()",
    "    tn = ((1 - pred_bin) * (1 - target)).sum()",
    "",
    "    iou      = (tp + 1e-6) / (tp + fp + fn + 1e-6)",
    "    dice     = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)",
    "    accuracy = (tp + tn) / (tp + tn + fp + fn)",
    "",
    "    return {",
    "        'iou': iou.item(),",
    "        'dice': dice.item(),",
    "        'pixel_accuracy': accuracy.item(),",
    "        'precision': (tp / (tp + fp + 1e-6)).item(),",
    "        'recall': (tp / (tp + fn + 1e-6)).item(),",
    "    }",
    "",
    "",
    "# Evaluate on validation set",
    "all_metrics = []",
    "model.eval()",
    "with torch.no_grad():",
    "    for images, masks in val_loader:",
    "        images = images.to(DEVICE)",
    "        logits = model(images)",
    "        probs  = torch.sigmoid(logits).cpu()",
    "        for j in range(images.size(0)):",
    "            m = compute_metrics(probs[j], masks[j], threshold=best_th)",
    "            all_metrics.append(m)",
    "",
    "# Aggregate",
    "avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}",
    "print('Validation Metrics (mean):')",
    "print(f'  IoU:            {avg[\"iou\"]:.4f}')",
    "print(f'  Dice:           {avg[\"dice\"]:.4f}')",
    "print(f'  Pixel Accuracy: {avg[\"pixel_accuracy\"]:.4f}')",
    "print(f'  Precision:      {avg[\"precision\"]:.4f}')",
    "print(f'  Recall:         {avg[\"recall\"]:.4f}')"
)))

cells.append(code(L(
    "# ── Visual comparison: image / ground truth / prediction ─────────────",
    "model.eval()",
    "fig, axes = plt.subplots(3, 5, figsize=(18, 11))",
    "",
    "for i in range(5):",
    "    img_t, mask_t = val_ds[i * 10]",
    "    with torch.no_grad():",
    "        pred = torch.sigmoid(model(img_t.unsqueeze(0).to(DEVICE))).cpu().squeeze()",
    "",
    "    # Denormalize image for display",
    "    img_np = img_t.permute(1, 2, 0).numpy()",
    "    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])",
    "    img_np = np.clip(img_np, 0, 1)",
    "",
    "    axes[0, i].imshow(img_np)",
    "    axes[0, i].set_title('Image')",
    "    axes[1, i].imshow(mask_t.squeeze(), cmap='gray')",
    "    axes[1, i].set_title('Ground Truth')",
    "    axes[2, i].imshow((pred.numpy() > best_th).astype(float), cmap='gray')",
    "    axes[2, i].set_title(f'Pred (th={best_th:.2f})')",
    "",
    "for ax in axes.flat:",
    "    ax.axis('off')",
    "",
    "plt.suptitle('Predictions vs Ground Truth', fontsize=14, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

# ── 13. Competition Tips ──────────────────────────────────────────────────
cells.append(md(L(
    '<a id="tips"></a>',
    "## 10. Competition Tips",
    "",
    "### Sliding Window Inference",
    "For high-resolution images, predict on overlapping patches and blend:",
    "",
    "```",
    "+-------+-------+-------+",
    "|       | overlap       |",
    "|  P1   |  P2   |  P3   |",
    "|       | region        |",
    "+-------+-------+-------+",
    "|  P4   |  P5   |  P6   |",
    "+-------+-------+-------+",
    "```"
)))

cells.append(code(L(
    "def sliding_window_inference(model, image, window_size=256, stride=128, device='cpu'):",
    "    '''Predict on overlapping patches and average.'''",
    "    model.eval()",
    "    C, H, W = image.shape",
    "    pred_sum   = torch.zeros(1, H, W)",
    "    count_map  = torch.zeros(1, H, W)",
    "",
    "    for y in range(0, H - window_size + 1, stride):",
    "        for x in range(0, W - window_size + 1, stride):",
    "            patch = image[:, y:y+window_size, x:x+window_size].unsqueeze(0).to(device)",
    "            with torch.no_grad():",
    "                out = torch.sigmoid(model(patch)).cpu()",
    "            pred_sum[:, y:y+window_size, x:x+window_size] += out.squeeze(0)",
    "            count_map[:, y:y+window_size, x:x+window_size] += 1",
    "",
    "    # Handle edges that might be missed",
    "    count_map[count_map == 0] = 1",
    "    return (pred_sum / count_map).squeeze()",
    "",
    "",
    "# Demo on a larger synthetic image",
    "big_img, big_mask = generate_synthetic_sample(size=512, num_shapes=12, seed=42)",
    "big_tensor = val_transform(image=big_img)['image']",
    "sw_pred = sliding_window_inference(model, big_tensor, window_size=256, stride=128, device=DEVICE)",
    "print(f'Sliding window prediction shape: {sw_pred.shape}')"
)))

cells.append(md(L(
    "### Multi-Scale Inference & Pseudo Labeling"
)))

cells.append(code(L(
    "def multi_scale_inference(model, image, scales=(0.75, 1.0, 1.25), device='cpu'):",
    "    '''Predict at multiple scales and average.'''",
    "    model.eval()",
    "    C, H, W = image.shape",
    "    pred_sum = torch.zeros(1, H, W)",
    "",
    "    for scale in scales:",
    "        sH, sW = int(H * scale), int(W * scale)",
    "        scaled = F.interpolate(image.unsqueeze(0), size=(sH, sW),",
    "                               mode='bilinear', align_corners=False)",
    "        with torch.no_grad():",
    "            out = torch.sigmoid(model(scaled.to(device))).cpu()",
    "        resized = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)",
    "        pred_sum += resized.squeeze(0)",
    "",
    "    return (pred_sum / len(scales)).squeeze()",
    "",
    "",
    "def generate_pseudo_labels(model, unlabeled_loader, threshold_high=0.9,",
    "                           threshold_low=0.1, device='cpu'):",
    "    '''Generate high-confidence pseudo labels for semi-supervised learning.'''",
    "    model.eval()",
    "    pseudo_images, pseudo_masks = [], []",
    "",
    "    with torch.no_grad():",
    "        for images, _ in unlabeled_loader:",
    "            probs = torch.sigmoid(model(images.to(device))).cpu()",
    "            for i in range(images.size(0)):",
    "                mask = probs[i].squeeze()",
    "                confident = (mask > threshold_high) | (mask < threshold_low)",
    "                if confident.float().mean() > 0.8:  # >80% confident pixels",
    "                    pseudo_masks.append((mask > 0.5).float())",
    "                    pseudo_images.append(images[i])",
    "",
    "    print(f'Generated {len(pseudo_images)} pseudo-labeled samples')",
    "    return pseudo_images, pseudo_masks",
    "",
    "",
    "# Demo multi-scale",
    "ms_pred = multi_scale_inference(model, sample_img, device=DEVICE)",
    "print(f'Multi-scale pred shape: {ms_pred.shape}')"
)))

cells.append(md(L(
    "### Run-Length Encoding (RLE) for Submission"
)))

cells.append(code(L(
    "def rle_encode(mask):",
    "    '''Run-Length Encode a binary mask (Kaggle format).'''",
    "    pixels = mask.flatten(order='F')  # Fortran-order (column-major)",
    "    pixels = np.concatenate([[0], pixels, [0]])",
    "    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1",
    "    runs[1::2] -= runs[::2]",
    "    return ' '.join(str(x) for x in runs)",
    "",
    "",
    "def rle_decode(rle_string, shape):",
    "    '''Decode RLE string back to binary mask.'''",
    "    s = list(map(int, rle_string.split()))",
    "    starts, lengths = s[0::2], s[1::2]",
    "    starts = np.array(starts) - 1",
    "    ends = starts + np.array(lengths)",
    "    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)",
    "    for start, end in zip(starts, ends):",
    "        mask[start:end] = 1",
    "    return mask.reshape(shape, order='F')",
    "",
    "",
    "# Demo round-trip",
    "demo_mask = (raw_pred > 0.5).astype(np.uint8)",
    "encoded = rle_encode(demo_mask)",
    "decoded = rle_decode(encoded, demo_mask.shape)",
    "print(f'RLE length: {len(encoded)} chars')",
    "print(f'Round-trip match: {np.array_equal(demo_mask, decoded)}')"
)))

# ── 14. Competition References ────────────────────────────────────────────
cells.append(md(L(
    "## Competition References",
    "",
    "These techniques are directly applicable to recent Kaggle segmentation competitions:",
    "",
    "| Competition | Key Challenge | Winning Approach |",
    "|------------|---------------|-----------------|",
    "| **Vesuvius Challenge** | Ink detection on CT scans | 3D U-Net, heavy TTA, sliding window |",
    "| **Scientific Image Forgery Detection** | Tampering localisation | ELA features + U-Net++ |",
    "| **HuBMAP + HPA** | Organ segmentation | DeepLabV3+ with multi-scale |",
    "| **SenNet + HOA** | Blood vessel segmentation | U-Net++ with aggressive augmentation |",
    "| **UW-Madison GI Tract** | 2.5D segmentation | 2.5D slices + FPN backbone |",
    "",
    "> **Key Takeaway:** Competition-winning solutions almost always combine",
    "> *ensemble of architectures* + *heavy augmentation* + *meticulous post-processing*."
)))

# ── 15. Further Reading ───────────────────────────────────────────────────
cells.append(md(L(
    "## Interpretation, Trade-offs, and Limitations",
    "",
    "- **Observation:** segmentation quality often improves more from cleaner masks and augmentation policy than from swapping one decoder for another.",
    "- **Interpretation:** boundary-sensitive losses help small structures because they reward pixel-level precision instead of only region overlap.",
    "- **Trade-off:** heavier ensembles and post-processing can lift scores, yet they also raise latency, memory use, and deployment complexity.",
    "- **Limitation:** offline IoU gains do not always transfer if the validation split misses rare artifact patterns, so challenge-specific evaluation still matters."
)))

cells.append(md(L(
    '<a id="reading"></a>',
    "## Further Reading",
    "",
    "### Papers",
    "- Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (2015)",
    "- Zhou et al., [UNet++: A Nested U-Net Architecture](https://arxiv.org/abs/1807.10165) (2018)",
    "- Chen et al., [DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611) (2018)",
    "- Xie et al., [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (2021)",
    "- Lin et al., [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) (2017)",
    "",
    "### Libraries",
    "- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) -- 50+ encoders, 9 decoder architectures",
    "- [Albumentations](https://albumentations.ai/) -- fast, flexible image augmentation",
    "- [torchmetrics](https://torchmetrics.readthedocs.io/) -- plug-and-play metric computation",
    "",
    "### Kaggle Notebooks (Top Solutions)",
    "- [HuBMAP 1st Place Solution](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature) -- multi-model ensemble",
    "- [Vesuvius Challenge Discussions](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion) -- 3D techniques"
)))

# ── 16. CTA ───────────────────────────────────────────────────────────────
cells.append(md(L(
    "---",
    "",
    "## What Next?",
    "",
    "If you found this notebook useful:",
    "",
    "1. **Upvote** this notebook to help others find it",
    "2. **Fork** it and swap in your own dataset",
    "3. **Try different encoders** -- `efficientnet-b5`, `resnet101`, `mit_b3`",
    "4. **Experiment with losses** -- Tversky works great for small objects",
    "5. **Check out my other notebooks** in this series on Feature Engineering, Ensemble Stacking, and more",
    "",
    "> *Built with PyTorch + segmentation_models_pytorch. All code MIT licensed.*",
    "",
    "---",
    "",
    "**Happy segmenting!**"
)))

# ---------------------------------------------------------------------------
# Assemble the notebook and write to disk
# ---------------------------------------------------------------------------

write_notebook(cells, __file__, "image_segmentation_masterclass.ipynb")
