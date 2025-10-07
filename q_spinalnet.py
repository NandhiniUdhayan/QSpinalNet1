# q_spinalnet.py
import os
import math
from typing import Tuple
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision

# Optional better segmentation backbone:
# pip install segmentation-models-pytorch timm
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False

###############################################################################
# Utilities
###############################################################################
def load_image(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")  # grayscale mammogram
    return np.array(im)

def save_image(np_img: np.ndarray, out_path: str):
    im = Image.fromarray(np.clip(np_img,0,255).astype(np.uint8))
    im.save(out_path)

###############################################################################
# Preprocessing: CEAMF approximate + Z-score normalization + Context-Aware Contrast
# These follow the paper's triple-way preprocessing description (CEAMF, Z-score, energy curves).
# Implementation notes:
# - CEAMF is implemented as an adaptive median/median-of-window where window size adapts by local brightness
#   (paper algorithm approximated for practicality).
# - Context-aware contrast enhancement builds an "energy curve" across graylevels using local spatial correlations
#   and applies piecewise histogram transforms (three ranges) per algorithm.
###############################################################################

def ceamf_filter(img: np.ndarray, max_window=15, min_window=3) -> np.ndarray:
    """
    Approximate CEAMF described in paper: For each pixel, pick window U adaptive to local bright-spot size,
    then apply median filter in that window. For performance we vectorize using multiple median filters
    and pick based on local variance.
    """
    img = img.astype(np.float32)
    h, w = img.shape
    # compute local variance map (fast)
    mean = cv2.blur(img, (7,7))
    sqmean = cv2.blur(img*img, (7,7))
    var = np.maximum(0, sqmean - mean*mean)

    # map variance -> window size between min_window..max_window
    var_norm = (var - var.min()) / (var.max() - var.min() + 1e-8)
    window_sizes = (var_norm * (max_window - min_window)).astype(np.int32) + min_window
    # precompute median filtered images for a few window sizes (odd only)
    odd_sizes = sorted(list(set([s if s%2==1 else s+1 for s in [min_window, 5, 7, 9, 11, 13, max_window]])))
    medians = {}
    for s in odd_sizes:
        medians[s] = cv2.medianBlur(img.astype(np.uint8), s)

    out = np.zeros_like(img)
    for s in odd_sizes:
        mask = (window_sizes == s)
        if mask.sum() == 0: continue
        out[mask] = medians[s][mask]
    # fallback: if some pixels still zero (shouldn't), use original
    out[out==0] = img[out==0]
    return out.astype(np.uint8)

def zscore_normalize(img: np.ndarray) -> np.ndarray:
    mu = img.mean()
    sigma = img.std() + 1e-8
    return (img - mu) / sigma

def context_aware_contrast(img: np.ndarray, levels=256) -> np.ndarray:
    """
    Implements the paper's 3-region energy curve transfer function approach:
    - compute spatial correlation matrix approx via local co-occurrence (here local mean similarity)
    - derive energy curve, clip, split into low/med/high, compute pdf/cdf per region and form composite transfer function.
    """
    img_uint8 = np.clip(((img - img.min()) / (img.max()-img.min()+1e-8) * 255).astype(np.uint8),0,255)
    h, w = img_uint8.shape

    # Spatial correlation: for each pixel, count neighbors with similar intensity within distance d
    d = 1
    padded = np.pad(img_uint8, 1, mode='reflect')
    neighbors = []
    # sum similarity over 8-neighborhood
    sim = np.zeros_like(img_uint8, dtype=np.float32)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if dy==0 and dx==0: continue
            shifted = padded[1+dy:h+1+dy, 1+dx:w+1+dx]
            sim += (np.abs(shifted.astype(np.int16) - img_uint8.astype(np.int16)) < 15).astype(np.float32)
    # Normalize spatial correlation to [0,1]
    Cij = sim / 8.0
    # energy curve: for each gray level g, compute sum(Cij for pixels with intensity g)
    energy = np.zeros(levels, dtype=np.float32)
    for g in range(levels):
        mask = (img_uint8 == g)
        if mask.any():
            energy[g] = Cij[mask].sum()
    # clip
    cmean = energy.mean()
    cmedian = np.median(energy)
    Cclip = cmean + cmedian
    eclip = np.minimum(energy, Cclip)

    # compute thresholds by std as in paper
    enemean = eclip.mean()
    std = eclip.std()
    Llow = max(0, int(max(0, enemean - std) / (eclip.max()+1e-8) * (levels-1)))
    Lhigh = min(levels-1, int(min(levels-1, enemean + std) / (eclip.max()+1e-8) * (levels-1)))

    # Build pdf & cdf for three regions
    def region_transform(img_arr, lo, hi):
        # get histogram within [lo,hi]
        hist = np.bincount(img_arr.flatten(), minlength=levels)
        # zero out outside region
        reg_hist = np.zeros_like(hist, dtype=np.float32)
        reg_hist[lo:hi+1] = hist[lo:hi+1]
        pdf = reg_hist / (reg_hist.sum()+1e-8)
        cdf = np.cumsum(pdf)
        # normalize cdf to remap [lo..hi] -> [lo..hi]
        mapping = np.arange(levels).astype(np.uint8)
        if pdf.sum() > 0:
            # linear stretch within region using cdf
            region_values = (cdf[lo:hi+1] - cdf[lo]) / (cdf[hi]-cdf[lo]+1e-8)
            region_map = (region_values * (hi-lo) + lo).astype(np.uint8)
            mapping[lo:hi+1] = region_map
        return mapping

    map_lo = region_transform(img_uint8, 0, Llow)
    map_me = region_transform(img_uint8, Llow+1, Lhigh)
    map_hi = region_transform(img_uint8, Lhigh+1, levels-1)

    # compose final mapping: prefer region maps for their ranges
    final_map = np.arange(levels, dtype=np.uint8)
    final_map[:Llow+1] = map_lo[:Llow+1]
    final_map[Llow+1:Lhigh+1] = map_me[Llow+1:Lhigh+1]
    final_map[Lhigh+1:] = map_hi[Lhigh+1:]
    # apply mapping
    out = final_map[img_uint8]
    return out.astype(np.uint8)

def triple_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Full triple preprocessing pipeline from the paper:
      1. CEAMF denoising (approx)
      2. Z-score normalization
      3. Context-aware contrast enhancement
    Returns: uint8 image scaled to [0,255]
    """
    den = ceamf_filter(img)
    # normalize then map back to uint8 for contrast function
    z = zscore_normalize(den)
    # rescale z to 0..255 for contrast function
    z_min, z_max = z.min(), z.max()
    z_scaled = ((z - z_min) / (z_max - z_min + 1e-8) * 255).astype(np.uint8)
    enhanced = context_aware_contrast(z_scaled)
    return enhanced

###############################################################################
# Segmentation: SwinResUnet3+ style model skeleton
# - If segmentation-models-pytorch is available, we use UNet with a swin backbone (if available).
# - Otherwise we provide a minimal UNet fallback.
###############################################################################

class SimpleUNetDecoder(nn.Module):
    def __init__(self, in_ch=64, n_classes=1):
        super().__init__()
        # tiny decoder for prototyping
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_ch//2, n_classes, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class SegmentationModelWrapper(nn.Module):
    """
    Wrapper exposing the interface used by training loop:
      forward(x) -> segmentation_mask (B,1,H,W), features (B, F)
    For simplicity the features are global pooled encoder features.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        if SMP_AVAILABLE:
            # try a swin-based encoder from smp: if not available fallback
            try:
                self.model = smp.Unet(encoder_name="swin_tiny_patch4_window7_224", encoder_weights="imagenet", in_channels=1, classes=1)
            except Exception:
                # fallback to simple UNET-like architecture
                self.model = None
        else:
            self.model = None

        if self.model is None:
            # fallback: small conv-based segmentation net
            self.encoder = nn.Sequential(
                nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64,32,2,stride=2), nn.ReLU(),
                nn.Conv2d(32,1,1), nn.Sigmoid()
            )
    def forward(self, x):
        # x: (B,1,H,W)
        if self.model is not None:
            mask = self.model(x)
            # get features by global pooling of encoder features if available
            # smp models expose encoder but we'll just compute mean over mask area as a crude feature
            feat = torch.nn.functional.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), -1)
            return mask, feat
        else:
            f = self.encoder(x)
            mask = self.decoder(f)
            feat = torch.nn.functional.adaptive_avg_pool2d(f, (1,1)).view(x.size(0), -1)
            return mask, feat

###############################################################################
# DQNN (quantum-inspired) module
# - The paper describes quantum perceptrons and unitaries.
# - Here we provide a parameterized, differentiable "quantum-inspired" module that maps segmentation features -> a feature vector U and an embedding V.
# - If you want full quantum circuits, replace this module with a PennyLane/Qiskit implementation.
###############################################################################

class DQNNInspired(nn.Module):
    def __init__(self, in_dim, out_dim=64, embed_dim=128):
        """
        in_dim: input feature vector size (e.g., pooled segmentation features)
        out_dim: low-dim classifier output (U)
        embed_dim: quantum feature vector V
        """
        super().__init__()
        # parameterized layers emulate parameterized unitaries / perceptrons
        self.perceptron_stack = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU()
        )
        # output head U
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.Tanh()  # act like measurement outputs in [-1,1]
        )
    def forward(self, x):
        # x: (B, in_dim)
        v = self.perceptron_stack(x)
        u = self.output_head(v)
        return u, v

###############################################################################
# SpinalNet implementation (paper description)
# - Splits input into segments and processes sequentially, producing local outputs that are combined.
# - This implementation follows the SpinalNet idea from the literature and paper description.
###############################################################################

class SpinalNet(nn.Module):
    def __init__(self, in_features, segment_size=32, hidden_per_segment=64, n_classes=2):
        super().__init__()
        assert in_features % segment_size == 0, "in_features must be divisible by segment_size"
        self.segment_size = segment_size
        self.n_segments = in_features // segment_size
        # For each segment we create an input->intermediate block and an intermediate->output block
        self.interm_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(segment_size + (0 if i==0 else hidden_per_segment), hidden_per_segment),
                          nn.ReLU())
            for i in range(self.n_segments)
        ])
        # collect outputs
        self.out_layer = nn.Sequential(
            nn.Linear(self.n_segments * hidden_per_segment, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, in_features)
        B = x.shape[0]
        segments = x.view(B, self.n_segments, self.segment_size)
        inter_outputs = []
        prev = None
        for i in range(self.n_segments):
            seg = segments[:, i, :]
            if i == 0:
                inp = seg
            else:
                inp = torch.cat([seg, prev], dim=1)
            out = self.interm_layers[i](inp)
            inter_outputs.append(out)
            prev = out
        # flatten intermediate outputs and pass through out_layer
        concat = torch.cat(inter_outputs, dim=1)
        logits = self.out_layer(concat)
        return logits

###############################################################################
# Full Q-SpinalNet Module integrating DQNN + SpinalNet
###############################################################################

class QSpinalNetModel(nn.Module):
    def __init__(self, feature_dim, spinal_segment_size=32, n_classes=2):
        super().__init__()
        self.dqnn = DQNNInspired(in_dim=feature_dim, out_dim=64, embed_dim=spinal_segment_size * (feature_dim // spinal_segment_size + 1))
        # We'll create the SpinalNet input by concatenating U and V and flattening/truncating/padding to desired size
        spinal_input_dim = 64 + 128  # U + V (from dqnn default)
        # adjust spinal_input_dim to be divisible by segment_size
        seg_size = spinal_segment_size
        pad = (seg_size - (spinal_input_dim % seg_size)) % seg_size
        self.pad = pad
        self.spinal = SpinalNet(in_features=spinal_input_dim + pad, segment_size=seg_size, hidden_per_segment=64, n_classes=n_classes)

    def forward(self, feat):
        # feat: (B, feature_dim)
        u, v = self.dqnn(feat)
        x = torch.cat([u, v], dim=1)
        if self.pad > 0:
            x = torch.cat([x, torch.zeros(x.shape[0], self.pad, device=x.device)], dim=1)
        logits = self.spinal(x)
        return logits

###############################################################################
# Dataset skeleton for DDSM/CBIS-DDSM images (you must point paths to your DICOM/JPEG images)
###############################################################################
class MammogramDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None, preprocess=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        if self.preprocess:
            img = triple_preprocess(img)
        # normalize to tensor
        img = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)
        if self.mask_paths:
            mask = load_image(self.mask_paths[idx])
            mask = (mask > 127).astype(np.float32)
            mask_t = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask_t = torch.zeros(1, img.shape[0], img.shape[1], dtype=torch.float32)
        label = 0  # placeholder: load actual label mapping externally
        return img_t, mask_t, label

###############################################################################
# Training loop
###############################################################################
def train_one_epoch(seg_model, q_model, dataloader, optim, device):
    seg_model.train()
    q_model.train()
    ce_loss = nn.BCELoss()
    ce_cls = nn.CrossEntropyLoss()
    total_loss = 0.0
    for imgs, masks, labels in tqdm(dataloader, desc="train"):
        imgs = imgs.to(device)
        masks = masks.to(device)
        labels = labels.to(device, dtype=torch.long)
        optim.zero_grad()
        pred_mask, feat = seg_model(imgs)  # mask + features
        # flatten or pool features to a vector
        if feat.dim() == 4:
            feat_vec = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1)).view(feat.size(0), -1)
        else:
            feat_vec = feat
        logits = q_model(feat_vec)
        # segmentation loss + classification loss
        s_loss = ce_loss(pred_mask, masks)
        c_loss = ce_cls(logits, labels)
        loss = s_loss + 0.5 * c_loss
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(seg_model, q_model, dataloader, device):
    seg_model.eval()
    q_model.eval()
    import sklearn.metrics as skm
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, masks, labels in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device, dtype=torch.long)
            pred_mask, feat = seg_model(imgs)
            if feat.dim() == 4:
                feat_vec = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1)).view(feat.size(0), -1)
            else:
                feat_vec = feat
            logits = q_model(feat_vec)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    acc = skm.accuracy_score(all_labels, all_preds)
    return {"accuracy":acc}

###############################################################################
# Example main to run training
###############################################################################
def main_example():
    # example file lists -- replace with real file paths and labels
    images = ["data/img1.png", "data/img2.png"]  # replace
    masks = ["data/mask1.png", "data/mask2.png"]  # replace or None
    ds = MammogramDataset(images, masks, preprocess=True)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = SegmentationModelWrapper().to(device)
    # example feature dim depends on seg output; we'll assume 64
    qmodel = QSpinalNetModel(feature_dim=64, spinal_segment_size=32, n_classes=2).to(device)

    optimizer = torch.optim.Adam(list(seg.parameters()) + list(qmodel.parameters()), lr=1e-4)
    for epoch in range(3):
        loss = train_one_epoch(seg, qmodel, dl, optimizer, device)
        print("Epoch", epoch, "loss", loss)
        print("Eval:", evaluate(seg, qmodel, dl, device))

if __name__ == "__main__":
    main_example()
