"""
Run inference on a single image and save a side-by-side visualisation.

Usage:
    python predict.py --img demo-dataset/test/images/xxx.png \
                      --checkpoint outputs/.../best.ckpt \
                      [--img_size 512] \
                      [--out result.png]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')           # headless-safe backend
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf

from src.models.seg_module import SegModule

# Colour palette for multi-class overlay (one colour per class)
_PALETTE = [
    [0,   0,   0  ],   # 0 — background (black)
    [0,   200, 83 ],   # 1 — class 1    (green)
    [255, 152, 0  ],   # 2 — class 2    (orange)
    [33,  150, 243],   # 3 — class 3    (blue)
    [156, 39,  176],   # 4 — class 4    (purple)
    [244, 67,  54 ],   # 5 — class 5    (red)
]


def colorise(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a class-index mask (H×W) to an RGB image."""
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        colour = _PALETTE[cls % len(_PALETTE)]
        rgb[mask == cls] = colour
    return rgb


def predict(img_path: str, ckpt_path: str, img_size: int = 512,
            out_path: str = None, device: str = 'cpu') -> None:

    # ── Load model ────────────────────────────────────────────────────────────
    model = SegModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval().to(device)

    # Read saved cfg from checkpoint
    saved_cfg  = OmegaConf.create(model.hparams['cfg'])
    num_classes = model.num_classes
    class_names = model.class_names

    # ── Pre-process ───────────────────────────────────────────────────────────
    image = np.array(Image.open(img_path).convert('RGB'))
    h_orig, w_orig = image.shape[:2]

    tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=list(saved_cfg.augmentation.normalize.mean),
            std=list(saved_cfg.augmentation.normalize.std),
        ),
        ToTensorV2(),
    ])
    tensor = tf(image=image)['image'].unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        if model._is_binary:
            pred = (torch.sigmoid(logits).squeeze() > 0.5).cpu().numpy().astype(np.uint8)
        else:
            pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Resize prediction back to original image size
    pred_pil  = Image.fromarray(pred).resize((w_orig, h_orig), resample=Image.NEAREST)
    pred_orig = np.array(pred_pil)

    # ── Visualise ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=13)
    axes[0].axis('off')

    if num_classes > 1:
        overlay = colorise(pred_orig, num_classes)
        axes[1].imshow(overlay)
        # Legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(color=[c / 255 for c in _PALETTE[i % len(_PALETTE)]], label=name)
            for i, name in enumerate(class_names)
        ]
        axes[1].legend(handles=legend_patches, loc='lower right', fontsize=9)
    else:
        axes[1].imshow(pred_orig, cmap='gray', vmin=0, vmax=1)

    axes[1].set_title('Prediction', fontsize=13)
    axes[1].axis('off')

    plt.tight_layout()

    if out_path is None:
        out_path = str(Path(img_path).stem) + '_pred.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] Prediction saved to: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single-image inference')
    parser.add_argument('--img',        required=True,          help='Input image path')
    parser.add_argument('--checkpoint', required=True,          help='Checkpoint .ckpt path')
    parser.add_argument('--img_size',   type=int, default=512,  help='Resize side for inference')
    parser.add_argument('--out',        default=None,           help='Output image path')
    parser.add_argument('--device',     default='cpu',          help='cpu / cuda / mps')
    args = parser.parse_args()

    predict(args.img, args.checkpoint, args.img_size, args.out, args.device)
