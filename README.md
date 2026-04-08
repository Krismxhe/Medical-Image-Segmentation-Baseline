# seg-baseline

A lightweight, config-driven baseline for 2D medical image segmentation.
Built on [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch), [PyTorch Lightning](https://lightning.ai/), and [Hydra](https://hydra.cc/).

---

## Features

- **Config-driven**: change model / backbone / optimizer / augmentation / all hyperparameters by editing YAML or passing command-line args — no code changes needed.
- **Multi-class & binary segmentation**: both are supported through a single code path.
- **20+ architectures × 100+ encoders**: UNet, UNet++, DeepLabV3+, FPN, MAnet, … with ResNet, EfficientNet, MiT, … backbones.
- **Automatic metric logging**: per-class and mean Dice / IoU, logged to TensorBoard or W&B.
- **Multi-GPU / DDP training**: configurable via `hardware.*` fields in `train.yaml` or command-line overrides.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `pytorch-lightning`, `segmentation-models-pytorch`, `albumentations`, `hydra-core`, `torchmetrics`

---

## Dataset Structure

Organise your dataset as follows (same as the demo dataset):

```
your-dataset/
├── train/
│   ├── images/          ← RGB images (.png / .jpg)
│   └── masks/           ← segmentation masks (.png)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Mask formats**

| `mask_mode` | pixel values | use case |
|---|---|---|
| `index` | 0, 1, 2, … (class indices) | multi-class semantic segmentation |
| `binary` | 0 or 255 → auto-converted to 0/1 | binary segmentation |

---

## Quick Start

### 1. Train with defaults (UNet + ResNet34, optic disc dataset)

```bash
python train.py
```

### 2. Change model architecture

```bash
python train.py model=unetplusplus
python train.py model=deeplabv3plus model.encoder=resnet50
python train.py model=fpn model.encoder=efficientnet-b4
```

### 3. Change training hyperparameters

```bash
python train.py train.batch_size=4 train.epochs=200 train.img_size=640
python train.py optimizer=sgd optimizer.lr=1e-2
python train.py augmentation=heavy
```

### 4. Switch to binary segmentation (optic disc only)

```bash
python train.py dataset.mask_dir=masks_od dataset.mask_mode=binary \
                dataset.num_classes=1 dataset.class_names=[background,optic_disc] \
                dataset.foreground_classes=[1] \
                checkpoint.monitor=val/dice_mean
```

### 5. Sweep multiple configs (Hydra multirun)

```bash
python train.py --multirun \
    model=unet,unetplusplus,deeplabv3plus \
    model.encoder=resnet34,resnet50
```

Results are saved under `outputs/<experiment_name>/`.

---

## Evaluate on Test Set

```bash
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt
```

---

## Single-Image Inference

```bash
python predict.py \
    --img demo-dataset/test/images/14010410123_20170329164027156.png \
    --checkpoint outputs/<name>/checkpoints/best.ckpt \
    --out result.png
```

Produces a side-by-side visualisation of the input image and the colour-coded prediction mask.

---

## Configuration Reference

### Directory layout

```
configs/
├── train.yaml              ← main config (edit epochs, batch_size, img_size, etc.)
├── model/
│   ├── unet.yaml           ← arch + encoder + weights
│   ├── unetplusplus.yaml
│   ├── deeplabv3plus.yaml
│   ├── fpn.yaml
│   └── manet.yaml
├── dataset/
│   ├── optic.yaml          ← root path, mask_dir, num_classes, class names
│   └── custom.yaml         ← template for your own dataset
├── augmentation/
│   ├── light.yaml
│   ├── medium.yaml         ← default
│   └── heavy.yaml
└── optimizer/
    ├── adamw.yaml          ← default
    ├── adam.yaml
    └── sgd.yaml
```

### Key parameters in `configs/train.yaml`

| Parameter | Default | Description |
|---|---|---|
| `train.epochs` | 100 | Number of training epochs |
| `train.batch_size` | 8 | Batch size per GPU |
| `train.img_size` | 512 | Images resized to `img_size × img_size` |
| `train.num_workers` | 4 | DataLoader workers |
| `train.precision` | `"32-true"` | `"16-mixed"` for AMP |
| `train.early_stopping` | false | Enable early stopping |
| `logging.logger` | `tensorboard` | `"tensorboard"` or `"wandb"` |
| `hardware.devices` | `auto` | Number of GPUs or list of GPU ids |
| `hardware.strategy` | `auto` | DDP strategy (`"ddp"`, `"fsdp"`, …) |
| `hardware.num_nodes` | `1` | Number of machines for multi-node training |

### Key parameters in `configs/model/*.yaml`

| Parameter | Example | Description |
|---|---|---|
| `model.arch` | `Unet` | Architecture name (see SMP docs) |
| `model.encoder` | `resnet34` | Encoder backbone |
| `model.encoder_weights` | `imagenet` | Pre-training (`null` for random init) |

### Key parameters in `configs/optimizer/*.yaml`

| Parameter | Example | Description |
|---|---|---|
| `optimizer.lr` | `1e-4` | Learning rate |
| `optimizer.weight_decay` | `1e-4` | Weight decay |
| `scheduler.name` | `cosine` | `cosine`, `step`, `plateau`, `none` |

---

## Adding a New Dataset

1. Copy `configs/dataset/custom.yaml` and fill in your values:

```yaml
dataset:
  name: my_dataset
  root: /path/to/dataset
  mask_dir: masks
  mask_mode: index        # or "binary"
  num_classes: 3
  class_names: [background, class1, class2]
  foreground_classes: [1, 2]
```

2. Run:

```bash
python train.py dataset=my_dataset
```

No Python code changes needed — as long as your data follows the `train/val/test` + `images/masks` directory structure.

---

## Multi-GPU / DDP Training

Multi-GPU training is supported via PyTorch Lightning's DDP backend. The relevant parameters live under `hardware` in `configs/train.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `hardware.accelerator` | `auto` | `"auto"`, `"gpu"`, `"cpu"` |
| `hardware.devices` | `auto` | `"auto"`, integer count, or list e.g. `[0,1]` |
| `hardware.num_nodes` | `1` | Number of machines (for multi-node jobs) |
| `hardware.strategy` | `auto` | `"auto"`, `"ddp"`, `"ddp_find_unused_parameters_false"`, `"fsdp"` |

**Examples**

```bash
# Use all available GPUs with DDP
python train.py hardware.devices=4 hardware.strategy=ddp

# Use specific GPUs
python train.py 'hardware.devices=[0,1]' hardware.strategy=ddp

# Mixed-precision + DDP (recommended for speed)
python train.py hardware.devices=4 hardware.strategy=ddp train.precision=16-mixed
```

> **Note**: With DDP, each process spawns its own DataLoader workers. If you see high memory or CPU usage, reduce `train.num_workers` accordingly (e.g. `train.num_workers=2`).

---

## Using a Custom Main Config

By default `train.py` loads `configs/train.yaml`. To use a **different top-level config file** (e.g. `configs/experiment_retina.yaml`), pass Hydra's built-in `--config-name` flag:

```bash
# Create your own top-level config
cp configs/train.yaml configs/experiment_retina.yaml
# Edit configs/experiment_retina.yaml as needed, then run:
python train.py --config-name=experiment_retina
```

You can combine this with any command-line override as usual:

```bash
python train.py --config-name=experiment_retina model=deeplabv3plus train.epochs=200
```

---

## Viewing Training Curves

```bash
tensorboard --logdir outputs/
```

Then open `http://localhost:6006` in your browser.

---

## Supported Architectures (via SMP)

`Unet` · `UnetPlusPlus` · `MAnet` · `Linknet` · `FPN` · `PSPNet` · `DeepLabV3` · `DeepLabV3Plus` · `PAN`

Full encoder list: https://smp.readthedocs.io/en/latest/encoders.html

---

## Project Structure

```
seg-baseline/
├── configs/               ← all hyperparameters live here
├── src/
│   ├── datasets/
│   │   └── seg_dataset.py     ← SegDataset + SegDataModule
│   ├── models/
│   │   └── seg_module.py      ← Lightning module (model + loss + metrics)
│   └── transforms/
│       └── build_transforms.py ← augmentation pipeline builder
├── train.py               ← training entry point
├── evaluate.py            ← test-set evaluation
├── predict.py             ← single-image inference + visualisation
└── requirements.txt
```
