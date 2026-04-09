# seg-baseline

A lightweight, config-driven baseline for 2D medical image segmentation.
Built on [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch), [PyTorch Lightning](https://lightning.ai/), and [Hydra](https://hydra.cc/).

**Key features**

- **Config-driven**: swap model / backbone / optimizer / augmentation / all hyperparameters from the command line — no code changes needed.
- **Multi-class & binary segmentation**: both supported through a single code path.
- **20+ architectures × 100+ encoders**: UNet, UNet++, DeepLabV3+, FPN, MAnet, … with ResNet, EfficientNet, MiT, … backbones.
- **Automatic metric logging**: per-class and mean Dice / IoU saved to TensorBoard and W&B.
- **Flexible evaluation**: evaluate any split (train / val / test) with a single flag; results saved as TensorBoard events and CSV.
- **Multi-GPU / DDP**: configurable via `hardware.*` fields or command-line overrides.

**Project structure**

```
seg-baseline/
├── configs/               ← all hyperparameters live here (no code edits needed)
│   ├── train.yaml         ← main config
│   ├── model/             ← architecture + encoder configs
│   ├── dataset/           ← dataset root, mask format, class definitions
│   ├── augmentation/      ← light / medium / heavy pipelines
│   └── optimizer/         ← optimizer + scheduler configs
├── src/
│   ├── datasets/
│   │   └── seg_dataset.py      ← SegDataset + SegDataModule
│   ├── models/
│   │   └── seg_module.py       ← Lightning module (model + loss + metrics)
│   └── transforms/
│       └── build_transforms.py ← augmentation pipeline builder
├── train.py               ← training entry point
├── evaluate.py            ← evaluation on any split (train / val / test)
├── predict.py             ← single-image inference + visualisation
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `pytorch-lightning`, `segmentation-models-pytorch`, `albumentations`, `hydra-core`, `torchmetrics`

---

## Dataset Structure

```
your-dataset/
├── train/
│   ├── images/          ← RGB images (.png / .jpg)
│   └── <mask_dir>/      ← segmentation masks (.png)
├── val/
│   ├── images/
│   └── <mask_dir>/
└── test/
    ├── images/
    └── <mask_dir>/
```

**Mask formats**

| `mask_mode` | pixel values | use case |
|---|---|---|
| `index` | 0, 1, 2, … (class indices) | multi-class semantic segmentation |
| `binary` | 0 or 255 → auto-converted to 0/1 | binary segmentation |

---

## Training

### Basic usage

```bash
python train.py
```

Trains with default settings (UNet + ResNet34, demo dataset). Results and checkpoints are saved under `outputs/`.

### Common overrides

```bash
# Change architecture / encoder
python train.py model=unetplusplus
python train.py model=deeplabv3plus model.encoder=resnet50
python train.py model=fpn model.encoder=efficientnet-b4

# Change training hyperparameters
python train.py train.batch_size=4 train.epochs=200 train.img_size=640
python train.py optimizer=sgd optimizer.lr=1e-2
python train.py augmentation=heavy

# Use your own dataset
python train.py dataset=my_dataset
```

### Multi-GPU / DDP

```bash
# All available GPUs
python train.py hardware.devices=4 hardware.strategy=ddp

# Specific GPUs
python train.py 'hardware.devices=[0,1]' hardware.strategy=ddp

# Mixed-precision + DDP (recommended for speed)
python train.py hardware.devices=4 hardware.strategy=ddp train.precision=16-mixed
```

> **Note**: With DDP, each process spawns its own DataLoader workers. Reduce `train.num_workers` if you see high memory or CPU usage.

### Hyperparameter sweep (Hydra multirun)

```bash
python train.py --multirun \
    model=unet,unetplusplus,deeplabv3plus \
    model.encoder=resnet34,resnet50
```

### Custom top-level config

```bash
cp configs/train.yaml configs/experiment_v2.yaml
# edit configs/experiment_v2.yaml as needed, then:
python train.py --config-name=experiment_v2
python train.py --config-name=experiment_v2 model=deeplabv3plus train.epochs=200
```

---

## Evaluation

Evaluate a saved checkpoint on any dataset split:

```bash
# Test set (default)
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt

# Validation set
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt split=val

# Training set
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt split=train
```

Prints a metric table (per-class and mean Dice / IoU). Results are also saved automatically under `outputs/<run>_eval_<split>/` as TensorBoard events and `metrics.csv`.

---

## Single-Image Inference

```bash
python predict.py \
    --img path/to/image.png \
    --checkpoint outputs/<name>/checkpoints/best.ckpt \
    --out result.png
```

Produces a side-by-side visualisation of the input image and the colour-coded prediction mask.

---

## Configuration Reference

### Adding a new dataset

1. Copy the appropriate template and rename it (e.g. `configs/dataset/my_dataset.yaml`):

**Multi-class** — copy `configs/dataset/multiclass.yaml`:

```yaml
name: my_dataset
root: /path/to/dataset
mask_dir: masks
mask_mode: index            # pixel values are class indices 0, 1, 2, …
num_classes: 3
class_names: [background, class_a, class_b]
foreground_classes: [1, 2]  # indices used for mean Dice/IoU (excludes background)
```

**Binary** — copy `configs/dataset/binaryclass.yaml`:

```yaml
name: my_dataset
root: /path/to/dataset
mask_dir: masks
mask_mode: binary           # pixel values {0, 255} → auto-converted to {0, 1}
num_classes: 1
class_names: [foreground]
foreground_classes: [0]
```

2. Run:

```bash
python train.py dataset=my_dataset
```

No Python code changes needed.

### Key parameters

**`configs/train.yaml`**

| Parameter | Default | Description |
|---|---|---|
| `train.epochs` | 100 | Number of training epochs |
| `train.batch_size` | 8 | Batch size per GPU |
| `train.img_size` | 512 | Images resized to `img_size × img_size` |
| `train.num_workers` | 4 | DataLoader workers |
| `train.precision` | `"32-true"` | `"16-mixed"` for AMP |
| `train.early_stopping` | false | Enable early stopping |
| `split` | `test` | Split evaluated by `evaluate.py` (`train` / `val` / `test`) |
| `logging.logger` | `tensorboard` | `"tensorboard"` or `"wandb"` |
| `hardware.accelerator` | `auto` | `"auto"`, `"gpu"`, `"cpu"` |
| `hardware.devices` | `auto` | `"auto"`, integer count, or list e.g. `[0,1]` |
| `hardware.num_nodes` | `1` | Number of machines for multi-node training |
| `hardware.strategy` | `auto` | `"auto"`, `"ddp"`, `"fsdp"`, … |

**`configs/model/*.yaml`**

| Parameter | Example | Description |
|---|---|---|
| `model.arch` | `Unet` | Architecture name (see SMP docs) |
| `model.encoder` | `resnet34` | Encoder backbone |
| `model.encoder_weights` | `imagenet` | Pre-training (`null` for random init) |

**`configs/optimizer/*.yaml`**

| Parameter | Example | Description |
|---|---|---|
| `optimizer.lr` | `1e-4` | Learning rate |
| `optimizer.weight_decay` | `1e-4` | Weight decay |
| `scheduler.name` | `cosine` | `cosine`, `step`, `plateau`, `none` |

### Viewing logs

```bash
tensorboard --logdir outputs/
```

---

## Supported Architectures (via SMP)

`Unet` · `UnetPlusPlus` · `MAnet` · `Linknet` · `FPN` · `PSPNet` · `DeepLabV3` · `DeepLabV3Plus` · `PAN`

Full encoder list: https://smp.readthedocs.io/en/latest/encoders.html
