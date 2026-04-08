"""
Evaluate a saved checkpoint on any dataset split and print a metric table.

Usage:
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt split=val
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt split=train
    python evaluate.py checkpoint=outputs/xxx/checkpoints/best.ckpt dataset=optic split=test
"""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from src.datasets.seg_dataset import SegDataModule
from src.models.seg_module import SegModule


@hydra.main(version_base=None, config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.get('checkpoint', None)
    if ckpt_path is None:
        raise ValueError(
            'Please provide a checkpoint path:\n'
            '  python evaluate.py checkpoint=outputs/.../best.ckpt'
        )

    split = cfg.get('split', 'test')
    datamodule = SegDataModule(cfg, eval_split=split)

    # Load model weights from checkpoint; architecture is defined by cfg
    model = SegModule.load_from_checkpoint(ckpt_path, cfg=cfg)

    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        num_nodes=cfg.hardware.num_nodes,
        strategy=cfg.hardware.strategy,
        logger=False,
    )
    results = trainer.test(model, datamodule=datamodule)

    print(f'\n── Results ({split}) ──────────────────────')
    for k, v in results[0].items():
        print(f'  {k:<30} {v:.4f}')


if __name__ == '__main__':
    main()
