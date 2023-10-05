import json
import os
from pathlib import Path
from multipage_classifier.encoder.swin_encoder import SwinEncoderConfig

import pytorch_lightning as pl
import torch
from multipage_classifier.datasets.mosaic_dataset import MosaicDataModule
from multipage_classifier.dual_classifier import DualClassifierConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dual_classifier.lightning_module import DualClassifierPLModule



torch.cuda.empty_cache()

NAME = "dual_classifier"

DATASET_PATH = "/data/training/master_thesis/datasets/2023-05-23"
CLASS_PATH = "/data/training/master_thesis/datasets/bzuf_classes.json"
LIGHTNING_PATH = "/data/training/master_thesis/lightning_logs"

MAX_PAGES = 8
BATCH_SIZE = 1
NUM_WORKERS = 4

N_EPOCHS = 30

PRETRAINED_MODEL = "/data/training/master_thesis/models/donut-encoder/"
IMAGE_SIZE = (704, 512) # height, width TODO 1024, 1408

if __name__ == "__main__":

    classes = [c for c in json.load(open(CLASS_PATH))]

    # Define encoder
    swin_config = DualClassifierConfig(
        num_classes=len(classes),
        max_page_nr= 96,
        encoder_cfg=SwinEncoderConfig(
            image_size=IMAGE_SIZE, 
            pretrained_model_name_or_path=PRETRAINED_MODEL
        )
    )
    encoder_module = DualClassifierPLModule(swin_config)

    # Define data module
    data_module = MosaicDataModule(Path(DATASET_PATH), classes, encoder_module.model.encoder.prepare_input, batch_size=BATCH_SIZE, max_pages=MAX_PAGES, num_workers=NUM_WORKERS)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val/loss",
        mode="min",
        save_last=True,

    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "checkpoint-{epoch:02d}-{val_loss:.4f}"

    logger = TensorBoardLogger(LIGHTNING_PATH, name=NAME)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[2],
        logger=logger,
        max_epochs=N_EPOCHS,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        precision=16
    )
    
    trainer.fit(encoder_module, data_module)
