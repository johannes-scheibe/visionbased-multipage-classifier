import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from multipage_classifier.datasets.mosaic_dataset import MosaicDataModule
from multipage_classifier.encoder.vision_encoder import VisionEncoderConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from vision_encoder.lightning_module import VisionEncoderPLModule


torch.cuda.empty_cache()

NAME = "vision_encoder"

DATASET_PATH = "/data/training/master_thesis/datasets/2023-05-23"
CLASS_PATH = "/data/training/master_thesis/datasets/bzuf_classes.json"
LIGHTNING_PATH = "/data/training/master_thesis/evaluation_logs/" + NAME

MAX_PAGES = 4
BATCH_SIZE = 1
NUM_WORKERS = 8

N_EPOCHS = 25

MODEL_FOLDER = ""
PRETRAINED_MODEL = "google/vit-base-patch16-224"

IMAGE_SIZE = (704, 512)

if __name__ == "__main__":
    classes = [c for c in json.load(open(CLASS_PATH))]

    # Define encoder
    swin_config = VisionEncoderConfig(
        image_size=IMAGE_SIZE,
        pretrained_model_name_or_path=str(Path(MODEL_FOLDER) / PRETRAINED_MODEL),
    )
    encoder_module = VisionEncoderPLModule(swin_config)

    # Define data module
    data_module = MosaicDataModule(
        Path(DATASET_PATH),
        classes,
        encoder_module.encoder.prepare_input,
        batch_size=BATCH_SIZE,
        max_pages=MAX_PAGES,
        num_workers=NUM_WORKERS,
    )

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

    logger = TensorBoardLogger(LIGHTNING_PATH, name=PRETRAINED_MODEL)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[3],
        logger=logger,
        max_epochs=N_EPOCHS,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        precision=32,
    )

    trainer.fit(encoder_module, data_module)
