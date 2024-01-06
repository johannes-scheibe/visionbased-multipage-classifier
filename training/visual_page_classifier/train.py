import json
from pathlib import Path

import pytorch_lightning as pl
from visual_page_classifier.lightning_module import VisualPageClassifierPLModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from multipage_classifier.visual_page_classifier import (
    VisualPageClassifier,
    VisualPageClassifierConfig,
)

from multipage_classifier.datasets.mosaic_dataset import MosaicDataModule


NAME = "visual_page_classifier"

DATASET_PATH = "/data/training/master_thesis/datasets/2023-05-23"
CLASS_PATH = "/data/training/master_thesis/datasets/bzuf_classes.json"
LIGHTNING_PATH = "/data/training/master_thesis/evaluation_logs"

N_EPOCHS = 25
MAX_PAGES = 32
NUM_WORKERS = 8
BATCH_SIZE = 1

IMAGE_SIZE = (704, 512)
PRETRAINED_ENCODER = "/data/training/master_thesis/evaluation_logs/swin_encoder/microsoft/swinv2-base-patch4-window8-256/version_1/"

if __name__ == "__main__":
    # BZUF classes
    classes = [c for c in json.load(open(CLASS_PATH))]

    config = VisualPageClassifierConfig(
        num_classes=len(classes),
        max_pages=MAX_PAGES,
        pretrained_encoder=str(Path(PRETRAINED_ENCODER) / "model.path"),
        detached=True,
    )

    model = VisualPageClassifierPLModule(config)

    data_module = MosaicDataModule(
        Path(DATASET_PATH),
        classes,
        model.classifier.encoder.page_encoder.prepare_input,
        batch_size=BATCH_SIZE,
        max_pages=MAX_PAGES,
        num_workers=NUM_WORKERS,
    )

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
        precision=32,
    )

    trainer.fit(model, data_module)