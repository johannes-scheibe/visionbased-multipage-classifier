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
from transformers import DonutSwinModel, Swinv2Model, SwinModel



torch.cuda.empty_cache()

NAME = "page_comparision_encoder"

DATASET_PATH = "/data/training/master_thesis/datasets/2023-05-23"
CLASS_PATH = "/data/training/master_thesis/datasets/bzuf_classes.json"
LIGHTNING_PATH = "/data/training/master_thesis/evaluation_logs/" + NAME

MAX_PAGES = 6
BATCH_SIZE = 1
NUM_WORKERS = 4

N_EPOCHS = 25

PRETRAINED_MODEL_TYPE = Swinv2Model
MODEL_FOLDER = ""
PRETRAINED_MODEL = "microsoft/swinv2-base-patch4-window8-256"

IMAGE_SIZE = (704, 512) # height, width TODO 1024, 1408

if __name__ == "__main__":

    classes = [c for c in json.load(open(CLASS_PATH))]

    # Define encoder
    swin_config = DualClassifierConfig(
        num_classes=len(classes),
        max_page_nr= 96,
        encoder_cfg=SwinEncoderConfig(
            image_size=IMAGE_SIZE, 
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            pretrained_model_type=PRETRAINED_MODEL_TYPE
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

    logger = TensorBoardLogger(LIGHTNING_PATH, name=PRETRAINED_MODEL)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[3],
        logger=logger,
        max_epochs=N_EPOCHS,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=2,
        precision=32,
    )
    
    trainer.fit(encoder_module, data_module)
