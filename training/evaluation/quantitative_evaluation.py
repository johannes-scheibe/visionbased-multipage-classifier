import json
from pathlib import Path

import pytorch_lightning as pl
from page_comparsion_encoder.lightning_module import PageComparisonEncoderPLModule
from multipage_transformer.lightning_module import (
    MultipageTransformerPLModule,
    MultipagePLDataModule,
)
from visual_page_classifier.lightning_module import VisualPageClassifierPLModule

from swin_encoder.lightning_module import SwinEncoderPLModule, ORDER_NAMES
from vision_encoder.lightning_module import VisionEncoderPLModule, ORDER_NAMES

from vision_encoder.lightning_module import VisionEncoderPLModule

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from multipage_classifier.datasets.mosaic_dataset import MosaicDataModule

LIGHTNING_PATH = "/data/training/master_thesis/testing_logs"
MODEL_PATH = "/data/training/master_thesis/evaluation_logs/classifiers/visual_page_classifier/version_0/checkpoints/best-checkpoint.ckpt"

MODEL_NAME = MODEL_PATH.split("/")[5]
print(MODEL_NAME)

# Load Model
model: VisualPageClassifierPLModule = VisualPageClassifierPLModule.load_from_checkpoint(
    MODEL_PATH, map_location="cpu"
)
model.metric_labels["order"] = ORDER_NAMES

DATASET_PATH = "/data/training/master_thesis/datasets/2023-05-23"
CLASS_PATH = "/data/training/master_thesis/datasets/bzuf_classes.json"

TASK_PROMPT = "<s_classification>"
N_EPOCHS = 25
MAX_PAGES = 32
NUM_WORKERS = 8
BATCH_SIZE = 1

IMAGE_SIZE = (704, 512)

# BZUF classes
classes = [c for c in json.load(open(CLASS_PATH))]

model.metric_labels["doc_class"] = classes

data_module = MosaicDataModule(
    Path(DATASET_PATH),
    classes,
    model.classifier.encoder.page_encoder.prepare_input,
    batch_size=BATCH_SIZE,
    max_pages=MAX_PAGES,
    num_workers=NUM_WORKERS,
)
# data_module = MultipagePLDataModule(
#     Path(DATASET_PATH), model.model, task_prompt=TASK_PROMPT, num_workers=NUM_WORKERS
# )

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

logger = TensorBoardLogger(LIGHTNING_PATH, name=MODEL_NAME)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[3],
    logger=logger,
    max_epochs=N_EPOCHS,
    callbacks=[checkpoint_callback],
    num_sanity_val_steps=0,
    precision=32,
)
print("Testing", MODEL_NAME, MODEL_PATH)
trainer.validate(model, data_module)
