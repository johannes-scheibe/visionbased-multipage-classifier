from pathlib import Path
import torch
import pytorch_lightning as pl
from lightning_module import SwinEncoderPLModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from multipage_classifier.datasets.ucsf_dataset import UCSFDataModule
from multipage_classifier.encoder.swin_encoder import SwinEncoderConfig
from lightning_module import SwinEncoderPLModule

torch.cuda.empty_cache()

NAME = "swin_encoder"
DATASET_PATH = "../dataset/ucsf-idl-resized-without_emails/"
DS_FILE = "IDL-less_20_pages.json"

N_EPOCHS = 50
MAX_PAGES = 4
NUM_WORKERS = 8

IMAGE_SIZE = (512, 704)


# Define encoder
swin_config = SwinEncoderConfig(
    patch_size=[4, 4],
    embed_dim=128,
    depths=[2, 2, 14, 2],
    num_heads=[4, 8, 16, 32],
    window_size=[10, 10],
)

encoder = SwinEncoderPLModule(swin_config)

# Define preprocessor
from multipage_classifier.preprocessor import ImageProcessor

image_processor = ImageProcessor(img_size=IMAGE_SIZE)

data_module = UCSFDataModule(
    Path(DATASET_PATH),
    DS_FILE,
    prepare_function=image_processor.prepare_input,
    split=[0.8, 0.2],
    max_pages=MAX_PAGES,
    num_workers=NUM_WORKERS,
)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

# Configure checkpointing
checkpoint_callback = ModelCheckpoint(
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
    save_last=True,
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "checkpoint-{epoch:02d}-{val_loss:.4f}"

logger = TensorBoardLogger("lightning_logs", name=NAME)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    logger=logger,
    max_epochs=N_EPOCHS,
    callbacks=[checkpoint_callback],
    num_sanity_val_steps=0
)

trainer.fit(encoder, data_module)
