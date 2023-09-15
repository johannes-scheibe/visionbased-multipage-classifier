from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import ModuleDict
from torchmetrics import Accuracy, ConfusionMatrix, Metric, MetricCollection

from pytorch_lightning.loggers import TensorBoardLogger

class Mode(Enum):
    TRAIN = "train"
    VALID = "val"
    TEST = "test"


class BaseLightningModule(pl.LightningModule):
    metrics: ModuleDict
    confmat: ModuleDict

    mode: Mode = Mode.TRAIN

    logger: TensorBoardLogger
    def __init__(self) -> None:
        super().__init__()

        self.init_metrics()
        

    def init_metrics(self):
        metrics = torch.nn.ModuleDict({})
        confmat = torch.nn.ModuleDict({})
        for mode in [m.value for m in Mode]:
            metrics[f"_{mode}"] = torch.nn.ModuleDict({})
            confmat[f"_{mode}"] = torch.nn.ModuleDict({})

        self.metrics = metrics
        self.confmat = confmat

    def add_metrics(self, key: str, metric: MetricCollection, mode: Mode):
        self.metrics[f"_{mode.value}"][key] = metric # type: ignore

    def add_confmat(self, key: str, matrix: Metric, mode: Mode):
        self.confmat[f"_{mode.value}"][key] = matrix # type: ignore

    def set_default_metrics(
        self,
        key: str,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: int | None = None,
        num_labels: int | None= None,
        confmat: bool = True
    ) -> None:
        for mode in Mode:
            n = num_classes or num_labels 
            assert n
            top_k = min(n - 1, 3)

            self.add_metrics(key, MetricCollection(
                {
                    "acc": Accuracy(task=task, num_classes=num_classes, num_labels=num_labels),
                    f"top{top_k}_acc": Accuracy(
                        task=task, num_classes=num_classes, num_labels=num_labels, top_k=top_k
                    ),
                },
                postfix=f"_{key}",
            ), mode)
            if confmat:
                self.add_confmat(key, ConfusionMatrix(
                    task=task, num_labels=num_labels, num_classes=num_classes, normalize="true"
                ), mode)

    @abstractmethod
    def step(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """Override this method"""

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        self.mode = Mode.TRAIN
        preds, gt, losses = self.step(batch, batch_idx)
        self.update_metrics(preds, gt)
        return self.log_losses(losses)

    def on_training_epoch_end(self) -> None:
        self.mode = Mode.TRAIN
        self.shared_epoch_end()

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        self.mode = Mode.VALID
        preds, gt, losses = self.step(batch, batch_idx)
        self.update_metrics(preds, gt)
        return self.log_losses(losses)

    def on_validation_epoch_end(self) -> None:
        self.mode = Mode.VALID
        self.shared_epoch_end()

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        self.mode = Mode.TEST
        preds, gt, losses = self.step(batch, batch_idx)
        self.update_metrics(preds, gt)
        return self.log_losses(losses)

    def on_test_epoch_end(self) -> None:
        self.mode = Mode.TEST
        self.shared_epoch_end()

    def shared_epoch_end(self) -> None:
        for k, m in self.metrics[f"_{self.mode.value}"].items(): # type: ignore
            self.log_metrics(m.compute())
            m.reset()
        for k, c in self.confmat[f"_{self.mode.value}"].items(): # type: ignore
            fig_ = self.get_confusion_matrix(
                c.compute().cpu().data.numpy(), c.num_classes
            )
            self.logger.experiment.add_figure(
                f"{self.mode.value}_confmat_{k}",
                fig_,
                self.current_epoch,
            )
            c.reset()

    def get_confusion_matrix(self, cf_matrix, num_classes):
        fig, ax = plt.subplots(
            figsize=(max(num_classes * 0.25, 5), max((num_classes * 0.25, 5)))
        )
        ConfusionMatrixDisplay(cf_matrix).plot(
            ax=ax,
            include_values=False,
            xticks_rotation="vertical",
        )
        plt.close()
        return fig

    def update_metrics(self, pred: Dict, gt):
        for k, m in self.metrics[f"_{self.mode.value}"].items(): # type: ignore
            m.update(pred[k], gt[k])
        for k, c in self.confmat[f"_{self.mode.value}"].items(): # type: ignore
            c.update(pred[k], gt[k])

    def log_metrics(self, metrics: Dict):
        self.log_dict(
            {f"{self.mode.value}/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            logger=True,
        )

    def log_losses(self, losses: Union[Dict, torch.Tensor]) -> Dict[str, Any]:
        if not isinstance(losses, Dict):
            losses = {"loss": losses}

        if "loss" in losses and len(losses) > 1:
            raise Exception(
                "Either return a dictionary of multiply losses or just one loss and add the rest to metrics"
            )

        if "loss" not in losses:
            loss = sum(losses.values())
            losses["loss"] = loss

        self.log_dict(
            {f"{self.mode.value}/{k}": v for k, v in losses.items()},
            on_step=self.mode == Mode.TRAIN,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        return losses
