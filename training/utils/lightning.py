from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import ModuleDict
from torchmetrics import Accuracy, ConfusionMatrix, Metric, MetricCollection


class Mode(Enum):
    TRAIN = "train"
    VALID = "val"
    TEST = "test"


class BaseLightningModule(pl.LightningModule):
    metrics: ModuleDict
    confmat: ModuleDict

    mode: Mode = Mode.TRAIN

    def get_metrics(
        self,
        name: str,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: int,
    ) -> Tuple[ModuleDict, ModuleDict]:
        metrics = ModuleDict({})
        confusion_matrices = ModuleDict({})

        for mode in [m.value for m in Mode]:
            top_k = min(num_classes - 1, 3)

            curr_metrics = MetricCollection(
                {
                    "acc": Accuracy(task=task, num_classes=num_classes),
                    f"top{top_k}_acc": Accuracy(
                        task=task, num_classes=num_classes, top_k=top_k
                    ),
                },
                postfix=f"_{name}",
            )

            curr_confusion_matrix = ConfusionMatrix(
                task=task, num_classes=num_classes, normalize="true"
            )
            metrics[f"_{mode}"] = curr_metrics
            confusion_matrices[f"_{mode}"] = curr_confusion_matrix

        return metrics, confusion_matrices

    @abstractmethod
    def step(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """Override this method"""

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        self.mode = Mode.TRAIN
        pred, gt, losses = self.step(batch, batch_idx)
        self.update_metrics(pred, gt)
        return self.log_losses(losses)

    def training_epoch_end(self, outputs) -> None:
        self.mode = Mode.TRAIN
        self.shared_epoch_end(outputs)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        self.mode = Mode.VALID
        pred, gt, losses = self.step(batch, batch_idx)
        self.update_metrics(pred, gt)
        return self.log_losses(losses)

    def validation_epoch_end(self, outputs) -> None:
        self.mode = Mode.VALID
        self.shared_epoch_end(outputs)

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        self.mode = Mode.TEST
        pred, gt, losses = self.step(batch, batch_idx)
        self.update_metrics(pred, gt)
        return self.log_losses(losses)

    def test_epoch_end(self, outputs) -> None:
        self.mode = Mode.TEST
        self.shared_epoch_end(outputs)

    def shared_epoch_end(self, _) -> None:
        # Compute metrics:
        if hasattr(self, "metrics"):
            for k, v in self.metrics.items():
                metric: Metric = v[f"_{self.mode.value}"]
                self.log_metrics(metric.compute())
                metric.reset()

        # Compute confusion matrax
        if hasattr(self, "confmat"):
            for k, v in self.confmat.items():
                confmat: Metric = v[f"_{self.mode.value}"]
                # Log the confusion matrix as a figure
                self.logger.experiment.add_figure(
                    f"{self.mode.value}_confmat_{k}",
                    self.get_confusion_matrix(
                        confmat.compute().cpu().data.numpy(), confmat.num_classes
                    ),
                    self.global_step,
                )
                # Reset the confusion matrix for the next epoch
                confmat.reset()

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
        for k, v in pred.items():
            self.metrics[k][f"_{self.mode.value}"].update(v, gt[k])
            self.confmat[k][f"_{self.mode.value}"].update(v, gt[k])

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
        )

        return losses
