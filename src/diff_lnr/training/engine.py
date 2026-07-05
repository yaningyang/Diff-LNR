"""One implementation of training, validation, checkpointing, and evaluation."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ProjectConfig
from ..evaluation import compute_binary_metrics
from ..losses import build_classification_loss
from ..utils.io import write_csv, write_json
from ..utils.reproducibility import move_model_inputs


class TrainingEngine:
    """Train the configured model and restore the best validation-F1 state."""

    def __init__(
        self,
        model: nn.Module,
        config: ProjectConfig,
        device: torch.device,
        output_dir: str | Path,
        logger: logging.Logger,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.loss_fn = build_classification_loss()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.checkpoint_path = self.output_dir / "checkpoints" / "best.pt"

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> dict[str, Any]:
        best_f1 = float("-inf")
        best_metrics: dict[str, Any] = {}
        best_state_in_memory: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.training.epochs + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch - 1)
            train_metrics, _ = self._run_epoch(
                train_loader,
                training=True,
                description=f"Epoch {epoch} train",
            )
            validation_metrics, _ = self._run_epoch(
                validation_loader,
                training=False,
                description=f"Epoch {epoch} validation",
            )
            epoch_result = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(epoch_result)
            self._save_history(history)
            self.logger.info(
                "epoch=%d train_loss=%.4f train_f1=%.4f "
                "val_loss=%.4f val_f1=%.4f val_auroc=%.4f val_mcc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["f1"],
                validation_metrics["loss"],
                validation_metrics["f1"],
                validation_metrics["auroc"],
                validation_metrics["mcc"],
            )

            improved = (
                float(validation_metrics["f1"])
                > best_f1 + self.config.training.early_stopping_min_delta
            )
            if improved:
                best_f1 = float(validation_metrics["f1"])
                best_metrics = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
                epochs_without_improvement = 0
                if self.config.training.save_checkpoint:
                    self._save_checkpoint(epoch, best_metrics)
                else:
                    best_state_in_memory = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.training.early_stopping_patience:
                self.logger.info(
                    "early stopping after %d epochs without sufficient improvement",
                    epochs_without_improvement,
                )
                break

        if not best_metrics:
            raise RuntimeError("training finished without a valid best epoch")
        if self.config.training.save_checkpoint:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            if best_state_in_memory is None:
                raise RuntimeError("best model state was not retained")
            self.model.load_state_dict(best_state_in_memory)
        write_json(self.output_dir / "best_metrics.json", best_metrics)
        return best_metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        training: bool,
        description: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.model.train(training)
        labels: list[int] = []
        predictions: list[int] = []
        probabilities: list[float] = []
        prediction_rows: list[dict[str, Any]] = []
        loss_sum = 0.0
        example_count = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in tqdm(loader, desc=description, leave=False):
                model_inputs = move_model_inputs(batch, self.device)
                targets = batch["label"].to(self.device)
                logits = self.model(**model_inputs)
                loss = self.loss_fn(logits, targets)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                batch_size = targets.shape[0]
                loss_sum += float(loss.detach().item()) * batch_size
                example_count += batch_size
                batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
                batch_predictions = torch.argmax(logits, dim=1)
                target_values = targets.detach().cpu().tolist()
                prediction_values = batch_predictions.detach().cpu().tolist()
                probability_values = batch_probabilities.detach().cpu().tolist()
                labels.extend(target_values)
                predictions.extend(prediction_values)
                probabilities.extend(probability_values)
                for record_id, source_index, target, prediction, probability in zip(
                    batch["record_id"],
                    batch["source_index"].tolist(),
                    target_values,
                    prediction_values,
                    probability_values,
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "record_id": record_id,
                            "source_index": source_index,
                            "label": target,
                            "prediction": prediction,
                            "pathogenic_probability": probability,
                        }
                    )

        metrics = compute_binary_metrics(labels, predictions, probabilities)
        metrics["loss"] = loss_sum / max(example_count, 1)
        return metrics, prediction_rows

    def _save_checkpoint(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "config": self.config.to_dict(),
            },
            self.checkpoint_path,
        )

    def _save_history(self, history: list[dict[str, Any]]) -> None:
        write_json(self.output_dir / "history.json", history)
        rows = []
        for item in history:
            row: dict[str, Any] = {"epoch": item["epoch"]}
            for split in ("train", "validation"):
                for metric, value in item[split].items():
                    if metric != "confusion_matrix":
                        row[f"{split}_{metric}"] = value
            rows.append(row)
        if rows:
            write_csv(
                self.output_dir / "history.csv",
                rows,
                fieldnames=list(rows[0]),
            )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    output_dir: str | Path | None = None,
    save_predictions: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a model without changing its parameters."""

    model = model.to(device)
    model.eval()
    loss_fn = build_classification_loss()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    example_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation", leave=False):
            inputs = move_model_inputs(batch, device)
            targets = batch["label"].to(device)
            logits = model(**inputs)
            loss = loss_fn(logits, targets)
            batch_size = targets.shape[0]
            loss_sum += float(loss.item()) * batch_size
            example_count += batch_size
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            batch_predictions = torch.argmax(logits, dim=1)
            target_values = targets.cpu().tolist()
            prediction_values = batch_predictions.cpu().tolist()
            probability_values = batch_probabilities.cpu().tolist()
            labels.extend(target_values)
            predictions.extend(prediction_values)
            probabilities.extend(probability_values)
            for record_id, source_index, target, prediction, probability in zip(
                batch["record_id"],
                batch["source_index"].tolist(),
                target_values,
                prediction_values,
                probability_values,
                strict=True,
            ):
                rows.append(
                    {
                        "record_id": record_id,
                        "source_index": source_index,
                        "label": target,
                        "prediction": prediction,
                        "pathogenic_probability": probability,
                    }
                )
    metrics = compute_binary_metrics(labels, predictions, probabilities)
    metrics["loss"] = loss_sum / max(example_count, 1)
    if output_dir is not None:
        output_path = Path(output_dir)
        write_json(output_path / "metrics.json", metrics)
        if save_predictions and rows:
            write_csv(output_path / "predictions.csv", rows, list(rows[0]))
    return metrics, rows
