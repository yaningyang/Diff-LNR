import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diff_lnr.config import DataConfig, ProjectConfig, TrainingConfig
from diff_lnr.training.engine import TrainingEngine, evaluate_loader


class TinyDataset(Dataset):
    def __init__(self):
        self.features = [
            ([0, 0, 0], 0),
            ([0, 1, 0], 0),
            ([1, 0, 1], 1),
            ([1, 1, 1], 1),
        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        values, label = self.features[index]
        return {
            "single_input_ids": torch.tensor(values, dtype=torch.long),
            "single_attention_mask": torch.ones(3, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "source_index": index,
            "record_id": f"record-{index}",
        }


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, single_input_ids, single_attention_mask):
        del single_attention_mask
        return self.classifier(single_input_ids.float())


def collate(batch):
    return {
        "single_input_ids": torch.stack([item["single_input_ids"] for item in batch]),
        "single_attention_mask": torch.stack([item["single_attention_mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "source_index": torch.tensor([item["source_index"] for item in batch]),
        "record_id": [item["record_id"] for item in batch],
    }


def test_training_engine_saves_and_restores_best_checkpoint(tmp_path: Path):
    config = ProjectConfig(
        data=DataConfig(context_file=tmp_path / "unused.tsv"),
        training=TrainingConfig(
            epochs=2,
            learning_rate=0.1,
            batch_size_by_max_deletion={50: 2},
            evaluation_batch_size=2,
            num_workers=0,
            pin_memory=False,
            early_stopping_patience=1,
        ),
    )
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=collate)
    model = TinyClassifier()
    engine = TrainingEngine(
        model,
        config,
        torch.device("cpu"),
        tmp_path,
        logging.getLogger("test_engine"),
    )
    best = engine.fit(loader, loader)
    assert best["epoch"] >= 1
    assert (tmp_path / "checkpoints" / "best.pt").exists()

    metrics, predictions = evaluate_loader(model, loader, torch.device("cpu"))
    assert len(predictions) == 4
    assert "mcc" in metrics
