"""
PyTorch Dataset wrapper for EC-DarkPattern splits.
Used by baseline training scripts.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_label_map(processed_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    with open(processed_dir / "label_map.json") as f:
        return json.load(f)


class DarkPatternDataset(Dataset):
    """
    Tokenises text samples from a processed CSV split.

    Args:
        split:     "train", "val", or "test"
        tokenizer: HuggingFace tokenizer
        max_length: token truncation length
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        processed_dir: Path = PROCESSED_DIR,
    ):
        # Allow callers to pass a full filename stem (e.g. "train_v2") or
        # just the split name ("train").  Always appends ".csv".
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run `python -m src.data.preprocess` first."
            )
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            str(row["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }

    @property
    def num_labels(self) -> int:
        label_map = load_label_map(PROCESSED_DIR)
        return len(label_map)

    @property
    def texts(self) -> list[str]:
        return self.df["text"].tolist()

    @property
    def categories(self) -> list[str]:
        return self.df["category"].tolist()
