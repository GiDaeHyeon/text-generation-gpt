"""
DataModules
"""
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class PatentDataset(Dataset):
    def __init__(self, data_dir: str, is_train: bool) -> None:
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        with Path(data_dir).open("r") as fp:
            all_data = fp.readlines()
        self.data = all_data[:250000] if is_train else all_data[250000:]

    def __getitem__(self, idx) -> tuple:
        txt = self.tokenizer.eos_token.join(self.data[idx].split("\t"))
        tokens = self.tokenizer(txt, return_tensors="pt", truncation=True,
                                max_length=1024, padding="max_length")
        return tokens["input_ids"], tokens["attention_mask"]

    def __len__(self) -> int:
        return len(self.data)


class PatentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.train_dataset = PatentDataset(data_dir=data_dir, is_train=True)
        self.val_dataset = PatentDataset(data_dir=data_dir, is_train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4
        )
