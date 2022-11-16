"""

"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PatentGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("")

    def forward(self, text: str) -> torch.Tensor:
        return torch.Tensor()
