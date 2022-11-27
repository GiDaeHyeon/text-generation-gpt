"""

"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class PatentGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)

    def forward(self, texts: list[str]) -> torch.Tensor:
        tokens = [self.tokenizer(t, return_tensors="pt") for t in texts]
        return self.model(**tokens)  # type: ignore

    def generate(self, text: str, max_length=1024) -> list:
        token = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(  #type: ignore
            inputs=token, max_length=max_length, do_sample=True,
            top_k=50, top_p=.95, num_return_sequences=5
            )
        return [self.tokenizer(output[0], skip_special_tokens=True) for output in outputs]
