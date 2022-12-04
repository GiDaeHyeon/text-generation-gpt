from typing import Optional

import torch
import torch.optim as optim
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class PatentGenerator(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids if labels is None else labels
            )  # type: ignore
        return model_output.loss

    def generate(self, text: str, max_length: int = 1024) -> list:
        token = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(  #type: ignore
            inputs=token, max_length=max_length, do_sample=True,
            top_k=50, top_p=.95, num_return_sequences=5
            )
        return [self.tokenizer(output[0], skip_special_tokens=True) for output in outputs]

    def configure_callbacks(self):
        return optim.AdamW(
            params=self.model.parameters(), lr=1e-4
        )

    def training_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        input_ids, attention_masks = batch
        loss = self(input_ids=input_ids, attention_masks=attention_masks, labels=input_ids)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        input_ids, attention_masks = batch
        loss = self(input_ids=input_ids, attention_masks=attention_masks, labels=input_ids)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"val_loss": loss}
