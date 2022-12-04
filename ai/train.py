from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from model import PatentGenerator
from dataset import PatentDataModule
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, required=True)
args = parser.parse_args()

logger = TensorBoardLogger(
    save_dir="logs",
    name="gpt2",
    default_hp_metric=False
)

ckpt_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="ckpt",
    filename="patent_gpt",
    mode="min"
)

earlystop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=3,
    verbose=True,
    mode='min'
)

trainer = Trainer(
    logger=logger, callbacks=[ckpt_callback, earlystop_callback],
    accelerator="gpu", log_every_n_steps=1000, strategy="ddp", devices=2
)

if __name__ == "__main__":
    model = PatentGenerator()
    data_module = PatentDataModule(data_dir=args.data_path)
    trainer.fit(model=model, datamodule=data_module)
