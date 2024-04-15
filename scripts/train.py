import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from mask2former.model import Mask2FormerFinetuner
from mask2former.dataset import SegmentationDataModule
import mask2former.config as config


if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model=Mask2FormerFinetuner(config.ID2LABEL, config.LEARNING_RATE)

    trainer = pl.Trainer(
        logger=config.LOGGER,
        precision=config.PRECISION,
        accelerator='cuda',
        devices=config.DEVICES,
        callbacks=[config.EARLY_STOPPING_CALLBACK, config.CHECKPOINT_CALLBACK],
        max_epochs=config.EPOCHS
    )
    print("Training starts!!")
    trainer.fit(model,data_module)
    print("saving model!")
    trainer.save_checkpoint("mask2former.ckpt")
