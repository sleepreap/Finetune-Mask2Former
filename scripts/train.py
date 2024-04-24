import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from mask2former import ( Mask2FormerFinetuner, 
                        SegmentationDataModule, 
                        DATASET_DIR, 
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        ID2LABEL, 
                        LEARNING_RATE, 
                        LOGGER, 
                        PRECISION, 
                        DEVICES, 
                        EARLY_STOPPING_CALLBACK, 
                        CHECKPOINT_CALLBACK, 
                        EPOCHS )


if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model=Mask2FormerFinetuner(ID2LABEL, LEARNING_RATE)

    trainer = pl.Trainer(
        logger=LOGGER,
        precision=PRECISION,
        accelerator='cuda',
        devices=DEVICES,
        strategy="ddp",
        callbacks=[EARLY_STOPPING_CALLBACK, CHECKPOINT_CALLBACK],
        max_epochs=EPOCHS
    )
    print("Training starts!!")
    trainer.fit(model,data_module)
    print("saving model!")
    trainer.save_checkpoint("mask2former.ckpt")
