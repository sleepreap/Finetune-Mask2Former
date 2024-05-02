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
                        PRECISION)
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--model_path',
    type=str,
    default='',
    help="Enter the path of your model.ckpt file"
    )

    args = parser.parse_args()
    model_path = args.model_path
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model=Mask2FormerFinetuner.load_from_checkpoint(model_path, id2label= ID2LABEL, lr=LEARNING_RATE)

    trainer = pl.Trainer(
        logger=LOGGER,
        precision=PRECISION,
        accelerator='cuda',
        devices=[0],
        num_nodes=1,
    )
    print("Test starts!!")
    trainer.test(model,data_module)
