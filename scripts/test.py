import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from mask2former.model import Mask2FormerFinetuner
from mask2former.dataset import SegmentationDataModule
import mask2former.config as config
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
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model = Mask2FormerFinetuner.load_from_checkpoint(model_path,id2label=config.ID2LABEL, lr=config.LEARNING_RATE)

    trainer = pl.Trainer(
        logger=config.LOGGER,
        precision=config.PRECISION,
        accelerator='cuda',
        devices=[0],
        num_nodes=1,
    )
    print("Test starts!!")
    trainer.test(model,data_module)
