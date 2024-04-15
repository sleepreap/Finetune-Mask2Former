import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segformer.model import SegformerFinetuner
from segformer.dataset import SegmentationDataModule
import segformer.config as config
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from colorPalette import color_palette
from colorPalette import apply_palette

def dataset_predictions(dataloader):
    pred_set=[]
    label_set=[]
    for batch in tqdm((dataloader), desc="Doing predictions"):
        images, labels = batch['pixel_values'], batch['labels']
        outputs = model(images, labels)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            #size of original image is 640x640
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted_mask = upsampled_logits.argmax(dim=1).numpy()
        labels = labels.numpy()
        pred_set.append(predicted_mask)
        label_set.append(labels)        
    return pred_set, label_set

def savePredictions(pred_set, label_set, save_path):
    palette = color_palette()
    for i in tqdm(range(len(pred_set)), desc="Saving predictions"):
        file_name = f"result_{i}"
        n_plots = len(pred_set[i])  # Number of items per batch
        # Dynamically adjust subplot layout based on batch size
        f, axarr = plt.subplots(n_plots, 2, figsize=(15, 15 * n_plots))  # Ensure each subplot has enough space
        f.subplots_adjust(hspace=0.5)  # Adjust horizontal space if needed

        if n_plots > 1:
            for j in range(n_plots):
                image = pred_set[i][j, :]
                label = label_set[i][j, :]
                colored_image = apply_palette(image, palette)
                colored_label = apply_palette(label, palette)
                axarr[j, 0].imshow(colored_image)
                axarr[j, 1].imshow(colored_label)
                if j == 0:
                    axarr[j, 0].set_title("Predictions", {'fontsize': 30})
                    axarr[j, 1].set_title("Ground Truth", {'fontsize': 30})
        else:
            # For single plot adjust for non-indexed axarr
            image = pred_set[i][0, :]
            label = label_set[i][0, :]
            colored_image = apply_palette(image, palette)
            colored_label = apply_palette(label, palette)
            axarr[0].imshow(colored_image)
            axarr[1].imshow(colored_label)
            axarr[0].set_title("Predictions", {'fontsize': 30})
            axarr[1].set_title("Ground Truth", {'fontsize': 30})

        # Save the figure
        file_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(f)

    print("Predictions saved")

if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--model_path',
    type=str,
    default='',
    help="Enter the path of your model.ckpt file"
    )
    parser.add_argument(
    '--save_path',
    type=str,
    default='',
    help="enter the path to save your images"
    )

    args = parser.parse_args()
    model_path = os.path.join( '..', args.model_path)
    save_path = os.path.join( '..', args.save_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model = SegformerFinetuner.load_from_checkpoint(model_path,id2label=config.ID2LABEL, lr=config.LEARNING_RATE)
    
    model.eval()
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    pred_set, label_set= dataset_predictions(test_dataloader)
    savePredictions(pred_set, label_set, save_path)
        
    
