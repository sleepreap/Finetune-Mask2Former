import pytorch_lightning as pl
import torch
from pathlib import Path
torch.set_float32_matmul_precision("medium")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mask2former.model import Mask2FormerFinetuner
from mask2former.dataset import SegmentationDataModule
import mask2former.config as config
from transformers import AutoImageProcessor
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm 
from colorPalette import color_palette
from colorPalette import apply_palette

def dataset_predictions(dataloader):
    pred_set = []
    label_set=[]
    prog_bar = tqdm(dataloader, desc="Doing predictions", total=len(dataloader))
    for data in prog_bar:
        original_images = data["original_images"]
        original_lables=data['original_segmentation_maps']
        target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
        pixel_values = data['pixel_values'].to(device)
        mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
        class_labels = [class_label.to(device) for class_label in data['class_labels']]
        outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        
        for pred in pred_maps:
            pred_set.append(pred.cpu().numpy())
        for label in original_lables:
            label_set.append(label)
    return pred_set, label_set
        

def savePredictions(pred_set, label_set, save_path):
    palette = color_palette()  # Ensure this returns a valid color mapping
    for i, (image, label) in enumerate(tqdm(zip(pred_set, label_set), desc="Saving predictions")):
        file_name = f"result_{i}"
        new_array = np.zeros_like(image)
        new_array[(image == 0) & (label == 0)] = 0
        new_array[(image == 1) & (label == 1)] = 1
        new_array[(image == 0) & (label == 1)] = 2
        new_array[(image == 1) & (label == 0)] = 3
        colored_array = apply_palette(new_array, palette)
        plt.imshow(colored_array)
        plt.axis('off')
        file_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

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
    model = Mask2FormerFinetuner.load_from_checkpoint(model_path,id2label=config.ID2LABEL, lr=config.LEARNING_RATE)
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    pred_set, label_set= dataset_predictions(test_dataloader)
    savePredictions(pred_set,label_set, save_path)
    
