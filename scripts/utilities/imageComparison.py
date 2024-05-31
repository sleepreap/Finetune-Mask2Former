import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mask2former import ( Mask2FormerFinetuner, 
                        SegmentationDataModule, 
                        DATASET_DIR, 
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        ID2LABEL, 
                        LEARNING_RATE)
from transformers import AutoImageProcessor
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from colorPalette import color_palette,  apply_palette

def dataset_predictions(dataloader):
    pred_set=[]
    label_set=[]
    image_set=[]
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
        for image in original_images:
            image_set.append(image)
    return pred_set, label_set, image_set

def savePredictions(pred_set, label_set, image_set, save_path):
    palette = color_palette()
    for i in tqdm(range(len(pred_set)), desc="Saving predictions"):
        file_name = f"result_{i}"
        pred = pred_set[i]
        label = label_set[i]
        image= image_set[i]
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))  # Convert image from (C, H, W) to (H, W, C)
        colored_prediction = apply_palette(pred, palette)
        color_seg = apply_palette(pred, palette)
        
        # Create an overlay image by blending the original image with the colored segmentation mask
        overlay_image = (image * 0.5 + color_seg * 0.5).astype(np.uint8)
        
        f, axarr = plt.subplots(1, 3, figsize=(22, 7.5))  # One row, three columns
        axarr[0].imshow(colored_prediction)
        axarr[0].set_title("Predictions", fontsize=20)
        axarr[0].axis('off')
        axarr[1].imshow(overlay_image)
        axarr[1].set_title("Overlay Image", fontsize=20)
        axarr[1].axis('off')
        axarr[2].imshow(image)
        axarr[2].set_title("Original Image", fontsize=20)
        axarr[2].axis('off')
        plt.savefig(os.path.join(save_path, f"{file_name}.png"), bbox_inches='tight')
        plt.close(f)
            
    print("Predictions saved")


if __name__=="__main__":
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
        
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = Mask2FormerFinetuner.load_from_checkpoint(model_path,id2label=ID2LABEL, lr=LEARNING_RATE)
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    pred_set, label_set, image_set= dataset_predictions(test_dataloader)
    savePredictions(pred_set, label_set,image_set, save_path)
    
