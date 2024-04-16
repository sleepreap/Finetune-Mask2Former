import pytorch_lightning as pl
import torch
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor
from torch import nn
import mask2former.config as config
import evaluate
import time
import json 
import numpy as np

class Mask2FormerFinetuner(pl.LightningModule):

    def __init__(self, id2label, lr ):
        super(Mask2FormerFinetuner, self).__init__()
        self.id2label = id2label
        self.lr = lr
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-ade-semantic",
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        evaluate.load
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")
        
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        # Your model's forward method
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        batch['pixel_values'] = batch['pixel_values'].to(device)
        batch['mask_labels'] = [label.to(device) for label in batch['mask_labels']]
        batch['class_labels'] = [label.to(device) for label in batch['class_labels']]
        return batch

    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        loss = outputs.loss
        self.log("loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        self.log("loss", loss, sync_dist=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        original_images = batch["original_images"]
        ground_truth = batch["original_segmentation_maps"]
        target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
        # predict segmentation maps
        predicted_segmentation_maps =self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_sizes)
        
        predictions=predicted_segmentation_maps[0].cpu().numpy()
        
        # Calculate FN and FP
        false_negatives = np.sum((predictions == 0) & (ground_truth[0] == 1))
        false_positives = np.sum((predictions == 1) & (ground_truth[0] == 0))
        
        # Total number of instances
        total_instances = np.prod(predictions.shape)
        
        # Calculate percentages
        percentage_fn = (false_negatives / total_instances) 
        percentage_fp = (false_positives / total_instances) 
        
        # Optionally log loss here
        metrics = self.train_mean_iou._compute(
            predictions=predictions,
            references=ground_truth[0],
            num_labels=self.num_classes,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
    
        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss, 
            "mean_iou": metrics["mean_iou"], 
            "mean_accuracy": metrics["mean_accuracy"],
            "False Negative": percentage_fn,
            "False Positive": percentage_fp,
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True, batch_size=config.BATCH_SIZE)
        return(metrics)
        
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], self.lr)
