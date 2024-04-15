import pytorch_lightning as pl
import torch
from transformers import SegformerForSemanticSegmentation
from torch import nn
import evaluate
import time
import json 
import numpy as np

class SegformerFinetuner(pl.LightningModule):

    def __init__(self, id2label, lr ):
        super(SegformerFinetuner, self).__init__()
        self.lr=lr
        self.id2label = id2label
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        evaluate.load
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")

    def forward(self, pixel_values, labels):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return(outputs)
   
    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        total_time = time.time() - self.start_time
        metrics = {'final_epoch': self.current_epoch, 'training_time': total_time}
        with open('segformer_hyperparameters.json', 'w') as f:
            json.dump(metrics, f)

    def training_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.train_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
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
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True, on_epoch=True, logger=True)
        return(metrics)

    def validation_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)   
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks[0].shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.val_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
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
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True)
        return(metrics)
        
    def test_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.test_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
            num_labels=self.num_classes,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        
        # Calculate FN and FP
        false_negatives = np.sum((predicted.detach().cpu().numpy() == 0) & (masks.detach().cpu().numpy() == 1))
        false_positives = np.sum((predicted.detach().cpu().numpy() == 1) & (masks.detach().cpu().numpy() == 0))
        
        # Total number of instances
        total_instances = np.prod(predicted.shape)
        
        # Calculate percentages
        percentage_fn = (false_negatives / total_instances) 
        percentage_fp = (false_positives / total_instances) 
    
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
            self.log(k,v,sync_dist=True)
        return(metrics)
        
    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], self.lr)
