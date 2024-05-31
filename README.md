# Finetuning Mask2Former on custom Dataset

## Introduction
Mask2Former is a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K).

### [Mask2Former Project page](https://github.com/facebookresearch/Mask2Former) | [Mask2Former Paper](https://arxiv.org/abs/2112.10764) | 
Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)
### [Mask2Former Hugging Face](https://huggingface.co/docs/transformers/model_doc/mask2former)

## Purpose
The purpose of this document is to build a process of finetuning Mask2Former for custom dataset on semantic segmentation. The code is done using Pytorch Lightning and the model can be imported from hugging face.

1. Create a virtual environment: `conda create -n Mask2Former python=3.10 -y` and `conda activate Mask2Former `
2. Install [Pytorch CUDA 12.1](https://pytorch.org/): ` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 `
3. Download code: `git clone https://github.com/sleepreap/Finetune-Mask2Former.git` 
4. `cd Finetune-Mask2Former` and run `pip install -e .`

## Dataset
Use createDataset.py to create the folders.
Refer to the README file in the folder "Data" on where to upload the images and labels.

## Training
1. 'cd scripts' 
2. set up the configs required in config.py
3. run the train.py file

A CSVlogger and the trained model file will be saved after the training has been completed. The model file would be saved as "Mask2Former.ckpt" in the same directory. An "output" folder will be created to store the contents of the CSVlogger.

## Testing
The testing is done using Mean-IOU, as well as pixel accuracy from the evaluate package. It will provide individual accuracy and IOU scores for each class label specified, as well as the mean accuracy and IOU scores of all the class labels. To run the test file, the model path of the trained model must be provided as an argument.

1. 'cd scripts' 
2. run the test.py file using this command: python test.py --model_path MODEL_PATH
   
```bash
e.g python test.py --model_path Mask2Former.ckpt
```

## Utilities
This folder contains the following scripts:
1. inference.py
2. labelComparison.py
3. imageComparison.py
4. predictionOverlay.py
5. saveComparisonWithOverlay.py
   
### All the scripts already reference the parent folder for the command line arguments. The images and labels used are from the test dataset respectively.

Inference.py would save all the predictions by the model on the test dataset in the specified save path folder.

```bash
1. 'cd scripts/utilities'
2. run the inference.py file using this command: python inference.py --model_path MODEL_PATH --save_path SAVE_PATH
```

labelComparison.py would save a plot of the prediction and the ground truth side by side in the specified save path folder. 

```bash
1. 'cd scripts/utilities'
2. run the labelComparison.py file using this command: python labelComparison.py --model_path MODEL_PATH --save_path SAVE_PATH
```
imageComparison.py would save a plot of the prediction, an overlay of the prediction on the image, as well as the original image side by side in the specified save path folder. 

```bash
1. 'cd scripts/utilities'
2. run the imageComparison.py file using this command: python imageComparison.py --model_path MODEL_PATH --save_path SAVE_PATH
```

predictionOverlay.py would save the overlay that shows the TP+TN+FP+FN of the predictions done by the model for all the images in the specified save path folder. Black means TN (background), Green means TP (metal-line), Red means FN (metal-line as background), Blue means FP (background as metal-line).

```bash
1. 'cd scripts/utilities'
2. run the predictionOverlay.py file using this command: python predictionOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```
saveComparisonWithOverlay.py would save a plot of the overlay and the ground truth side by side in the specified save path folder. There is only 1 comparison per image due to memory constraint.

```bash
1. 'cd scripts/utilities'
2. run the saveComparisonWithOverlay.py file using this command: python saveComparisonWithOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```

## Citation
```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```
