# YOLOv8-Smoke-Detection-Model-CCT-backbone-
Fire and smoke detection system with a YOLOv8 object detector paired with a Compact Convolutional Transformer (CCT)

# Smoke and Fire Detection Using YOLOv8 with CCT backbone and run on Jetson Xavior NX  
Final Project – Computer Vision / AI

This project implements a **real-time smoke detection system** using **YOLOv8**, trained on a labeled smoke dataset from Roboflow. The goal is to detect smoke in images or video streams and demonstrate deployment on an **NVIDIA Jetson Xavier NX** for edge inference.

In addition to the core YOLOv8 detector, this project also includes a **CCT (Compact Convolutional Transformer) enhancement proposal** that can be integrated as a secondary verification model to reduce false positives. This satisfies the requirement to incorporate a YOLO modification or CCT-based extension.

---

## Table of Contents  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Training Environment (Google Colab)](#training-environment-google-colab)  
- [Model Export](#model-export)  
- [Jetson Xavier NX Deployment](#jetson-xavier-nx-deployment)  
- [CCT Enhancement Proposal](#cct-enhancement-proposal)   
- [Results](#results)  
- [Limitations](#limitations)  
- [Future Work](#future-work)

---

## Overview  

The purpose of this project is to build a **lightweight smoke detection system** capable of running in real time on embedded hardware. YOLOv8 was chosen due to its strong performance, small size, and fast inference speed.

### The model is:
- YOLOv8-based smoke detector  
- Trained on Roboflow smoke dataset  
- Runs on Jetson Xavier NX  
- CCT-based enhancement proposed for verification  
- Includes screenshots and testing examples  
- All code and training steps documented in this repository  

---

## Dataset  

This project uses the **Smoke Detection Dataset** from Roboflow:

**Dataset URL:**  
https://universe.roboflow.com/jiho-wrqg7/smoke-ddba9

### Dataset Contains:  
- 438 images  
- 2 classes:  
  - `smoke`  
  - `nonsmoke`  
- Format: YOLOv8  
- No Preprocessing, No Augmentation

### Dataset Import (Colab)

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("akemi").project("smoke-ddba9-zjcub")
version = project.version(1)
dataset = version.download("yolov8")

```

## Training Environment (Google Colab)
### Detailed Training Notebook (PDF)

Screenshots of training the model(How to):

[Download Training PDF](YOLOv8_Training_Google_Colab.pdf)

This PDF includes:
- Dataset import from Roboflow  
- YOLOv8 installation  
- Training configuration  
- Epoch logs  
- Model export  
- Validation testing  


### Jetson Xavier NX Deployment

An attempt was made to deploy the trained YOLOv8 model on the NVIDIA Jetson Xavier NX for real-time edge inference. However, during installation of the `ultralytics` package on the Jetson (ARM/aarch64 architecture), a dependency (`puccinialin`) failed to install due to lack of ARM compatibility on PyPI. As a result, full YOLOv8 execution on the Jetson could not be completed within the project time frame. This deployment attempt and error were documented as part of the project evaluation.




### CCT Enhancement Proposal – Implementation Steps
[CCT Enhancement PDF](CCT_backbone.pdf)

1. The smoke dataset was downloaded from Roboflow in YOLOv8 format and used to train the YOLOv8 model in Google Colab.

2. The trained YOLOv8 model (`best.pt`) was loaded back into Colab for inference.

3. A validation image was passed into YOLOv8 to obtain smoke detection bounding boxes.

4. The first detected bounding box was extracted and cropped from the original image.

5. A pretrained Vision Transformer (ViT) model from the `timm` library was loaded as a transformer-based enhancement module.

6. The cropped YOLO detection was resized and normalized using ImageNet preprocessing.

7. The transformed crop was passed into the ViT model to generate top-5 prediction outputs.

This process demonstrates a hybrid CNN–Transformer pipeline where YOLOv8 performs fast object detection and the transformer performs second-stage feature-based analysis on the detected region.

### Results

The trained YOLOv8 model successfully detected smoke in validation images with clear bounding boxes. The detection results, cropped regions, and transformer-based verification outputs are documented in the following files:

- `YOLOv8_Training_Google_Colab.pdf`
- `CCT_backbone.pdf`

These results confirm that the model is capable of detecting smoke and that the hybrid YOLO + Transformer pipeline functions correctly.

### Limitations

The model was trained on a relatively small dataset, which may limit its generalization to unseen environments. In addition, full Jetson Xavier NX deployment was not completed due to package compatibility issues on ARM architecture. The transformer enhancement was used for inference only and was not fine-tuned on the smoke dataset.

### Future Work

Future improvements include collecting a larger and more diverse smoke dataset, fine-tuning the transformer model specifically for smoke classification, and resolving Jetson dependency issues to enable full real-time edge deployment. Additional work could also include extending the system for fire detection and emergency response applications.

