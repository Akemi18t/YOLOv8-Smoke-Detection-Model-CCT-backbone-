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
- [Screenshots](#screenshots)  
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






### Transformer (CCT / ViT) Enhancement – Implementation Steps

1. The smoke dataset was downloaded from Roboflow in YOLOv8 format and used to train the YOLOv8 model in Google Colab.

2. The trained YOLOv8 model (`best.pt`) was loaded back into Colab for inference.

3. A validation image was passed into YOLOv8 to obtain smoke detection bounding boxes.

4. The first detected bounding box was extracted and cropped from the original image.

5. A pretrained Vision Transformer (ViT) model from the `timm` library was loaded as a transformer-based enhancement module.

6. The cropped YOLO detection was resized and normalized using ImageNet preprocessing.

7. The transformed crop was passed into the ViT model to generate top-5 prediction outputs.

This process demonstrates a hybrid CNN–Transformer pipeline where YOLOv8 performs fast object detection and the transformer performs second-stage feature-based analysis on the detected region.

