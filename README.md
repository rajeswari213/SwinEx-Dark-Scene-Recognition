# SwinEx Dark Scene Recognition
## Overview

This repository contains the implementation for dark scene image preprocessing and evaluation using SwinEx-based enhancement techniques. The framework focuses on improving visibility in low-light environments using Retinex-based preprocessing and evaluates performance on the EX-DARK dataset using several image quality metrics.

Dark scene recognition is challenging due to low illumination, noise, and reduced contrast. This project applies enhancement techniques to improve visual quality and evaluates the results using standard metrics such as PSNR, SSIM, FSIM, GMSD, SAM, UQI, RMSE, and MAE.

## Features

Dark image preprocessing using Retinex enhancement

Dataset analysis (brightness, noise estimation, image resolution)

Visualization of before vs after enhancement

Performance evaluation across multiple image quality metrics

Epoch-wise performance plotting

## Dataset

The implementation is designed to work with the EX-DARK dataset.

Dataset link:

https://www.kaggle.com/datasets/washingtongold/exdark-dataset

Datase Split

Split	#       Images	% of Dataset

Training	       5155	     70%

Validation	     1104	     15%

Test	           1104	     15%


Each class folder contains dark scene images belonging to different object categories.

## How to Run

Clone the repository:

git clone https://github.com/rajeswari213/SwinEx-Dark-Scene-Recognition.git

Install dependencies:

pip install -r requirements.txt

Run the code:

python main.py

## Hardware and Software Environment

Processor: Intel Core i7 
GPU: NVIDIA RTX 3060 
RAM: 16 GB
OS: Windows 11 
Python: 3.10
Framework: PyTorch 2.2


# Evaluation Protocol for SwinEx Dark Scene Recognition

## 1. Dataset Preparation
- Download EX-DARK dataset from Kaggle
- Organize as: dataset/
  ├── train/ (class folders)
  ├── val/
  └── test/
- Verify class distribution (12 classes minimum)

## 2. Preprocessing Pipeline
- Apply Retinex enhancement to all test images
- Normalize images to [0,1] range
- Resize to consistent dimensions (e.g., 384x384)

## 3. Metrics Calculation
Compute the following metrics per image, then average:

| Metric | Description | Range | Higher is Better |
|--------|-------------|-------|------------------|
| PSNR   | Peak Signal-to-Noise Ratio | 0-∞ dB | ✓ |
| SSIM   | Structural Similarity | 0-1 | ✓ |
| FSIM   | Feature Similarity | 0-1 | ✓ |
| GMSD   | Gradient Magnitude Similarity Deviation | 0-1 | ✗ (lower better) |
| SAM    | Spectral Angle Mapper | 0°-90° | ✗ |
| UQI    | Universal Quality Index | -1 to 1 | ✓ |
| RMSE   | Root Mean Square Error | 0-∞ | ✗ |
| MAE    | Mean Absolute Error | 0-∞ | ✗ |

## 4. Evaluation Steps

```python
# evaluation.py
import torch
from metrics import calculate_all_metrics

def evaluate_model(model, test_loader, config):
    results = {metric: [] for metric in config['metrics']}
    
    for images, targets in test_loader:
        enhanced = model(images)
        metrics = calculate_all_metrics(enhanced, targets)
        
        for metric_name, value in metrics.items():
            results[metric_name].append(value)
    
    # Average results
    final_results = {k: np.mean(v) for k, v in results.items()}
    return final_results
