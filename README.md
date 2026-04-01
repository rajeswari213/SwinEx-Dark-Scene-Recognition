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
