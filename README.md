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

Each class folder contains dark scene images belonging to different object categories.
