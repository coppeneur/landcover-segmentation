# Land Cover Segmentation with Deep Learning

## Overview

This repository contains the code and resources used for a project on land cover segmentation using deep learning techniques, particularly focusing on satellite imagery./

## Project Goals

- Utilize state-of-the-art deep learning architectures for land cover segmentation.
- Merge and adjust datasets from LandCover.ai and DeepGlobe to create a comprehensive training dataset.
- Address class imbalance and optimize training procedures for accurate segmentation.
- Compare the performance of different architectures, including U-Net with ResNet and DeepLabV3+.
- Conduct an ablation study to identify computationally efficient yet effective architectures.

## Datasets/Resources

### LandCover.ai

The LandCover.ai dataset provides manually annotated aerial imagery for land cover classification. It consists of two versions (v0 and v1), with version v1 introducing additional labels such as "road" for enhanced segmentation. [Dataset Link](https://landcover.ai.linuxpolska.com/)

### DeepGlobe Land Cover Classification Dataset

The DeepGlobe dataset offers diverse high-resolution satellite images for land cover classification. It complements the LandCover.ai dataset, enriching the training data with varied and extensive annotations. [Dataset Link](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

## Code Inspiration

The code for this project draws inspiration from the following sources:
- [Chrisnick92 - Deep Learning on LandCover.ai](https://www.kaggle.com/code/chrisnick92/deeplearning-on-landcoverai)
- [Balraj98 - DeepGlobe Land Cover Classification DeepLabV3+](https://www.kaggle.com/code/balraj98/deepglobe-land-cover-classification-deeplabv3/notebook)
