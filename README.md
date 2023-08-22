# SleepTransformer and OOD Detection (ViM)

This repository contains the implementation of data processing and OOD detection for sleep stage classification using the SleepTransformer model.

The research poster for the entire project is shown below:
![SleepTransformer](figure/poster.png)

## Steps to Reproduce

### 1. Dataset Preparation

1. Download part of the SHHS dataset, not for training, just to generate features for OOD detection.

<br/>These are the source code and experimental setup for [SHHS](https://sleepdata.org/datasets/shhs).

2. Download the sc and st data from the SleepEDF dataset and organize them into a 'meta' folder.

<br/>These are the source code and experimental setup for [SleepEDF](https://www.physionet.org/content/sleep-edfx/1.0.0/).

### 2. Running SleepTransformer

1. Download the SleepTransformer model code and pretrained weights and biases obtained from the SHHS dataset.

**The model weights trained on SHHS** are available at https://zenodo.org/record/7927282 for reproducing the results in the paper.

2. Preprocess the original data (.edf) into files with 21 time steps per sample (.mat) using MATLAB. Generate 'test' and 'train' lists. Note that the 'train' list is used only for model benchmarking and not actual training.
3. Modify the dataset address in the SleepTransformer code, run it, and generate prediction results in .mat format. Additionally, include code to output features extracted from fc2 and save them in .pkl format.
4. Process the output predictions using the tools in the 'evaluation' folder to consolidate the 21 time step predictions into a single aggregated result.

### 3. Running OOD Detection (ViM)

1. Read the 'benchmark' file to understand the idea behind ViM. Design code for OOD detection based on this idea. During this process, preprocess the data by taking logarithms and apply different thresholds for varying levels of OOD detection.
2. Output a list of detected OOD data, and use this list to remove OOD samples. Pay attention to time step correspondences; each data group has a last sample of 21 time steps that hasn't been through detection. Therefore, be cautious while matching indices for removal.
3. Compare the performance using different lists obtained from different thresholds.

## Disclaimer

This repository serves as a guide for the described steps and their implementation. Please refer to the respective codes and tools for detailed usage instructions.

## Tips
This README provides an overview and may omit some details.
