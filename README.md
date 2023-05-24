# SimVP with optical flow




This repository contains the implementation code my master's thesis:

**Improvement in Frame Prediction using Optical Flow**  

## Introduction

This implementation is based on the repository https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction/tree/master. Any code files in this repository are ment to be added to or replace the present files in their implementation to produce a functioning model. Some of the model modules are also missing in this repository since no edits were made to them and they can be acessed by the original work.

Future frame prediction is a technology that allows computers to predict what future video frames will look like. This can be used to predict future occurrences in a video, anticipate anomalies, and aid autonomous devices in smart decision making. Although there is potential with frame prediction technology, there is still progress that needs to be made with it. As the predicted frame becomes farther away from the last input frame, the image becomes blurry and distorted. This indicates that the model is more uncertain about the motion occurring in the image frame. To reduce model uncertainty shown in predictions, optical flow information from each video was extracted and combined with the video frames. Optical flow is the change in direction and magnitude of a moving object in a video. This type of information is helpful for making frame predictions because it gives the model additional information on how objects are moving to base its predictions on. In this work, the change in image quality evaluation metrics and overall image quality is analyzed across 4 different datasets between a state-of-the-art frame prediction model and a modified model that combines optical flow information. The results demonstrate that adding optical flow information improves the model Mean Squared Error (MSE) by 4.11% and its Structural Similarity Index Metric (SSIM) by 0.41% for the Moving MNIST dataset. Optical flow improved the SSIM value of Taxi BJ, KTH, and KITTI by 0.02%, 0.0108342%, and 1.29663% respectively. While there was a consistent improvement in performance, the models still need more improvement in terms of the quality of images predicted in the distant future.

## Dependencies
* torch
* scikit-image=0.16.2
* numpy
* argparse
* tqdm
* open-cv

## Overview

* `API/` contains dataloaders and metrics.
* `main.py` is the executable python file with possible arguments.
* `model.py` contains the SimVP model.
* `exp.py` is the core file for training, validating, and testing pipelines.

## SimVP Citation
@InProceedings{Gao_2022_CVPR,
    author    = {Gao, Zhangyang and Tan, Cheng and Wu, Lirong and Li, Stan Z.},
    title     = {SimVP: Simpler Yet Better Video Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3170-3180}
}
