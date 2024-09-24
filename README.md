
# :zap:FastestDet for Tooth:zap:
[![DOI](https://zenodo.org/badge/508635170.svg)](https://zenodo.org/badge/latestdoi/508635170)
![Static Badge](https://img.shields.io/badge/Original%20Source-Github--Dog%20Qiuqiu-orange?link=https%3A%2F%2Fgithub.com%2Fdog-qiuqiu%2FFastestDet)
![image](https://img.shields.io/github/license/dog-qiuqiu/FastestDet)

# Implementation of Object Detection on Embedded Devices for Dental Health Monitoring using FastestDet and YOLOv8n

## Overview
This repository contains the source code for my undergraduate thesis project, which focuses on implementing an object detection system on an embedded device for dental health monitoring. 

The primary goal of this project is to investigate the feasibility of deploying lightweight object detection algorithms on resource-constrained embedded systems to facilitate accessible and cost-effective dental health monitoring. 

## Key Features
* **Lightweight Algorithms:** The project evaluates two lightweight object detection algorithms: FastestDet and YOLOv8n, chosen for their speed and efficiency on resource-limited devices.
* **Embedded Deployment:** The models are deployed on an SBC HG680P, an affordable embedded system with an ARM processor, specifically chosen for its suitability for running FastestDet.
* **Real-time Detection:**  The system is designed to perform real-time object detection on images captured by an intraoral camera, enabling continuous monitoring.
* **Performance Evaluation:**  The performance of both algorithms is evaluated using relevant metrics such as mean average precision (mAP), inference time, and frames per second (FPS).
* **Open Source:** The source code for deploying both FastestDet and YOLOv8n is provided, offering a valuable resource for researchers and developers interested in similar applications.

## Algorithms Explored
* **FastestDet:**  An ultra-lightweight, anchor-free object detection algorithm specifically optimized for ARM platforms. It boasts improvements in speed and a smaller model size compared to other lightweight options, making it ideal for embedded systems.
* **YOLOv8n:** The most lightweight variant of the state-of-the-art YOLOv8 family, chosen for its accuracy and speed. YOLOv8n serves as a benchmark for comparison with FastestDet in this project.

## Dataset
The project utilizes a dataset of 9,880 images with 82,830 annotations, encompassing seven different classes related to dental health.

## Results
The findings of this research demonstrate that object detection can be effectively implemented on embedded systems, even with limited computational resources.  However, there are trade-offs to consider between inference speed and detection accuracy, particularly when choosing between different resolutions and algorithms.

## Contributions
* **Feasibility Demonstration:** This project successfully demonstrates the feasibility of deploying object detection algorithms on embedded devices for dental health monitoring, paving the way for the development of more accessible and cost-effective solutions.
* **Comparative Analysis:** Provides a comparative analysis of FastestDet and YOLOv8n, offering insights into the strengths and weaknesses of each algorithm when deployed on an embedded device.
* **Open-Source Code:**  The provided source code serves as a valuable reference for other researchers and developers interested in exploring similar applications on embedded systems.



# Reference
* https://github.com/dog-qiuqiu/FastestDet
* https://github.com/Tencent/ncnn
* https://github.com/jahongir7174/YOLOv8-onnx
