
# :zap:FastestDet for Tooth:zap:
[![DOI](https://zenodo.org/badge/508635170.svg)](https://zenodo.org/badge/latestdoi/508635170)
![Static Badge](https://img.shields.io/badge/Original%20Source-Github--Dog%20Qiuqiu-orange?link=https%3A%2F%2Fgithub.com%2Fdog-qiuqiu%2FFastestDet)
![image](https://img.shields.io/github/license/dog-qiuqiu/FastestDet)
![image](https://github.com/dog-qiuqiu/FastestDet/blob/main/data/data.png)

* FastestDet is a new lightweight object detection algorithm designed to replace the yolo-fastest series of algorithms by Dog-Qiuqiu. It is suitable for ARM platforms with limited computing resources and emphasizes single-core performance.
* Single lightweight detection head: FastestDet uses a single detection head with a parallel convolutional network structure similar to inception, which can fuse features with different receptive fields and adapt to detect objects of different scales.
* Anchor-free: FastestDet directly regresses the scale values of the ground truth boxes on the feature map, without using any prior width and height. This simplifies the model post-processing and improves the inference speed.
* Cross-grid multiple candidate objects: FastestDet not only considers the grid cell where the ground truth center point is located as a candidate object, but also includes the nearby three cells, increasing the number of positive candidate boxes.
* Dynamic positive and negative sample allocation: FastestDet dynamically allocates positive and negative samples during model training, based on the mean of the SIOU between the predicted boxes and the ground truth boxes. If the current SIOU is greater than the mean, it is a positive sample, otherwise it is a negative sample.
* Simple data augmentation: FastestDet uses simple data augmentation methods such as random translation and scaling, without using moscia and Mixup. This is because lightweight models have low learning ability and cannot handle complex data augmentation.

# Evaluating indicator/Benchmark
Network|mAPval 0.5|mAPval 0.5:0.95|Resolution|Run Time(4xCore)|Run Time(1xCore)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[yolov5s](https://github.com/ultralytics/yolov5)|56.8%|37.4%|640X640|395.31ms|1139.16ms|7.2M
[yolov6n](https://github.com/meituan/YOLOv6)|-|30.8%|416X416|109.24ms|445.44ms|4.3M
[yolox-nano](https://github.com/Megvii-BaseDetection/YOLOX)|-|25.8%|416X416|76.31ms|191.16ms|0.91M
[nanodet_m](https://github.com/RangiLyu/nanodet)|-|20.6%|320X320|49.24ms|160.35ms|0.95M
[yolo-fastestv1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40%|-|320X320|26.60ms|75.74ms|0.35M
[yolo-fastestv2](https://github.com/dog-qiuqiu/Yolo-FastestV2/tree/main/modelzoo)|24.10%|-|352X352|23.8ms|68.9ms|0.25M
FastestDet|25.3%|13.0%|352X352|23.51ms|70.62ms|0.24M
* ***Test platform Radxa Rock3A RK3568 ARM Cortex-A55 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
* ***CPU lock frequency 2.0GHz***
# Improvement
* Anchor-Free
* Single scale detector head
* Cross grid multiple candidate targets
* Dynamic positive and negative sample allocation
# Multi-platform benchmark
Equipment|Computing backend|System|Framework|Run time(Single core)|Run time(Multi core)
:---:|:---:|:---:|:---:|:---:|:---:
Radxa rock3a|RK3568(arm-cpu)|Linux(aarch64)|ncnn|70.62ms|23.51ms
Radxa rock3a|RK3568(NPU)|Linux(aarch64)|rknn|28ms|-
Qualcomm|Snapdragon 835(arm-cpu)|Android(aarch64)|ncnn|32.34ms|16.24ms
Intel|i7-8700(X86-cpu)|Linux(amd64)|ncnn|4.51ms|4.33ms
# How to use
## Dependent installation
* PiP(Note pytorch CUDA version selection)
  ```
  pip install -r requirements.txt
  ```
## Test
* Picture test
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg
  ```
<div align=center>
<img src="https://github.com/dog-qiuqiu/FastestDet/blob/main/result.png"> />
</div>

## How to train
### Building data sets(The dataset is constructed in the same way as darknet yolo)
* The format of the data set is the same as that of Darknet Yolo, Each image corresponds to a .txt label file. The label format is also based on Darknet Yolo's data set label format: "category cx cy wh", where category is the category subscript, cx, cy are the coordinates of the center point of the normalized label box, and w, h are the normalized label box The width and height, .txt label file content example as follows:
  ```
  11 0.344192634561 0.611 0.416430594901 0.262
  14 0.509915014164 0.51 0.974504249292 0.972
  ```
* The image and its corresponding label file have the same name and are stored in the same directory. The data file structure is as follows:
  ```
  .
  ├── train
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  └── val
      ├── 000043.jpg
      ├── 000043.txt
      ├── 000057.jpg
      ├── 000057.txt
      ├── 000070.jpg
      └── 000070.txt
  ```
* Generate a dataset path .txt file, the example content is as follows：
  
  train.txt
  ```
  /home/qiuqiu/Desktop/dataset/train/000001.jpg
  /home/qiuqiu/Desktop/dataset/train/000002.jpg
  /home/qiuqiu/Desktop/dataset/train/000003.jpg
  ```
  val.txt
  ```
  /home/qiuqiu/Desktop/dataset/val/000070.jpg
  /home/qiuqiu/Desktop/dataset/val/000043.jpg
  /home/qiuqiu/Desktop/dataset/val/000057.jpg
  ```
* Generate the .names category label file, the sample content is as follows:
 
  category.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  
  ```
* The directory structure of the finally constructed training data set is as follows:
  ```
  .
  ├── category.names        # .names category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # train dataset path .txt file
  ├── val                    # val dataset
  │   ├── 000043.jpg
  │   ├── 000043.txt
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000070.jpg
  │   └── 000070.txt
  └── val.txt                # val dataset path .txt file

  ```
### Build the training .yaml configuration file
* Reference./configs/coco.yaml
  ```
  DATASET:
    TRAIN: "/home/qiuqiu/Desktop/coco2017/train2017.txt"  # Train dataset path .txt file
    VAL: "/home/qiuqiu/Desktop/coco2017/val2017.txt"      # Val dataset path .txt file 
    NAMES: "dataset/coco128/coco.names"                   # .names category label file
  MODEL:
    NC: 80                                                # Number of detection categories
    INPUT_WIDTH: 352                                      # The width of the model input image
    INPUT_HEIGHT: 352                                     # The height of the model input image
  TRAIN:
    LR: 0.001                                             # Train learn rate
    THRESH: 0.25                                          # ？？？？
    WARMUP: true                                          # Trun on warm up
    BATCH_SIZE: 64                                        # Batch size
    END_EPOCH: 350                                        # Train epichs
    MILESTIONES:                                          # Declining learning rate steps
      - 150
      - 250
      - 300
  ```
### Train
* Perform training tasks
  ```
  python3 train.py --yaml configs/coco.yaml
  ```
### Evaluation
* Calculate map evaluation
  ```
  python3 eval.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth
  ```
* COCO2017 evaluation
  ```
  creating index...
  index created!
  creating index...
  index created!
  Running per image evaluation...
  Evaluate annotation type *bbox*
  DONE (t=30.85s).
  Accumulating evaluation results...
  DONE (t=4.97s).
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.130
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.253
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.119
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.237
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.142
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.208
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.043
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.236
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.372

  ```
# Deploy
## Export onnx
* You can export .onnx by adding the --onnx option when executing test.py
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg --onnx
  ```
## Export torchscript
* You can export .pt by adding the --torchscript option when executing test.py
  ```
  python3 test.py --yaml configs/coco.yaml --weight weights/weight_AP05:0.253207_280-epoch.pth --img data/3.jpg --torchscript
  ```
## NCNN
* Need to compile ncnn and opencv in advance and modify the path in build.sh
  ```
  cd example/ncnn/
  sh build.sh
  ./FastestDet
  ```
## onnx-runtime
* You can learn about the pre and post-processing methods of FastestDet in this Sample
  ```
  cd example/onnx-runtime
  pip install onnx-runtime
  python3 runtime.py
  ```
# Citation
* If you find this project useful in your research, please consider cite:
  ```
  @misc{=FastestDet,
        title={FastestDet: Ultra lightweight anchor-free real-time object detection algorithm.},
        author={xuehao.ma},
        howpublished = {\url{https://github.com/dog-qiuqiu/FastestDet}},
        year={2022}
  }
  ```
# Reference
* https://github.com/Tencent/ncnn
