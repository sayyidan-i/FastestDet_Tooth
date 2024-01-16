
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

# Reference
* https://github.com/dog-qiuqiu/FastestDet
* https://github.com/Tencent/ncnn
* https://github.com/jahongir7174/YOLOv8-onnx
