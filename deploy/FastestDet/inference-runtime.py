import cv2
import time
import numpy as np
import onnxruntime
import screeninfo
import argparse
import os
import imutils

# sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# tanh函数
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

# 数据预处理
def preprocess(src_img, size):
    output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
    output = output.transpose(2,0,1)
    output = output.reshape((1, 3, size[1], size[0])) / 255

    return output.astype('float32')

# nms算法
def nms(dets, thresh=0.45):
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # #thresh:0.3,0.5....
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标

    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)

        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]

        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    
    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output

# 人脸检测
def detection(session, img, input_width, input_height, thresh):
    pred = []

    # 输入图像的原始宽高
    H, W, _ = img.shape

    # 数据预处理: resize, 1/255
    data = preprocess(img, [input_width, input_height])

    # 模型推理
    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    # 输出特征图转置: CHW, HWC
    feature_map = feature_map.transpose(1, 2, 0)
    # 输出特征图的宽高
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    # 特征图后处理
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            # 解析检测框置信度
            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            # 阈值筛选
            if score > thresh:
                # 检测框类别
                cls_index = np.argmax(data[5:])
                # 检测框中心点偏移
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                # 检测框归一化后的宽高
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                # 检测框归一化后中心点
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])

    return nms(np.array(pred))
    
def display_image(image_path):
	image = cv2.imread(image_path)
	screen = screeninfo.get_monitors()[0]
	
	if image is None:
		print("could not open or find the image.")
	else:
		window_name ='result'
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
		cv2.moveWindow(window_name, screen.x -1, screen.y -1)
		cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow(window_name, image)
		cv2.waitKey()
		cv2.waitKey()
		cv2.destroyAllWindows()
	


if __name__ == '__main__':
    # source
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="weight/FastestDet_160_AP2.6.onnx")
    opt = parser.parse_args()
    assert os.path.exists(opt.weight)
    
    model_onnx = opt.weight
    label = "tooth.names"
    thresh = 0.6
    
    # Load label names
    names = []
    with open(label, 'r') as f:
        for line in f.readlines():
            names.append(line.strip())
    
    # Initialize label colors
    label_colors = {
        "Normal": (0, 255, 0),
        "Karies kecil": (0, 0, 255),
        "Karies sedang": (0, 0, 130),
        "Karies besar": (0, 0, 50),
        "Stain": (255, 0, 255),
        "Karang gigi": (0, 255, 255),
        "Lain-Lain": (128, 128, 128)
    }
    
    input_width, input_height = 160, 160
    
    for i in range(1, 13):
        source = f"image/inferensi ({i}).jpg"       
        img = cv2.imread(source)
        # img = imutils.resize(img, width=640, height=640, inter=cv2.INTER_NEAREST)
        # Model input width and height
        # Load model
        session = onnxruntime.InferenceSession(model_onnx)
        # Object detection
        start = time.perf_counter()
        bboxes = detection(session, img, input_width, input_height, thresh)
        end = time.perf_counter()
        inference_time = (end - start) * 1000.
        
        text_size, _ = cv2.getTextSize('Inference time: %.2fms' % inference_time, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        box_height = text_size[1] + 30  # Increase the height of the box
        cv2.rectangle(img, (0, 0), (10 + text_size[0] + 10, 10 + box_height), (0, 0, 0), -1)
        cv2.putText(img, 'Inference time: %.2fms' % inference_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        print("Inference time: %fms" % inference_time)
        
        # Create the "inference_result" folder if it doesn't exist
        os.makedirs("inference_result", exist_ok=True)
        
        output_file = f"inference_result/inference_{i}.jpg"
        
        print("=================box info===================")
        for i, b in enumerate(bboxes):
            print(b)
            obj_score, cls_index = b[4], int(b[5])
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            label = names[cls_index]
            
            # Get color based on label
            color = label_colors.get(label, (255, 255, 255))  # Use white if label is not found
            
            # Modify the detection to use the specified color
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Determine the text label coordinates
            text_y = y1 - 5 if y1 >= 5 else y1 + 20  # Shift the text down if close to the top edge
            if y1 < img.shape[0] // 2:  # If the object is in the top half of the image
                text_y = y2 + 20  # Place the text below the object
            else:  # If the object is in the bottom half of the image
                text_y = y1 - 5 - 20  # Place the text above the object
            
            cv2.putText(img, '%.2f' % obj_score, (x1, text_y), 0, 1.2, color, 2)
            cv2.putText(img, label, (x1, text_y - 40), 0, 1.2, color, 2)
        
        cv2.imwrite(output_file, img)
    









