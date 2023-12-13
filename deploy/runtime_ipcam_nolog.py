import cv2
import time
import numpy as np
import onnxruntime
import requests
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
                         
    if len(pred) > 0:
        return nms(np.array(pred))
    else:
        print("Arahkan kamera dengan benar")

if __name__ == '__main__':
    
    frames = 0
    fps = 0
    start_time = time.time()
    
    # source
    source = "http://192.168.100.17:8080/shot.jpg"
    model_onnx = 'epoch230.onnx'
    label = "tooth.names"
    thresh = 0.5
    # Inisialisasi ukuran jendela OpenCV
    window_width = 800
    window_height = 600
    cv2.namedWindow("Android_cam", cv2.WINDOW_NORMAL)  # Menentukan jenis jendela untuk menyesuaikan ukuran

        # 加载label names
    names = []
    with open(label, 'r') as f:
        for line in f.readlines():
            names.append(line.strip())
            
    # Inisialisasi warna untuk setiap label
    label_colors = {
        "Normal": (0, 255, 0),         
        "Karies kecil": (0, 0, 255),   
        "Karies sedang": (0, 0, 130),  
        "Karies besar": (0, 0, 50), 
        "Stain": (255, 0, 255),        
        "Karang gigi": (0, 255, 255),  
        "Lain-Lain": (128, 128, 128) 
    }
    
    # Menyesuaikan ukuran jendela OpenCV
    cv2.resizeWindow("Android_cam", window_width, window_height)
    
    session = onnxruntime.InferenceSession(model_onnx)
    

    while True:
        img_resp = requests.get(source)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
              
        
        # Menyesuaikan ukuran gambar tanpa mengubah resolusi
        img_resized = imutils.resize(img, width=800, height=800, inter=cv2.INTER_NEAREST)

        
        input_width, input_height = 352, 352
        bboxes = detection(session, img_resized, input_width, input_height, thresh)

        if bboxes is not None:
            #print("=================box info===================")
            for i, b in enumerate(bboxes):
                #print(b)
                obj_score, cls_index = b[4], int(b[5])
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                label = names[cls_index]

                # Mengambil warna sesuai dengan label
                color = label_colors.get(label, (255, 255, 255))  # Jika label tidak ditemukan, gunakan putih

                # Memodifikasi deteksi untuk menggunakan warna yang ditentukan
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)

                # Menentukan koordinat teks label
                text_y = y1 - 5 if y1 >= 5 else y1 + 20  # Menggeser teks ke bawah jika dekat batas atas
                if y1 < img_resized.shape[0] // 2:  # Jika objek berada di bagian atas gambar
                    text_y = y2 + 20  # Tempatkan teks di bawah objek
                else:  # Jika objek berada di bagian bawah gambar
                    text_y = y1 - 5 - 20  # Tempatkan teks di atas objek

                cv2.putText(img_resized, '%.2f' % obj_score, (x1, text_y), 0, 0.7, color, 2)
                cv2.putText(img_resized, label, (x1, text_y - 20), 0, 0.7, color, 2)
        else:
            print("Arahkan kamera dengan benar")        
        
        # Hitung FPS setiap detik
        frames += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frames / elapsed_time
            #print(f"FPS: {fps:.2f}")
            frames = 0
            start_time = time.time()

        # Menampilkan FPS di layar
        cv2.putText(img_resized, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        
        # Menampilkan gambar dalam jendela yang telah diatur ukurannya
        cv2.imshow("Android_cam", img_resized)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()










