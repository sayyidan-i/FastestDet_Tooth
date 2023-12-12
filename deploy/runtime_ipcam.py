import cv2
import requests
import numpy as np
import imutils
import onnxruntime
import threading

# Sigmoid dan Tanh functions
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

# Preprocessing function
def preprocess(src_img, size):
    output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    output = output.transpose(2, 0, 1)
    output = output.reshape((1, 3, size[1], size[0])) / 255
    return output.astype('float32')

# Non-maximum suppression function
# Non-maximum suppression function
def nms(dets, thresh=0.45):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output

# Object detection function
def detection(session, img, input_width, input_height, thresh):
    pred = []

    H, W, _ = img.shape
    data = preprocess(img, [input_width, input_height])

    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    feature_map = feature_map.transpose(1, 2, 0)
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            if score > thresh:
                cls_index = np.argmax(data[5:])
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])

    return nms(np.array(pred))

# Camera stream function
def stream_camera(url):
    global img  # Make img a global variable
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=500)  # Set resolution if needed
        cv2.imshow("Camera Stream", img)
        if cv2.waitKey(1) == 27:
            break

        bboxes = detection(session, img, input_width, input_height, thresh)
        for b in bboxes:
            obj_score, cls_index = b[4], int(b[5])
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            label = names[cls_index]
            color = label_colors.get(label, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 5 if y1 >= 5 else y1 + 20
            if y1 < img.shape[0] // 2:
                text_y = y2 + 20
            else:
                text_y = y1 - 5 - 20
            cv2.putText(img, '%.2f' % obj_score, (x1, text_y), 0, 0.7, color, 2)
            cv2.putText(img, label, (x1, text_y - 20), 0, 0.7, color, 2)
            cv2.imshow("Object Detection", img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Placeholder for initialization and other variables
    camera_url = "http://192.168.100.17:8080/shot.jpg"
    input_width, input_height = 352, 352  # Placeholder for YOLO model input width and height
    model_onnx = 'epoch230.onnx'  # Placeholder for ONNX model file name
    label = "tooth.names"  # Placeholder for label file
    thresh = 0.6  # Detection threshold

    session = onnxruntime.InferenceSession(model_onnx)
    names = []
    with open(label, 'r') as f:
        for line in f.readlines():
            names.append(line.strip())

    label_colors = {
        "Normal": (0, 255, 0),
        "Karies kecil": (0, 0, 255),
        "Karies sedang": (0, 0, 130),
        "Karies besar": (0, 0, 50),
        "Stain": (255, 0, 255),
        "Karang gigi": (0, 255, 255),
        "Lain-Lain": (128, 128, 128)
    }

    # Start camera streaming in a separate thread
    camera_thread = threading.Thread(target=stream_camera, args=(camera_url,))
    camera_thread.daemon = True
    camera_thread.start()

    # Initialize img variable
    img = None

    while True:
        if img is not None:
            pass  # Object detection happens inside stream_camera function

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
