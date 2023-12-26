import cv2
import time
import numpy as np
import onnxruntime
import requests
import imutils

# sigmoid function
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# tanh function
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

# Data preprocessing
def preprocess(src_img, size):
    output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    output = output.transpose(2, 0, 1)
    output = output.reshape((1, 3, size[1], size[0])) / 255

    return output.astype('float32')

# nms algorithm
def nms(dets, thresh=0.45):
    # dets: N*M, N is the number of bboxes, M's first 4 elements are (x1, y1, x2, y2), the 5th element is the score
    # #thresh: 0.3, 0.5, etc.
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # Calculate the area of each bbox
    order = scores.argsort()[::-1]  # Sort the scores in descending order
    keep = []  # To store the indices of the bboxes to keep

    while order.size > 0:
        i = order[0]  # Always keep the bbox with the highest confidence in each iteration
        keep.append(i)

        # Calculate the intersection area between the bbox with the highest confidence and the remaining bboxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Calculate the intersection area of the bbox with the highest confidence and the remaining bboxes
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # Calculate the ratio of the intersection area to the sum of the areas of the two bboxes (the one with highest confidence and the other bbox)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep bboxes with ovr less than thresh and proceed to the next iteration
        inds = np.where(ovr <= thresh)[0]

        # Since the indices in ovr do not include order[0], we need to shift by one
        order = order[inds + 1]

    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output


def detection(session, img, input_width, input_height, thresh):
    pred = []

    # Original width and height of the input image
    H, W, _ = img.shape

    # Data preprocessing: resize, 1/255
    data = preprocess(img, [input_width, input_height])

    # Model inference
    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    # Transpose the output feature map: CHW, HWC
    feature_map = feature_map.transpose(1, 2, 0)
    # Width and height of the output feature map
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    # Post-processing of the feature map
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            # Parsing detection box confidence
            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            # Threshold filtering
            if score > thresh:
                # Detection box category
                cls_index = np.argmax(data[5:])
                # Offset of the detection box center point
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                # Normalized width and height of the detection box
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                # Normalized center point of the detection box
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height

                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])
    if len(pred)>0:
        return nms(np.array(pred))

def capture_image(img):
    global image_counter
    img_name = f"result/captured_image_{image_counter}.jpg"
    cv2.imwrite(img_name, img)
    image_counter += 1

if __name__ == '__main__':
    
    #find fps
    prev_frame_time = 0
    new_frame_time = 0
        
    # source
    source = cv2.VideoCapture(0)
    model_onnx = 'epoch230.onnx'
    label = "tooth.names"
    thresh = 0.5
    
    #OpenCV window
    #window_width = 800
   # indow_height = 600
    window_name = "usb_cam"
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Load label names
    names = []
    with open(label, 'r') as f:
        for line in f.readlines():
            names.append(line.strip())
            
    # Initialize colors for each label
    label_colors = {
        "Normal": (0, 255, 0),        # Normal - Hijau
        "Karies kecil": (0, 255, 255),   # Karies kecil - Kuning
        "Karies sedang": (0, 140, 255),  # Karies sedang - Orange
        "Karies besar": (0, 0, 255),    # Karies besar - Merah
        "Stain": (255, 0, 0),           # Stain - Biru
        "Karang gigi": (128, 0, 128),   # Karang gigi - Ungu
        "Lain-Lain": (128, 128, 128)    # Lain-Lain - Abu-abu
}

    
    # Adjust OpenCV window size
    #cv2.resizeWindow(window_name, window_width, window_height)
    #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty(window_name, 800, 800)
    
    session = onnxruntime.InferenceSession(model_onnx)
    
    #buat save gambar
    image_counter = 0
           
    while True:
        ret, img = source.read()
       
        # Resize image without changing resolution
        img_resized =  img #imutils.resize(img, width=800, height=800, inter=cv2.INTER_NEAREST)
        #img_resized = cv2.resize(img, (352, 352), interpolation=cv2.INTER_AREA)

        
        input_width, input_height = 352, 352
        bboxes = detection(session, img_resized, input_width, input_height, thresh)

        if bboxes is not None:
            #print("=================box info===================")
            for i, b in enumerate(bboxes):
                #print(b)
                obj_score, cls_index = b[4], int(b[5])
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                label = names[cls_index]

                # Fetch color according to label
                color = label_colors.get(label, (255, 255, 255))  # Use white if label not found

                # Modify detection to use the specified color
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)

                # Determine text label coordinates
                text_y = y1 - 5 if y1 >= 5 else y1 + 20  # Shift text downwards if close to top boundary
                if y1 < img_resized.shape[0] // 2:  # If the object is in the upper part of the image
                    text_y = y2 + 20  # Place the text below the object
                else:  # If the object is in the lower part of the image
                    text_y = y1 - 5 - 20  # Place the text above the object

                cv2.putText(img_resized, '%.2f' % obj_score, (x1, text_y), 0, 0.5, color, 1)
                cv2.putText(img_resized, label, (x1, text_y - 20), 0, 0.5, color, 1)
        else:
            cv2.putText(img_resized, "Correct the camera direction", (550,430), 0, 0.5, (0, 255, 255), 1)
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        # Display FPS on the screen
        cv2.putText(img_resized, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"FPS: {fps}")

        # Display the image
        #cv2.imshow(window_name, img_resized)

        if cv2.waitKey(1) == ord('c'):  # Tekan 'c' untuk capture gambar
            capture_image(img_resized)  
        
        if cv2.waitKey(1) == 27:
            break
        

    cv2.destroyAllWindows()
