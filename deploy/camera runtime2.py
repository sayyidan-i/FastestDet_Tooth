import cv2
import time

#find fps
prev_frame_time = 0
new_frame_time = 0

input_size = 160

source = cv2.VideoCapture(0)
if not source.isOpened():
    source = cv2.VideoCapture(1)

# Variables for FPS calculation
start_time = time.time()
frame_counter = 0

while True:
    start_cap = time.perf_counter()
    ret, frame = source.read()
    end_cap = time.perf_counter()
    time_cap = (end_cap - start_cap) * 1000.
    print("capture time:%fms"%time_cap)
    if not ret:
        print("Can't receive frame")
        break

    start_dis = time.perf_counter()
    cv2.imshow('Real-time Detection', frame)
    end_dis = time.perf_counter()
    time_dis = (end_dis- start_dis) * 1000.
    print("frame show time:%fms"%time_dis)
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    #fps = str(fps)
    fps = str(fps)
    print("FPS: ", fps)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
source.release()
cv2.destroyAllWindows()
