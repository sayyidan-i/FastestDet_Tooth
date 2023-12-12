import requests
import cv2
import numpy as np
import imutils
import time

url = "http://192.168.100.17:8080/shot.jpg"

# Inisialisasi ukuran jendela OpenCV dan variabel untuk menghitung FPS
window_width = 800
window_height = 600
cv2.namedWindow("Android_cam", cv2.WINDOW_NORMAL)
fps = 0
start_time = time.time()
frame_count = 0

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    img_resized = imutils.resize(img, width=325, height=325, inter=cv2.INTER_NEAREST)
    cv2.resizeWindow("Android_cam", window_width, window_height)

    # Menghitung FPS
    frame_count += 1
    if frame_count >= 10:  # Menghitung FPS setiap 10 frame
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = time.time()
        frame_count = 0

    # Menampilkan FPS pada gambar
    cv2.putText(img_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Android_cam", img_resized)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
