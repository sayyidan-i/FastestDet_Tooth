import requests
import cv2
import numpy as np
import imutils

url = "http://192.168.100.17:8080/shot.jpg"

# Inisialisasi ukuran jendela OpenCV
window_width = 800
window_height = 600
cv2.namedWindow("Android_cam", cv2.WINDOW_NORMAL)  # Menentukan jenis jendela untuk menyesuaikan ukuran

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    # Menyesuaikan ukuran gambar tanpa mengubah resolusi
    img_resized = imutils.resize(img, width=325, height=325, inter=cv2.INTER_NEAREST)

    # Menyesuaikan ukuran jendela OpenCV
    cv2.resizeWindow("Android_cam", window_width, window_height)
    
    # Menampilkan gambar dalam jendela yang telah diatur ukurannya
    cv2.imshow("Android_cam", img_resized)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
