import requests
import cv2
import numpy as np
import imutils
import time
import threading

url = "http://192.168.100.17:8080/shot.jpg"

def get_image():
    global img  # Variabel gambar dideklarasikan sebagai global

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=320)  # Mengatur resolusi menjadi lebih kecil
        time.sleep(0.1)  # Mengatur waktu antara setiap pengambilan gambar

# Memulai thread untuk pengambilan gambar
image_thread = threading.Thread(target=get_image)
image_thread.daemon = True  # Mengatur thread sebagai daemon agar bisa dihentikan dengan CTRL+C
image_thread.start()

while True:
    if 'img' in globals():  # Memeriksa apakah gambar sudah ada sebelum menampilkan
        cv2.imshow("Android_cam", img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
