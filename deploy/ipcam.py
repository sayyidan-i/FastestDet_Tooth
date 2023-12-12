import requests 
import cv2 
import numpy as np 
import imutils 
  
url = "http://192.168.100.17:8080/shot.jpg"
  
while True: 
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    img = cv2.imdecode(img_arr, -1) 

    # Mengurangi resolusi gambar
    img = imutils.resize(img, width=500, height=900, inter=cv2.INTER_NEAREST) 
    cv2.imshow("Android_cam", img) 
  
    if cv2.waitKey(1) == 27: 
        break
  
cv2.destroyAllWindows()
