import numpy as np 
import cv2

def draw_circle(event,x,y,flags,param):
    global loop
    #if event == cv2.EVENT_LBUTTONDBLCLK:
 
    if event == cv2.EVENT_RBUTTONDOWN:
        loop=False
        
        
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
loop=True


while loop:
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print (mouseX)
        print (mouseY)