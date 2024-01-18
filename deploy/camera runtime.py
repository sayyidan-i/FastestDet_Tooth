import cv2
import time
import numpy as np        

if __name__ == '__main__':
    
    
    #find fps
    prev_frame_time = 0
    new_frame_time = 0
        
    # source
    source = cv2.VideoCapture(1)
    window_name="usb camera"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    
           
    while True:
        ret, img = source.read()
               
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        #fps = str(fps)
        fps = str(int(fps))

        # Display FPS on the screen
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"FPS: {fps}")

        # Display the image
        cv2.imshow(window_name, img)

        #if cv2.waitKey(1) == ord('c'):  # Tekan 'c' untuk capture gambar
            #capture_image(img)  
        
        if cv2.waitKey(1) == 27:
            break
        

    cv2.destroyAllWindows()
