import cv2
import time

if __name__ == '__main__':
    
    # Find fps
    prev_frame_time = 0
    new_frame_time = 0
    
    # Source
    #source = cv2.VideoCapture(0)
    #if not source.isOpened():
    source = cv2.VideoCapture(1)
    
    window_name = "usb camera"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, img = source.read()
        
        # Resize the image to 640x640
        #img = cv2.resize(img, (640, 640))
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        # Display FPS on the screen
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"FPS: {fps}")

        # Display the resized image
        cv2.imshow(window_name, img) 
        
        if cv2.waitKey(1) == 27:
            break

    source.release()
    cv2.destroyAllWindows()
