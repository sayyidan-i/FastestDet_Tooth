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
        start_cap = time.perf_counter()
        ret, img = source.read()
        end_cap = time.perf_counter()
        time_cap = (end_cap - start_cap) * 1000.
        print("capture time:%fms"%time_cap)
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        # Display FPS on the screen
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        

        # Display the resized image
        start_dis = time.perf_counter()
        cv2.imshow(window_name, img) 
        end_dis = time.perf_counter()
        time_dis = (end_dis- start_dis) * 1000.
        print("frame show time:%fms"%time_dis)
        
        print(f"FPS: {fps}")
        
        
        if cv2.waitKey(1) == 27:
            break

    source.release()
    cv2.destroyAllWindows()
