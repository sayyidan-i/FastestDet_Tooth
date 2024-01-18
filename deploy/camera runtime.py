import cv2
import time

def main():
    # Open the default camera (usually camera index 0)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get the frames per second (fps) of the camera
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Camera FPS:", fps)

    # Create a window for displaying the camera feed
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            print("FPS:", fps)
            frame_count = 0
            start_time = time.time()

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
