#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Error: Could not access the camera" << std::endl;
        return -1;
    }

    cv::namedWindow("Camera", cv::WINDOW_NORMAL);

    int frameCount = 0;
    double fps;
    auto start = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Error: No frame captured" << std::endl;
            break;
        }

        cv::imshow("Camera", frame);

        // Increment frame count
        frameCount++;

        // Calculate elapsed time
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

        // Calculate FPS every second
        if (elapsed >= 1.0) {
            fps = frameCount / elapsed;

            // Reset frame count and start time for the next calculation
            frameCount = 0;
            start = end;

            // Display FPS in the terminal
            std::cout << "FPS: " << fps << std::endl;
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
