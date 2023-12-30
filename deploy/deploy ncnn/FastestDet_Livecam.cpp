#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include "net.h"
#include "benchmark.h"

float Sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

float Tanh(float x) { return 2.0f / (1.0f + exp(-2 * x)) - 1; }

class TargetBox {
private:
    float GetWidth() { return (x2 - x1); };
    float GetHeight() { return (y2 - y1); };

public:
    int x1;
    int y1;
    int x2;
    int y2;

    int category;
    float score;

    float area() { return GetWidth() * GetHeight(); };
};

float IntersectionArea(const TargetBox &a, const TargetBox &b) {
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b) { return (a.score > b.score); }

int nmsHandle(std::vector<TargetBox> &src_boxes, std::vector<TargetBox> &dst_boxes) {
    if (src_boxes.empty()) {
        std::cout << "Error: No source boxes provided for NMS" << std::endl;
        return -1;
    }

    std::vector<int> picked;

    // Sort the source boxes
    sort(src_boxes.begin(), src_boxes.end(), scoreSort);

    for (int i = 0; i < src_boxes.size(); i++) {
        if (i >= src_boxes.size()) {
            std::cout << "Error: Invalid index detected in source boxes during iteration" << std::endl;
            return -1;
        }

        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            if (j >= picked.size() || picked[j] >= src_boxes.size() || picked[j] < 0 || i < 0) {
                std::cout << "Error: Invalid index detected in picked list during iteration" << std::endl;
                return -1;
            }

            float inter_area = IntersectionArea(src_boxes[i], src_boxes[picked[j]]);
            float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;
            if (IoU > 0.45 && src_boxes[i].category == src_boxes[picked[j]].category) {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++) {
        if (picked[i] >= src_boxes.size() || picked[i] < 0) {
            std::cout << "Error: Invalid index detected in picked list during box copy" << std::endl;
            return -1;
        }

        dst_boxes.push_back(src_boxes[picked[i]]);
    }

    return 0;
}

int main() {
    static const char *class_names[] = {
        "Normal", "Karies kecil", "Karies sedang", "Karies besar", "Stain", "Karang gigi", "Lain-Lain"};
    int class_num = sizeof(class_names) / sizeof(class_names[0]);
    float thresh = 0.5;
    int input_width = 352;
    int input_height = 352;

    ncnn::Net net;
    net.load_param("epoch230-sim-opt-fp16.param");
    net.load_model("epoch230-sim-opt-fp16.bin");

    cv::VideoCapture cap("/dev/video0");

    if (!cap.isOpened()) {
        std::cout << "Error: Could not access the camera" << std::endl;
        return -1;
    }
    
    // Buat pemetaan antara kelas dan warna
	std::map<std::string, cv::Scalar> color_map = {
    {"Normal", cv::Scalar(0, 255, 0)},         // Hijau
    {"Karies kecil", cv::Scalar(0, 255, 255)}, // Kuning
    {"Karies sedang", cv::Scalar(255, 0, 0)},  // Biru
    {"Karies besar", cv::Scalar(0, 0, 255)},   // Merah
    {"Stain", cv::Scalar(128, 0, 128)},        // Ungu
    {"Karang gigi", cv::Scalar(255, 192, 203)},// Pink
    {"Lain-Lain", cv::Scalar(128, 128, 128)}   // Abu-abu
};

    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    int frameCount = 0;
    double fps;
    auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Error: No frame captured" << std::endl;
            break;
        }

        // Resize the frame to the desired input dimensions
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(input_width, input_height));

        // Convert and normalize the resized frame
        ncnn::Mat input = ncnn::Mat::from_pixels(resized_frame.data, ncnn::Mat::PIXEL_BGR, input_width, input_height);
        const float mean_vals[3] = {0.f, 0.f, 0.f};
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        input.substract_mean_normalize(mean_vals, norm_vals);

        // Perform inference
        ncnn::Extractor ex = net.create_extractor();
        ex.set_num_threads(4);
        ex.input("input.1", input);

        ncnn::Mat output;
        ex.extract("732", output);

        std::vector<TargetBox> target_boxes;
 

        for (int h = 0; h < output.h; h++) {
            for (int w = 0; w < output.w; w++) {
                int obj_score_index = (0 * output.h * output.w) + (h * output.w) + w;
                float obj_score = output[obj_score_index];

                int category;
                float max_score = 0.0f;
                for (size_t i = 0; i < class_num; i++) {
                    int obj_score_index = ((5 + i) * output.h * output.w) + (h * output.w) + w;
                    float cls_score = output[obj_score_index];
                    if (cls_score > max_score) {
                        max_score = cls_score;
                        category = i;
                    }
                }
                float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

                if (score > thresh) {
                    int x_offset_index = (1 * output.h * output.w) + (h * output.w) + w;
                    int y_offset_index = (2 * output.h * output.w) + (h * output.w) + w;
                    int box_width_index = (3 * output.h * output.w) + (h * output.w) + w;
                    int box_height_index = (4 * output.h * output.w) + (h * output.w) + w;

                    float x_offset = Tanh(output[x_offset_index]);
                    float y_offset = Tanh(output[y_offset_index]);
                    float box_width = Sigmoid(output[box_width_index]);
                    float box_height = Sigmoid(output[box_height_index]);

                    float cx = (w + x_offset) / output.w;
                    float cy = (h + y_offset) / output.h;

                    int x1 = (int)((cx - box_width * 0.5) * frame.cols);
                    int y1 = (int)((cy - box_height * 0.5) * frame.rows);
                    int x2 = (int)((cx + box_width * 0.5) * frame.cols);
                    int y2 = (int)((cy + box_height * 0.5) * frame.rows);

                    target_boxes.push_back(TargetBox{x1, y1, x2, y2, category, score});
                }
            }
        }
		
        std::vector<TargetBox> nms_boxes;
        
        //if (nms_boxes.size() !=0) {
        if (!target_boxes.empty()) {
            int nmsResult = nmsHandle(target_boxes, nms_boxes);
            if (nmsResult != 0) {
                std::cout << "Error: Non-Maximum Suppression failed" << std::endl;
                break;
            }
   
		//std::cout << "Size of target_boxes: " << target_boxes.size() << std::endl;
		//std::cout << "Size of nms_boxes: " << nms_boxes.size() << std::endl;
		//if (nms_boxes.size() !=0) {
        for (size_t i = 0; i < nms_boxes.size(); i++) {
			TargetBox box = nms_boxes[i];
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << box.score * 100;
			std::string text = std::string(class_names[box.category]) + ":" + stream.str() + "%";

			// Dapatkan warna berdasarkan kelas dari pemetaan
			cv::Scalar color = color_map[class_names[box.category]];

			cv::rectangle(frame, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), color, 2);
			cv::putText(frame, text, cv::Point(box.x1, box.y1), cv::FONT_HERSHEY_SIMPLEX, 0.75, color, 2);
		}
	}

        cv::imshow("Camera", frame);
        

        frameCount++;
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

        if (elapsed >= 1.0) {
            fps = frameCount / elapsed;
            frameCount = 0;
            start = end;
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
