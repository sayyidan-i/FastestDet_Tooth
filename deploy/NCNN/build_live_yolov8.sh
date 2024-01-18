g++ -g -o Yolov8_Live src/Yolov8_Live.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp -std=c++17
./Yolov8_Live
