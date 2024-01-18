g++ -o Yolov8 src/Yolov8.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp -std=c++17 -DOPENCV_GENERATE_PKGCONFIG=ON
./Yolov8
