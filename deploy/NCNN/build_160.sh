g++ -o FastestDet_160 src/FastestDet_160.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp -std=c++17 -DOPENCV_GENERATE_PKGCONFIG=ON
./FastestDet_160
