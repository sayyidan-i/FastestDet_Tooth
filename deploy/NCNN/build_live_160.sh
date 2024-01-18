g++ -g -o FastestDet_Live_160 src/FastestDet_Live_160.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp -std=c++17
./FastestDet_Live_160
