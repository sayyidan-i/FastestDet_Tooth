g++ -o FastestDet FastestDet.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp -std=c++17
