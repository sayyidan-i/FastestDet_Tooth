g++ -o FastestDet_fp16 FastestDet_fp16.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv` -fopenmp -std=c++17
