g++ -g -o FastestDet_Live FastestDet_Livecam.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv` -fopenmp -std=c++17
./FastestDet_Live
