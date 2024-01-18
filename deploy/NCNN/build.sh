g++ -o FastestDet src/FastestDet.cpp -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp -std=c++17 -DOPENCV_GENERATE_PKGCONFIG=ON
./FastestDet
