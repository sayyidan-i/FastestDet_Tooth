sudo apt install gcc
sudo apt install g++
sudo apt install cmake
sudo apt-get install autoconf automake libtool curl make g++ unzip
./configure
 make
 make check
 sudo make install sudo ldconfig # refresh shared library cache.