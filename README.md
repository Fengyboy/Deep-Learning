# Deep-Learning
My code in deep learning and computer vision.

## Installation of Caffe on Ubuntu 14.04 LTS.
The first task is to make sure that you have the GNU compiler collection (GCC) tools installed. This is carried out by installing the build-essential package:
> sudo apt-get install build-essential

Download CUDA-7.5 from [here](https://developer.nvidia.com/cuda-downloads), and install it as:
> sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb 

> sudo apt-get update

> sudo apt-get install cuda 

Add the following lines to our .bash_profile file in our home directory, in order to obtain the required compilation tools on our PATH:
> export PATH=/usr/local/cuda-7.5/bin:$PATH

> export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

Install general dependencies for Caffe:
> sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

> sudo apt-get install --no-install-recommends libboost-all-dev

> sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

Download NVIDIA cuDNN from [here](https://developer.nvidia.com/cudnn) and install as follow:
> tar -xzvf cudnn-7.0-linux-x64-v3.0-prod.tgz

> sudo cp cuda/lib64/* /usr/local/cuda-7.5/lib64/

> sudo cp cuda/include/cudnn.h /usr/local/cuda-7.5/include/
