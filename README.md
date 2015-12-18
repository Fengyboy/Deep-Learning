# Deep-Learning
My code in deep learning and computer vision.

## Installation of Caffe on Ubuntu 14.04 LTS.
The first task is to make sure that you have the GNU compiler collection (GCC) tools installed. This is carried out by installing the build-essential package:
```
sudo apt-get install build-essential
```

Download CUDA-7.5 from [here](https://developer.nvidia.com/cuda-downloads), and install it as:
```
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb 
sudo apt-get update
sudo apt-get install cuda 
```

Add the following lines to our .bashrc file in our home directory, in order to obtain the required compilation tools on our PATH:
```
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
```

Install general dependencies for Caffe:
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler gfortran liblapack-dev
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev ibeigen3-dev
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo apt-get install gtk2-engines-pixbuf
```

Download NVIDIA cuDNN from [here](https://developer.nvidia.com/cudnn) and install as follow:
```
tar -xzvf cudnn-7.0-linux-x64-v3.0-prod.tgz
sudo cp cuda/lib64/* /usr/local/cuda-7.5/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda-7.5/include/
```

For best performance, Caffe can be accelerated by NVIDIA cuDNN. Register for free at the cuDNN site, install it, then continue with these installation instructions. To compile with cuDNN set the USE_CUDNN := 1 flag set in your Makefile.config.

Caffe requires BLAS as the backend of its matrix and vector computations. There are several implementations of this library. The choice is yours:
* ATLAS: free, open source, and so the default for Caffe.
* Intel MKL: commercial and optimized for Intel CPUs, with a free trial and student licenses.
  1. Install MKL.
  2. Set BLAS := mkl in Makefile.config
* OpenBLAS: free and open source; this optimized and parallel BLAS could require more effort to install, although it might offer a speedup.
  1. Install OpenBLAS
  2. Set BLAS := open in Makefile.config

### Python

The main requirements are numpy and boost.python (provided by boost). pandas is useful too and needed for some examples.

You can install the dependencies with
```
sudo -H pip install -r requirements.txt
```

### Compilation
Now that you have the prerequisites, edit your Makefile.config to change the paths for your setup The defaults should work, but uncomment the relevant lines if using Anaconda Python.
```
cp Makefile.config.example Makefile.config
sudo make -j $(nproc)
sudo make test -j $(nproc)
make runtest
```

To compile the Python and MATLAB wrappers do make pycaffe and make matcaffe respectively. Be sure to set your MATLAB and Python paths in Makefile.config first. Then compile the Python inferface for Caffe as:
```
sudo make pycaffe
```

Then add the path to Caffe Python files in .bashrc file as:
```
export PYTHONPATH=/home/feng/shortbite/caffe/python:$PYTHONPATH
```
