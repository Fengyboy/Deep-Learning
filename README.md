# Deep-Learning
My code in deep learning and computer vision.

## Installation of Caffe on Ubuntu 16.04 LTS.
The first task is to make sure that you have the GNU compiler collection (GCC) tools installed. This is carried out by installing the build-essential package:
```
sudo apt-get install build-essential
```

Download CUDA-8.0 from [here](https://developer.nvidia.com/cuda-downloads), and install it as:
```
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb 
sudo apt-get update
sudo apt-get install cuda 
```

Add the following lines to our .bashrc file in our home directory, in order to obtain the required compilation tools on our PATH:
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Install general dependencies for Caffe:
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler gfortran liblapack-dev
sudo apt-get install python3-dev
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev libeigen3-dev
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo apt-get install gtk2-engines-pixbuf
sudo apt-get install python-opencv libatlas-base-dev
```

Download the latest NVIDIA cuDNN from [here](https://developer.nvidia.com/cudnn) and install as follow:
```
tar -xzvf cudnn-8.0-linux-x64-vX.X-prod.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
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
Now that you have the prerequisites, edit your Makefile.config to change the paths for your setup The defaults should work, but uncomment the relevant lines if using Anaconda Python. For hdf5 to work, you need add the following lines to Make
```
# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```
Then execute the following commands.
```
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s libhdf5_serial.10.1.0 libhdf5.so
sudo ln -s libhdf5_serial_hl.so.10 libhdf5_hl.so
cd ~
cp Makefile.config.example Makefile.config
make -j $(nproc)
make test -j $(nproc)
make runtest
```

To compile the Python and MATLAB wrappers do make pycaffe and make matcaffe respectively. Be sure to set your MATLAB and Python paths in Makefile.config first. Then compile the Python inferface for Caffe as:
```
make pycaffe
```

Then add the path to Caffe Python files in .bashrc file as:
```
export PYTHONPATH=/home/feng/shortbite/caffe/python:$PYTHONPATH
```
If en error like `libdc1394 error: Failed to initialize libdc1394` pops up, you need to run the following:
```
sudo ln /dev/null /dev/raw1394
```

A Python3 virtual evironment can be created by:
```
virtualenv -p /usr/bin/python3 yourenv
source yourenv/bin/activate
pip install package-name
```
Delete all docker containers by:
```
docker rm `docker ps --no-trunc -aq`
```
Set tiff to the correct version for openslide
```
sudo apt-get install libtiff5=4.0.6-1 libtiff5-dev=4.0.6-1 libtiffxx5=4.0.6-1
```
User SLURM to request GPU resources
```
srun -n1 -w HPC1004 --pty /bin/bash -i
```

https://github.com/BVLC/caffe/wiki/OpenCV-3.3-Installation-Guide-on-Ubuntu-16.04

I tried the following and it worked:

Change in FindCUDA.cmake the nppi library to the several splitted ones. This has to be done in 3 places. Remember this change is just to make it work with CUDA 9.0, I am not doing checks for version or anything, which should be done if you plan to give it to different people with different CUDA versions.

1) look for the line with:

find_cuda_helper_libs(nppi)
and replace it with the lines:

  find_cuda_helper_libs(nppial)
  find_cuda_helper_libs(nppicc)
  find_cuda_helper_libs(nppicom)
  find_cuda_helper_libs(nppidei)
  find_cuda_helper_libs(nppif)
  find_cuda_helper_libs(nppig)
  find_cuda_helper_libs(nppim)
  find_cuda_helper_libs(nppist)
  find_cuda_helper_libs(nppisu)
  find_cuda_helper_libs(nppitc)
2) find the line:

set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppi_LIBRARY};${CUDA_npps_LIBRARY}")
and change it to

set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppial_LIBRARY};${CUDA_nppicc_LIBRARY};${CUDA_nppicom_LIBRARY};${CUDA_nppidei_LIBRARY};${CUDA_nppif_LIBRARY};${CUDA_nppig_LIBRARY};${CUDA_nppim_LIBRARY};${CUDA_nppist_LIBRARY};${CUDA_nppisu_LIBRARY};${CUDA_nppitc_LIBRARY};${CUDA_npps_LIBRARY}")
3) find the unset variables and add the new variables as well So, find:

unset(CUDA_nppi_LIBRARY CACHE)
and change it to:

unset(CUDA_nppial_LIBRARY CACHE)
unset(CUDA_nppicc_LIBRARY CACHE)
unset(CUDA_nppicom_LIBRARY CACHE)
unset(CUDA_nppidei_LIBRARY CACHE)
unset(CUDA_nppif_LIBRARY CACHE)
unset(CUDA_nppig_LIBRARY CACHE)
unset(CUDA_nppim_LIBRARY CACHE)
unset(CUDA_nppist_LIBRARY CACHE)
unset(CUDA_nppisu_LIBRARY CACHE)
unset(CUDA_nppitc_LIBRARY CACHE)
And also in OpenCVDetectCUDA.cmake you have to remove the 2.0 architechture which is no longer supported.

It has:

  ...
  set(__cuda_arch_ptx "")
  if(CUDA_GENERATION STREQUAL "Fermi")
    set(__cuda_arch_bin "2.0")
  elseif(CUDA_GENERATION STREQUAL "Kepler")
    set(__cuda_arch_bin "3.0 3.5 3.7")
  ...
It should be:

  ...
  set(__cuda_arch_ptx "")
  if(CUDA_GENERATION STREQUAL "Kepler")
    set(__cuda_arch_bin "3.0 3.5 3.7")
  elseif(CUDA_GENERATION STREQUAL "Maxwell")
    set(__cuda_arch_bin "5.0 5.2")
  ...
Basically I removed the first if and the first elif turns to an if.

As mentionned by @matko It also has :

set(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0 5.2 6.0 6.1") 
Which should be changed to:

set(__cuda_arch_bin "3.0 3.5 3.7 5.0 5.2 6.0 6.1") 
One last thing it is needed. CUDA 9.0 has a separated file for the halffloat (cuda_fp16.h) now. This needs to be included in OpenCV.

From CUDA 9.0 manual:

Unsupported Features General CUDA ‣ CUDA library. The built-in functions __float2half_rn() and __half2float() have been removed. Use equivalent functionality in the updated fp16 header file from the CUDA toolkit.

To do this, you need to add:

#include <cuda_fp16.h>
in the header file

opencv-3.3.0\modules\cudev\include\opencv2\cudev\common.hpp
This are the basics for a definite patch for OpenCV. What it is missing, well as I told you before, I do not care about CUDA versions (it needs an IF). Also, CUDA 9.0 has a bunch of deprecated functions used by OpenCV ... this probably will be replaced by the OpenCV team at some point. It is also possible that one or more of the splitted libraries of nppi is not used.

Final recommendations: For this kind of complex cmakes with so many options you should use ccmake (sudo apt-get install cmake-curses-gui) to be able to change easily the variables or at least view the values, or a real GUI one.

For other people with windows and visual studio 7, I also had to change the CUDA_HOST_COMPILER variable, else you get a bunch of errors with cmd.exe exit with code 1 or something similar... it seems it couldn't get there with the autodetected one.

This worked for me with OpenCV 3.3 and CUDA 9.0 and Visual Studio 2017 with Windows 10. I think it should work also in Ubuntu, since the error and the changes are related to CUDA. I haven't tested it much, I compiled and run the some of the performance tests and all of them passed... So I think everything worked ok.

