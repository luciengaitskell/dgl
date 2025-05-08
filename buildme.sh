#!/bin/bash
cd build
export CUDA_HOME=/usr/local/cuda-11.8  
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 -DUSE_NCCL=ON ..
make -j
cd ..
