#!/bin/bash
LD_PRELOAD=/usr/lib64/libnvidia-ml.so DGL_DS_USE_NCCL=1 uv run test_sampling.py --n_ranks 2
