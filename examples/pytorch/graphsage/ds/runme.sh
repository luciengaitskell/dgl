#!/bin/bash
LD_PRELOAD=/usr/lib64/libnvidia-ml.so DGL_DS_USE_NCCL=1 DGL_DS_USE_PERSISTENT_SAMPLER=1 uv run test_sampling.py --n_ranks 2 --part_config $HOME/projects/artifacts/dgl/examples/pytorch/graphsage/ds/data-2/reddit.json
