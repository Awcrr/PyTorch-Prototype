#!/usr/bin/env sh
python main.py \
    -nGPU 4 \
    -data_dir data \
    -dataset MNIST \
    -save_path checkpoints \
    -n_epochs 18 
