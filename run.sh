#!/usr/bin/env sh
python main.py \
    -nGPU 4 \
    -data_dir data \
    -dataset CIFAR100 \
    -save_path checkpoints \
    -model VGG16 \
    -n_epochs 18 
