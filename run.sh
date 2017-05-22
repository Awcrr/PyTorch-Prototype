#!/usr/bin/env sh
python main.py \
    -nGPU 4 \
    -data_dir data \
    -dataset ImageNet \
    -save_path checkpoints \
    -model VGG19 \
    -batch_size 96 \
    -training_record record.npy \
    -num_classes 1000 \
    -n_epochs 150 
