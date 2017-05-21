#!/usr/bin/env sh
python main.py \
    -nGPU 4 \
    -data_dir data \
    -dataset ImageNet \
    -save_path checkpoints \
    -model VGG16 \
    -batch_size 256 \
    -training_record record.npy \
    -n_epochs 150 
