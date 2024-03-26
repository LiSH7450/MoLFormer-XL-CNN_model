#!/bin/bash
#dir增强5倍

/home/anaconda3/envs/pytorch2/bin/python "drug_induced_model_modify backup.py" \
        --device cuda \
        --batch_size 32  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.2 \
        --dropout 0.2 \
        --lr_start 3e-5 \
        --num_workers 14 \
        --max_epochs 100 \
        --num_feats 32 \
        --weight_decay 1e-6 \
        --seed_path '../data/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name 1 \
        --data_root ../data/DIR/1 \
        --checkpoints_folder './result_dir_final/checkpoints_dir1' \
        --dims 768 768 768 1  \
        --num_classes 2 \
        --measure_name label \
        
/home/anaconda3/envs/pytorch2/bin/python "drug_induced_model_modify backup.py" \
        --device cuda \
        --batch_size 32  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.2 \
        --dropout 0.2 \
        --lr_start 3e-5 \
        --num_workers 14 \
        --max_epochs 100 \
        --num_feats 32 \
        --weight_decay 1e-6 \
        --seed_path '../data/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name 2 \
        --data_root ../data/DIR/2 \
        --checkpoints_folder './result_dir_final/checkpoints_dir2' \
        --dims 768 768 768 1  \
        --num_classes 2 \
        --measure_name label \
        
/home/anaconda3/envs/pytorch2/bin/python "drug_induced_model_modify backup.py" \
        --device cuda \
        --batch_size 32  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.2 \
        --dropout 0.2 \
        --lr_start 3e-5 \
        --num_workers 14 \
        --max_epochs 100 \
        --num_feats 32 \
        --weight_decay 1e-6 \
        --seed_path '../data/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name 3 \
        --data_root ../data/DIR/3 \
        --checkpoints_folder './result_dir_final/checkpoints_dir3' \
        --dims 768 768 768 1  \
        --num_classes 2 \
        --measure_name label \
        
/home/anaconda3/envs/pytorch2/bin/python "drug_induced_model_modify backup.py" \
        --device cuda \
        --batch_size 32  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.2 \
        --dropout 0.2 \
        --lr_start 3e-5 \
        --num_workers 14 \
        --max_epochs 100 \
        --num_feats 32 \
        --weight_decay 1e-6 \
        --seed_path '../data/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name 4 \
        --data_root ../data/DIR/4 \
        --checkpoints_folder './result_dir_final/checkpoints_dir4' \
        --dims 768 768 768 1  \
        --num_classes 2 \
        --measure_name label \
        
        
/home/anaconda3/envs/pytorch2/bin/python "drug_induced_model_modify backup.py" \
        --device cuda \
        --batch_size 32  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.2 \
        --dropout 0.2 \
        --lr_start 3e-5 \
        --num_workers 14 \
        --max_epochs 100 \
        --num_feats 32 \
        --weight_decay 1e-6 \
        --seed_path '../data/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name 5 \
        --data_root ../data/DIR/5 \
        --checkpoints_folder './result_dir_final/checkpoints_dir5' \
        --dims 768 768 768 1  \
        --num_classes 2 \
        --measure_name label \
