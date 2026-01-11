#!/bin/bash

MASTER=$1

MODEL_FLAGS="\
--model_type jamba \
--channels 512 \
--depth 6 \
--heads 4 \
--headdim 16 \
--num_layers 8 \
--attn_dropout 0.25 \
--in_node_nf 512 \
--out_node_nf 512 \
--in_edge_nf 512 \
--hidden_nf 512 \
--pe_dim 20 \
--context_dim 16 \
--time_embed_dim 16 \
--d_state 512 \
--d_conv 256 \
--order_by_degree True \
--shuffle_ind 0 \
--num_tokens 2000 \
--num_graphs 55000"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --beta_start 1e-5 --beta_end 2e-2 --loss_type "ph" --c 0.000069"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"


export DDDM_LOGDIR=./checkpoints

torchrun \
--standalone \
--nproc_per_node=2 \
./scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


