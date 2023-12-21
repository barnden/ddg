#!/bin/bash

python train_ddgan.py --dataset cars --image_size 64 --exp relative_random --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 2 --num_epoch 24000 --ngf 64 --embedding_type positional --r1_gamma 2. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10  --num_process_per_node 1 --save_content --save_content_every 10 --save_ckpt_every 5 --embed_rays --relative_render --random_render
