#!/bin/bash

python sample_triplane.py --dataset cars --image_size 64 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional --z_emb_dim 256 \
--exp triplane_exp1 --epoch_id 740 --circular --embed_rays
# --exp triplane_exp1 --epoch_id 740 --random --render_from_side left --embed_rays
# --exp triplane_exp1 --epoch_id 740 --load_from_file all --embed_rays
# --exp triplane_exp2 --epoch_id 720 --circular
# --exp triplane_exp4 --epoch_id 260 --circular
