# Dataset
ShapeNet Cars:
https://drive.google.com/drive/folders/1n0cDiiuA8Fj6I_ZE9HBAVM7ltxqazGzR?usp=drive_link

In `train_ddgan.py` change the `root` kwarg for `ShapenetDataset` to point to `pxfcars_lmdb`.

For other datasets, look at `scripts/create_shapenet_lmdb.py` or [NVAE](https://github.com/NVlabs/NVAE/blob/master/scripts/create_celeba64_lmdb.py) on how to create an lmdb dataset to train on.

# Training

- Embed input pose to diffusion, renderer, and discriminator:
```
#!/bin/bash

python train_ddgan.py --dataset cars --image_size 64 --exp triplane_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 24 --num_epoch 1000 --ngf 64 --embedding_type positional --r1_gamma 2. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10  --num_process_per_node 1 --save_content --save_content_every 10 --save_ckpt_every 10 --embed_rays
```

# Sampling

- Create random samples facing left where pose is embedded into diffusion, renderer:
```
#!/bin/bash

python sample_triplane.py --dataset cars --image_size 64 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional --z_emb_dim 256 \
--exp triplane_exp1 --random --render_from_side left --embed_rays --epoch_id 740
```

- Create a render a random sample from a circular camera path:
```
#!/bin/bash

python sample_triplane.py --dataset cars --image_size 64 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional --z_emb_dim 256 \
--exp triplane_exp1 --circular --embed_rays --epoch_id 740
```
