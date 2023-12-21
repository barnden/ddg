import argparse
import lmdb
import os
from pathlib import Path
import torchvision.datasets as dset
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    size = 0
    base_dir = Path("/DATA/srn_cars/srn_cars/cars_train")
    models = os.listdir(base_dir.absolute())

    Path("/DATA/pxfcars-lmdb").mkdir(exist_ok=True)
    lmdb_path = Path("/DATA/pxfcars-lmdb/train.lmdb")
    env = lmdb.open(str(lmdb_path.absolute()), map_size=10485760 * 4096)

    subdirs = ["intrinsics", "pose", "rgb"]

    with env.begin(write=True) as txn:
        for model in models:
            subdir_paths = [base_dir.joinpath(model, subdirs) for subdirs in subdirs]

            pose_path = base_dir.joinpath(model, "pose")
            image_path = base_dir.joinpath(model, "rgb")

            filenames = sorted([fname.replace('.png', '') for fname in os.listdir(image_path)])

            # imgs = []
            # cams = []

            with open(base_dir.joinpath(model, 'intrinsics.txt'), "rb") as f:
                intrinsics = f.read()

            focal, cx, cy, _, _, _, _, _, height, width = map(float, intrinsics.decode().split())

            # Normalise intrinsics
            intrinsics = np.array([focal / width, 0, cx / width, 0, focal / height, cy / height, 0, 0, 1], dtype=np.float32)

            # print(np.fromstring(intrinsics, sep=' ', dtype=np.float32).reshape(-1, 3, 3))
            # exit()

            for filename in filenames:
                with open(pose_path.joinpath(f"{filename}.txt"), "rb") as f:
                    extrinsics = f.read()

                with open(image_path.joinpath(f"{filename}.png"), "rb") as f:
                    image_data = f.read()

                with Image.open(image_path.joinpath(f"{filename}.png")) as image:
                    image_data = np.array(image, dtype=np.uint8)[..., : 3].ravel()
                    txn.put(str(f"i{size}").encode(), image_data)
                    # imgs.append(image_data)

                # cams.append(np.fromstring(extrinsics + intrinsics, sep=' ', dtype=np.float32))

                extrinsics = np.fromstring(extrinsics, sep=' ', dtype=np.float32)
                camera_pose = np.float32(np.concatenate([extrinsics, intrinsics]))

                txn.put(str(f"p{size}").encode(), camera_pose)

                break

            # txn.put(str(f'i{size}').encode(), np.stack(imgs, axis=0))
            # txn.put(str(f'p{size}').encode(), np.stack(cams, axis=0))

            size += 1

    print(size)