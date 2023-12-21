import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image
import torch

class ShapenetDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            image_data = txn.get(str(f"i{index}").encode())
            pose_data = txn.get(str(f"p{index}").encode())
            index = 0

            if self.is_encoded:
                img = Image.open(io.BytesIO(image_data))
                img = img.convert('RGB')
            else:
                img = np.asarray(image_data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        pose = np.frombuffer(pose_data, dtype=np.float32).copy()

        return pose, img, target

    def __len__(self):
        # return 537642
        return 2458

