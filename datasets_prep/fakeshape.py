import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image
import torch

class DebugDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        print("Using debug dataset (root={})".format(root))
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
            image_data = txn.get(str(index).encode())
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

        # Create identity camera extrinsics and use Shapenet camera intrinsics
        # c = torch.zeros(25)
        # c[:16] = torch.eye(4).ravel()
        # c[16:] = torch.Tensor([131.25, 0., 64., 0., 131.25, 64., 0., 0., 1.])

        c = torch.Tensor([ 0.8540,      0.3085,    -0.4188,   0.5445,   0.5202,
                            -0.5066,      0.6876,    -0.8939,   0.0000,  -0.8051,
                            -0.5931,      0.7711,    -0.0000,   0.0000,  -0.0000,
                            1.0000,      525.0000,   0.0000, 256.0000,   0.0000,
                            525.0000,    256.0000,   0.0000,   0.0000,   1.0000])

        return c, img, target

    def __len__(self):
        return 300

