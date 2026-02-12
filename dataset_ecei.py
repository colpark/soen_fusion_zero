from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import h5py

from torch.utils.data import Dataset, DataLoader


class ECEiDataset(Dataset):
    def __init__(self,
                 root,
                 split,
                 context,
                 horizon):

        self.root = Path(root)

        assert self.root.exists()

        self.meta = pd.read_csv(self.root/'meta.csv')
        self.meta = self.meta[ self.meta['split'] == split ]

        self.context = context
        self.total_len = context + horizon

    def __len__(self, ):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]

        shot = int(row.shot)
        fname = self.root/f'{shot}.h5'

        # get a random slice of the data for the context and horizon
        length = row.length

        start_idx = np.random.randint(0, length - self.total_len)
        indices = np.arange(start_idx, start_idx + self.total_len)

        with h5py.File(fname, 'r') as h5_file:
            subseq = h5_file['LFS'][..., indices]

        subseq -= row.offset

        context = subseq[..., : self.context]
        horizon = subseq[..., self.context :]

        return context, horizon


def test(split):
    dataset = ECEiDataset(
        root    = '/global/cfs/cdirs/m5187/ECEi_excerpt/dsrpt',
        split   = split,
        context = 200_000,
        horizon = 20_000
    )

    dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

    pbar = tqdm(dataloader, total=len(dataloader))
    for context, horizon in tqdm(dataloader):
        context_shape = context.shape
        horizon_shape = horizon.shape
        pbar.set_postfix(context=context_shape, horizon=horizon_shape)


if __name__ == '__main__':
    test('train')
