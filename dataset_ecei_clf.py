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
                 chunk_length,
                 dsrpt_threshold,
                 clear_threshold = None,
                 baseline_length = 50_000,
                 prob_of_one     = None):

        """
        dsrpt_threshold: if the end time of a chunk >= disruption time - dsrpt_threshold,
            it is in class 1 (disruptive).
        clear_threshold: if the end time of a chunk < disruption time - clear_threshold,
            it is in class 0 (clear).

        If not provided, clear_threshold = dsrpt_threshold.

        The sequence should have time-to-disruption >= chunk_length +
                                                       dsrpt_threshold +
                                                       clear_threshold

        That is, a sequence should have more clear chunks than disruptive chunks.

        If prob_of_one is not provided, we randomly choose end time until a
        class 0 or class 1 chunk is obtained.

        If prob_of_one is provided, we first pick a label, and then pick up the
        end time of the chunk.
        """

        self.root = Path(root)

        assert self.root.exists()

        self.meta = pd.read_csv(self.root/'meta.csv')
        self.meta = self.meta[ self.meta['split'] == split ].reset_index(drop=True)

        self.chunk_length = chunk_length
        self.baseline_length = baseline_length

        # set up time to disruption threshold for clear and disruptive chunks
        if clear_threshold is None:
            clear_threshold = dsrpt_threshold

        min_length = chunk_length + dsrpt_threshold + clear_threshold
        self.meta = self.meta[ self.meta['t_disruption'] * 1000 > min_length ]

        self.dsrpt_threshold = dsrpt_threshold
        self.clear_threshold = clear_threshold

        assert self.clear_threshold >= self.dsrpt_threshold, \
            'clear threshold must be bigger than the disruptive threshold'

        self.prob_of_one = prob_of_one
        if prob_of_one is not None:
            self.probability = [1 - prob_of_one, prob_of_one]

    def __len__(self, ):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        shot = int(row.shot)

        # time to disruption is given in milliseconds
        t_dis = int(row.t_disruption * 1000)

        start_idx, end_idx, label = self.get_start_end_label(t_dis)

        print(f'{index}: shot={shot}, length = {t_dis}, label={label}')


        fname = self.root/f'{shot}.h5'

        with h5py.File(fname, 'r') as h5_file:
            data_0 = h5_file['LFS'][..., : self.baseline_length]
            baseline = np.mean(data_0, axis=-1, keepdims=True)

            chunk = h5_file['LFS'][..., start_idx: end_idx] - baseline

        return chunk, label

    def get_start_end_label(self, t_dis):

        if self.prob_of_one is None:

            while True:
                end_idx = np.random.randint(self.chunk_length, t_dis + 1)

                time_to_disruption = t_dis - end_idx

                if time_to_disruption <= self.dsrpt_threshold:
                    label = 1
                    break
                elif time_to_disruption > self.clear_threshold:
                    label = 0
                    break
        else:
            label = np.random.choice([0, 1], p=self.probability)
            if label == 0:
                end_idx = np.random.randint(
                    low  = t_dis + 1 - self.dsrpt_threshold,
                    high = t_dis + 1
                )
            else:
                end_idx = np.random.randint(
                    low  = self.chunk_length,
                    high = t_dis + 1 - self.clear_threshold
                )

        end_idx = end_idx + self.baseline_length
        start_idx = end_idx - self.chunk_length

        return start_idx, end_idx, label



def test(split):
    dataset = ECEiDataset(
        root            = './dsrpt',
        split           = split,
        chunk_length    = 100_000,
        baseline_length = 50_000,
        dsrpt_threshold = 300_000,
        clear_threshold = 600_000,
        prob_of_one     = .5,
    )

    print(len(dataset))
    # for i in range(len(dataset)):
    #     chunk, label = dataset[i]
    #     # print(f'\tchunk shape = {chunk.shape}')
    #     print(f'\tlabel = {label}')


    dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

    pbar = tqdm(dataloader, total=len(dataloader))
    for chunk, label in tqdm(dataloader):
        pbar.set_postfix(chunk=chunk.shape, label=label)


if __name__ == '__main__':
    test('train')
