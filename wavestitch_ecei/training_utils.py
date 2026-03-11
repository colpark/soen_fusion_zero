"""
WaveStitch training utilities — minimal copy for ECEi.
MyDataset, fetchModel (SSSDS4Imputer), fetchDiffusionConfig.
No pandas/datasets; ECEi data is loaded in data/dataset.py.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from TSImputers.SSSDS4Imputer import SSSDS4Imputer


class MyDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def fetchModel(in_features, out_features, args):
    """Build SSSDS4Imputer. in_features=162 (160 signal + 2 cond), out_features=160."""
    return SSSDS4Imputer(
        in_features,
        args.res_channels,
        args.skip_channels,
        out_features,
        args.num_res_layers,
        args.diff_step_embed_in,
        args.diff_step_embed_mid,
        args.diff_step_embed_out,
        args.s4_lmax,
        args.s4_dstate,
        args.s4_dropout,
        args.s4_bidirectional,
        args.s4_layernorm,
    )


def fetchDiffusionConfig(args):
    betas = np.linspace(args.beta_0, args.beta_T, args.timesteps).reshape((-1, 1))
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas).reshape((-1, 1))
    return {
        "betas": torch.from_numpy(betas).float(),
        "alpha_bars": torch.from_numpy(alpha_bars).float(),
        "alphas": torch.from_numpy(alphas).float(),
        "T": args.timesteps,
    }
