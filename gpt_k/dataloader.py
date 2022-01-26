import os
import torch
from torch.utils.data import Dataset


class PileDataset(Dataset):

    vocab_size = 50257

    def __init__(
        self,
        file_dir="../data/pile/tokenized",
    ):
        self.one = torch.load(file_dir + "/val_2_16384.pt")

    def __len__(self):
        return len(self.one)

    def __getitem__(self, idx):
        return torch.abs(self.one[idx].type(torch.LongTensor))