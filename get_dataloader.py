from gpt_k import PileDataset
from torch.utils.data import DataLoader

def get_dataloader(batch_size):
    dataset = PileDataset()
    dataloader = DataLoader(dataset, batch_size, num_workers=0, shuffle=True)
    return dataloader