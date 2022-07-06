from gpt_k import PileDataset
from torch.utils.data import DataLoader
from gpt_k.utils.GPT2_Dataset import GPT2Dataset
from gpt_k.utils.indexed_dataset import make_dataset as make_indexed_dataset
import numpy as np

def get_dataloader(batch_size):
    indexed_dataset = make_indexed_dataset("./data/pile_test_text_document", "mmap", skip_warmup=False)
    total_num_of_documents = indexed_dataset.sizes.shape[0]
    print(f"{total_num_of_documents} total documents")
    dataset = None
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    dataset = GPT2Dataset("train_test", "pile_test_text_document", documents, indexed_dataset, 7500, 100, 1)
    dataloader = DataLoader(dataset, batch_size, num_workers=0, shuffle=True)
    return dataloader