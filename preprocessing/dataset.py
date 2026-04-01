import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

MAX_SEQ_LEN = 50


class MoseiDataset(Dataset):
    def __init__(self, data, split='train'):
        self.samples = data[split]
        self.ids = list(self.samples.keys())

        counts = np.zeros(3)
        for s in self.samples.values():
            counts[s['label']] += 1
        weights = counts.sum() / (3 * counts)
        self.class_weights = torch.FloatTensor(weights / weights.mean())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = self.samples[self.ids[idx]]

        audio,  a_mask = self._pad(sample['audio'],  MAX_SEQ_LEN)
        visual, _      = self._pad(sample['visual'], MAX_SEQ_LEN)

        if sample['text'] is not None:
            text, _ = self._pad(sample['text'], MAX_SEQ_LEN)
        else:
            text = np.zeros((MAX_SEQ_LEN, 768), dtype=np.float32)

        return (torch.FloatTensor(audio),
                torch.FloatTensor(visual),
                torch.FloatTensor(text),
                torch.LongTensor([sample['label']])[0],
                torch.FloatTensor(a_mask))

    def _pad(self, arr, max_len):
        seq_len, feat_dim = arr.shape
        mask = np.zeros(max_len, dtype=np.float32)
        if seq_len >= max_len:
            mask[:max_len] = 1.0
            return arr[:max_len], mask
        pad = np.zeros((max_len - seq_len, feat_dim), dtype=np.float32)
        mask[:seq_len] = 1.0
        return np.concatenate([arr, pad], axis=0), mask


def get_dataloaders(pkl_path, batch_size=64, num_workers=0):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    train_ds = MoseiDataset(data, 'train')
    val_ds   = MoseiDataset(data, 'val')
    test_ds  = MoseiDataset(data, 'test')

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.class_weights
