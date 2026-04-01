import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

MAX_SEQ_LEN = 50


class MoseiDataset(Dataset):
    def __init__(self, data, split='train'):
        self.samples = data[split]
        self.ids     = list(self.samples.keys())

        counts = np.zeros(3)
        for s in self.samples.values():
            counts[s['label']] += 1

        # Мягкие веса через log — не такие агрессивные как counts.sum()/(3*counts)
        weights = np.log1p(counts.sum() / counts)
        self.class_weights = torch.FloatTensor(weights / weights.mean())

        print(f'[{split}] n={len(self.ids)} | '
              f'happy={int(counts[0])} sad={int(counts[1])} anger={int(counts[2])} | '
              f'w=[{self.class_weights[0]:.2f} {self.class_weights[1]:.2f} {self.class_weights[2]:.2f}]')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        s = self.samples[self.ids[idx]]

        audio,  mask = self._pad(s['audio'],  MAX_SEQ_LEN)  # [50, 74]
        visual, _    = self._pad(s['visual'], MAX_SEQ_LEN)  # [50, 713]

        # Текст теперь [768] — один вектор на предложение
        if s['text'] is not None:
            text = s['text'].copy()   # [768]
        else:
            text = np.zeros(768, dtype=np.float32)

        return (torch.FloatTensor(audio),
                torch.FloatTensor(visual),
                torch.FloatTensor(text),
                torch.LongTensor([s['label']])[0],
                torch.FloatTensor(mask))

    def _pad(self, arr, max_len):
        seq_len, feat_dim = arr.shape
        mask = np.zeros(max_len, dtype=np.float32)
        if seq_len >= max_len:
            mask[:] = 1.0
            return arr[:max_len].copy(), mask
        pad = np.zeros((max_len - seq_len, feat_dim), dtype=np.float32)
        mask[:seq_len] = 1.0
        return np.concatenate([arr, pad], axis=0), mask


def get_dataloaders(pkl_path, batch_size=32, num_workers=0):
    print(f'\nЗагружаем: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    train_ds = MoseiDataset(data, 'train')
    val_ds   = MoseiDataset(data, 'val')
    test_ds  = MoseiDataset(data, 'test')

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.class_weights
