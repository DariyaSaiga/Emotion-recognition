import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from PIL import Image


class CMUMOSEIDataset(Dataset):
    """
    Датасет CMU-MOSEI.

    Структура:
        CMU-MOSEI_CLEAN_5k/
            Audio_chunk/
                train/   <- .wav файлы
                val/
                test/
            Labels/
                train.csv
                val.csv
                test.csv
    """

    EMOTION_COLS = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']

    def __init__(self, csv_path, audio_dir):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sample_rate = 16000
        self.max_audio_len = 4  # секунды

        # Убираем строки с пустым текстом
        self.df = self.df.dropna(subset=['text']).reset_index(drop=True)
        self.df['text'] = self.df['text'].astype(str).str.strip()
        self.df = self.df[self.df['text'] != ''].reset_index(drop=True)

        print(f"  Загружено {len(self.df)} samples из {os.path.basename(csv_path)}")

    def __len__(self):
        return len(self.df)

    def _get_label(self, row):
        """
        Метка = индекс максимальной эмоции.
        Если все нули → 6 (neutral).
        0=happy 1=sad 2=anger 3=surprise 4=disgust 5=fear 6=neutral
        """
        scores = row[self.EMOTION_COLS].values.astype(float)
        if scores.max() == 0:
            return 6
        return int(np.argmax(scores))

    def _load_mel(self, audio_path):
        """wav → мел-спектрограмма (1, 64, 128)"""
        target_len = self.max_audio_len * self.sample_rate

        try:
            wav, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception:
            wav = np.zeros(target_len, dtype=np.float32)

        # Обрезаем или паддим
        if len(wav) >= target_len:
            wav = wav[:target_len]
        else:
            wav = np.pad(wav, (0, target_len - len(wav)))

        # Мел-спектрограмма
        mel = librosa.feature.melspectrogram(
            y=wav, sr=self.sample_rate,
            n_mels=64, n_fft=512, hop_length=256
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Нормализуем в [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        return torch.FloatTensor(mel_db).unsqueeze(0)  # (1, 64, 128)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_id   = str(row['video'])
        start_time = float(row['start_time'])
        text       = str(row['text'])
        label      = self._get_label(row)

        # Имя wav файла
        end_time = float(row['end_time'])
        audio_filename = f"{video_id}_{start_time}_{end_time:.4f}.wav"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        mel = self._load_mel(audio_path)

        return {
            'mel':   mel,
            'text':  text,
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders(data_root, batch_size=16, num_workers=0, collate_fn=None):
    """
    Возвращает train / val / test DataLoader.
    """
    splits = {
        'train': (
            os.path.join(data_root, 'Labels', 'train.csv'),
            os.path.join(data_root, 'Audio_chunk', 'train'),
        ),
        'val': (
            os.path.join(data_root, 'Labels', 'val.csv'),
            os.path.join(data_root, 'Audio_chunk', 'val'),
        ),
        'test': (
            os.path.join(data_root, 'Labels', 'test.csv'),
            os.path.join(data_root, 'Audio_chunk', 'test'),
        ),
    }

    loaders = {}
    for split, (csv_path, audio_dir) in splits.items():
        ds = CMUMOSEIDataset(csv_path, audio_dir)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return loaders['train'], loaders['val'], loaders['test']
