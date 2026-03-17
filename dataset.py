import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from PIL import Image
import torchvision.transforms as transforms


class CMUMOSEIDataset(Dataset):
    """
    Датасет CMU-MOSEI.

    Структура папок:
        CMU-MOSEI_CLEAN_5k/
            Audio_chunk/
                train/   <- .wav файлы для train
                val/     <- .wav файлы для val
                test/    <- .wav файлы для test
            Labels/
                train.csv
                val.csv
                test.csv

    Имя .wav файла: {video_id}_{start_time}.wav
    Пример: 268836_33.8555.wav
    """

    EMOTION_COLS = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    NUM_CLASSES = 7

    def __init__(self, csv_path, audio_dir, max_audio_len=4):
        """
        csv_path      : путь к train.csv / val.csv / test.csv
        audio_dir     : путь к папке Audio_chunk/train (или val/test)
        max_audio_len : длина аудио в секундах
        """
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.max_audio_len = max_audio_len
        self.sample_rate = 16000

        # Убираем строки с пустым текстом
        self.df = self.df.dropna(subset=['text']).reset_index(drop=True)
        self.df['text'] = self.df['text'].astype(str).str.strip()
        self.df = self.df[self.df['text'] != ''].reset_index(drop=True)

        print(f"  Загружено {len(self.df)} samples из {csv_path}")

    def __len__(self):
        return len(self.df)

    def _get_emotion_label(self, row):
        """
        Метка эмоции = индекс максимальной из 6 эмоций.
        Если все нули → класс 6 (neutral).
        Классы: 0=happy, 1=sad, 2=anger, 3=surprise, 4=disgust, 5=fear, 6=neutral
        """
        scores = row[self.EMOTION_COLS].values.astype(float)
        if scores.max() == 0:
            return 6  # neutral
        return int(np.argmax(scores))

    def _load_mel(self, audio_path):
        """
        Загружает .wav → мел-спектрограмма (1, 128, 128).
        Возвращает torch.FloatTensor.
        """
        target_len = self.max_audio_len * self.sample_rate

        try:
            wav, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception:
            # Если файл не найден — нули
            wav = np.zeros(target_len, dtype=np.float32)

        # Обрезаем или паддим
        if len(wav) >= target_len:
            wav = wav[:target_len]
        else:
            wav = np.pad(wav, (0, target_len - len(wav)))

        # Мел-спектрограмма (128 mel bins)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=self.sample_rate,
            n_mels=128, n_fft=512, hop_length=256
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Нормализуем
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Resize до 128×128 для удобства CNN
        mel_resized = np.array(
            Image.fromarray(mel_db).resize((128, 128), Image.BILINEAR)
        )

        return torch.FloatTensor(mel_resized).unsqueeze(0)  # (1, 128, 128)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_id  = str(row['video'])
        start_time = float(row['start_time'])
        text       = str(row['text'])
        label      = self._get_emotion_label(row)

        # Имя .wav файла: video_id_start_time.wav
        audio_filename = f"{video_id}_{start_time}.wav"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        mel = self._load_mel(audio_path)

        return {
            'mel':   mel,                            # (1, 128, 128) для CNN
            'text':  text,                           # строка для BERT
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders(data_root, batch_size=32, num_workers=2):
    """
    Создаёт train/val/test DataLoader.

    data_root: папка CMU-MOSEI_CLEAN_5k/
    """
    splits = {
        'train': (
            os.path.join(data_root, 'Labels', 'train.csv'),
            os.path.join(data_root, 'Audio_chunk', 'train')
        ),
        'val': (
            os.path.join(data_root, 'Labels', 'val.csv'),
            os.path.join(data_root, 'Audio_chunk', 'val')
        ),
        'test': (
            os.path.join(data_root, 'Labels', 'test.csv'),
            os.path.join(data_root, 'Audio_chunk', 'test')
        ),
    }

    loaders = {}
    for split, (csv_path, audio_dir) in splits.items():
        ds = CMUMOSEIDataset(csv_path, audio_dir)
        shuffle = (split == 'train')
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            pin_memory=True
        )

    return loaders['train'], loaders['val'], loaders['test']
