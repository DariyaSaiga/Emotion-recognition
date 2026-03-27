"""
dataset.py — загрузка CMU-MOSEI из cmu_mosei_final.csv

Формат CSV:
    video_id | text | label | audio_feat_0..73 (74 cols) | visual_feat_0..34 (35 cols)

Классы:
    0 = happy
    1 = sad
    2 = anger

Автоматически делает stratified split: train/val/test = 70/15/15
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════

AUDIO_DIM   = 74    # COVAREP признаки
VISUAL_DIM  = 35    # Facet42 признаки
NUM_CLASSES = 3     # happy, sad, anger

EMOTION_NAMES = {0: 'happy', 1: 'sad', 2: 'anger'}

# Веса классов для CrossEntropyLoss (из preprocessing)
CLASS_WEIGHTS = torch.tensor([0.441, 2.149, 3.713], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class CMUMOSEIDataset(Dataset):
    """
    Датасет CMU-MOSEI с pre-extracted признаками.

    Возвращает батч:
        text_inputs  : dict с input_ids + attention_mask (от BERT tokenizer)
        audio_feat   : FloatTensor (74,)
        visual_feat  : FloatTensor (35,)
        label        : LongTensor scalar
    """

    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer,
                 max_text_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Находим колонки признаков
        self.audio_cols  = _find_feat_cols(df, prefix='audio')
        self.visual_cols = _find_feat_cols(df, prefix='visual')

        assert len(self.audio_cols)  == AUDIO_DIM, \
            f"Ожидалось {AUDIO_DIM} аудио колонок, нашлось {len(self.audio_cols)}"
        assert len(self.visual_cols) == VISUAL_DIM, \
            f"Ожидалось {VISUAL_DIM} видео колонок, нашлось {len(self.visual_cols)}"

        print(f"  Датасет: {len(self.df)} сэмплов | "
              f"аудио: {len(self.audio_cols)}-dim | "
              f"видео: {len(self.visual_cols)}-dim")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text  = str(row['text'])
        label = int(row['label'])

        # Токенизируем текст
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        text_inputs = {
            'input_ids':      encoded['input_ids'].squeeze(0),       # (max_text_len,)
            'attention_mask': encoded['attention_mask'].squeeze(0),   # (max_text_len,)
        }

        # Признаки
        audio_feat  = torch.FloatTensor(
            row[self.audio_cols].values.astype(np.float32)
        )   # (74,)
        visual_feat = torch.FloatTensor(
            row[self.visual_cols].values.astype(np.float32)
        )   # (35,)

        return {
            'text_inputs': text_inputs,
            'audio_feat':  audio_feat,
            'visual_feat': visual_feat,
            'label':       torch.tensor(label, dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _find_feat_cols(df: pd.DataFrame, prefix: str) -> list:
    """
    Ищет колонки признаков по префиксу.
    Поддерживает форматы:
        audio_feat_0, audio_feat_1, ...
        audio_0, audio_1, ...
        audio_feat_mean_0, ...
    """
    cols = [c for c in df.columns if c.startswith(prefix)]
    # Сортируем по номеру в конце (чтобы 0,1,...,73 — не лексикографически)
    try:
        cols = sorted(cols, key=lambda c: int(c.split('_')[-1]))
    except ValueError:
        cols = sorted(cols)
    return cols


def make_splits(csv_path: str,
                val_size: float = 0.15,
                test_size: float = 0.15,
                random_state: int = 42) -> tuple:
    """
    Читает CSV и делает стратифицированный split на train/val/test.

    Returns:
        df_train, df_val, df_test
    """
    df = pd.read_csv(csv_path)

    # Заменяем NaN в тексте
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)

    labels = df['label'].values

    # train vs (val + test)
    relative_test = test_size / (1.0 - val_size)
    df_train, df_temp = train_test_split(
        df, test_size=(val_size + test_size),
        stratify=labels, random_state=random_state
    )
    # val vs test
    df_val, df_test = train_test_split(
        df_temp, test_size=relative_test,
        stratify=df_temp['label'].values, random_state=random_state
    )

    print(f"\nРазбивка датасета:")
    for name, part in [('train', df_train), ('val', df_val), ('test', df_test)]:
        counts = part['label'].value_counts().sort_index()
        dist   = ', '.join(f"{EMOTION_NAMES[k]}={v}" for k, v in counts.items())
        print(f"  {name:5s}: {len(part):4d} сэмплов  [{dist}]")

    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════
# ФАБРИКА ЗАГРУЗЧИКОВ
# ══════════════════════════════════════════════════════════════════

def get_dataloaders(csv_path: str,
                   batch_size: int = 32,
                   num_workers: int = 0,
                   max_text_len: int = 128,
                   bert_name: str = 'bert-base-uncased',
                   val_size: float = 0.15,
                   test_size: float = 0.15,
                   random_state: int = 42):
    """
    Создаёт DataLoader-ы для train / val / test.

    Args:
        csv_path     : путь к cmu_mosei_final.csv
        batch_size   : размер батча
        num_workers  : для DataLoader (0 = без параллелизма, хорошо для Colab)
        max_text_len : максимальная длина текста для BERT
        bert_name    : имя BERT модели
        val_size     : доля валидации (0.15)
        test_size    : доля теста (0.15)
        random_state : для воспроизводимости

    Returns:
        train_loader, val_loader, test_loader
    """
    print("Загружаем токенизатор BERT...")
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    df_train, df_val, df_test = make_splits(
        csv_path, val_size=val_size, test_size=test_size,
        random_state=random_state
    )

    def make_loader(df, shuffle):
        ds = CMUMOSEIDataset(df, tokenizer, max_text_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_loader(df_train, shuffle=True)
    val_loader   = make_loader(df_val,   shuffle=False)
    test_loader  = make_loader(df_test,  shuffle=False)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════
# БЫСТРАЯ ПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else 'cmu_mosei_final.csv'
    print(f"\nПроверяем dataset.py на файле: {csv}\n")

    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=csv, batch_size=8, num_workers=0
    )

    batch = next(iter(train_loader))
    print("\nПример батча:")
    print(f"  input_ids     : {batch['text_inputs']['input_ids'].shape}")
    print(f"  attention_mask: {batch['text_inputs']['attention_mask'].shape}")
    print(f"  audio_feat    : {batch['audio_feat'].shape}")
    print(f"  visual_feat   : {batch['visual_feat'].shape}")
    print(f"  label         : {batch['label']}")
    print(f"\n✓ dataset.py работает корректно!")
