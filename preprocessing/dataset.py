import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


# -------------------------------------------------------
# Максимальная длина последовательности
# Почему 50: стандарт для CMU-MOSEI word-aligned версии.
# Сегменты длиннее 50 — обрезаем (их мало).
# Сегменты короче 50 — дополняем нулями (padding).
# -------------------------------------------------------
MAX_SEQ_LEN = 50


class MoseiDataset(Dataset):
    """
    PyTorch Dataset для CMU-MOSEI.

    Параметры:
        data  — словарь загруженный из mosei_clean.pkl
        split — 'train', 'val' или 'test'

    Возвращает для каждого сэмпла:
        audio   — тензор [MAX_SEQ_LEN, 74]
        visual  — тензор [MAX_SEQ_LEN, 713]
        text    — тензор [MAX_SEQ_LEN, text_dim] или None
        label   — тензор [] (скаляр: 0=happy, 1=sad, 2=anger)
        mask    — тензор [MAX_SEQ_LEN] — 1 где реальные данные, 0 где padding
    """

    def __init__(self, data, split='train'):
        """
        data  — полный словарь из pkl (содержит train/val/test)
        split — какой сплит загружать
        """
        self.split_data = data[split]
        self.ids = list(self.split_data.keys())

        # Считаем веса классов для weighted loss
        # Это поможет модели не игнорировать редкие классы
        self.class_weights = self._compute_class_weights()

        print(f"[MoseiDataset] split='{split}', сэмплов={len(self.ids)}")

    def __len__(self):
        # Сколько всего сэмплов в этом сплите
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Возвращает один сэмпл по индексу.

        Здесь происходит три важных действия:
        1. Достаём numpy массивы из словаря
        2. Делаем padding/truncation до MAX_SEQ_LEN
        3. Конвертируем в torch тензоры
        """
        seg_id = self.ids[idx]
        sample = self.split_data[seg_id]

        audio  = sample['audio']   # numpy [seq_len, 74]
        visual = sample['visual']  # numpy [seq_len, 713]
        label  = sample['label']   # int: 0, 1 или 2

        # --- Padding / Truncation ---
        # Почему это важно: 1D-CNN и BiLSTM принимают батчи фиксированного размера.
        # Без padding каждый сэмпл будет разной длины — батч не соберётся.
        audio,  audio_mask  = self._pad_or_truncate(audio,  MAX_SEQ_LEN)
        visual, visual_mask = self._pad_or_truncate(visual, MAX_SEQ_LEN)

        # Текст — если есть в pkl
        if sample['text'] is not None:
            text, _ = self._pad_or_truncate(sample['text'], MAX_SEQ_LEN)
            text = torch.FloatTensor(text)
        else:
            # Если текст не был в HDF5 — ставим нули
            # BERT будет считать эмбеддинги отдельно в модели по raw тексту
            text = torch.zeros(MAX_SEQ_LEN, 768)

        # --- Конвертируем в тензоры ---
        audio  = torch.FloatTensor(audio)       # [MAX_SEQ_LEN, 74]
        visual = torch.FloatTensor(visual)      # [MAX_SEQ_LEN, 713]
        mask   = torch.FloatTensor(audio_mask)  # [MAX_SEQ_LEN] — где реальные данные
        label  = torch.LongTensor([label])[0]   # скаляр

        return audio, visual, text, label, mask

    # ============================================================
    # Вспомогательные методы
    # ============================================================

    def _pad_or_truncate(self, arr, max_len):
        """
        Приводим последовательность к фиксированной длине.

        Если arr длиннее max_len  → берём первые max_len шагов
        Если arr короче max_len   → дополняем нулями снизу

        Также создаём маску:
            1 = реальный кадр
            0 = padding (нули, которые мы добавили)

        Зачем маска: BiLSTM и Attention механизм могут использовать
        маску чтобы игнорировать padding при вычислении.
        Без маски модель будет "смотреть" на нули как на реальный сигнал.
        """
        seq_len, feat_dim = arr.shape
        mask = np.zeros(max_len, dtype=np.float32)

        if seq_len >= max_len:
            # Обрезаем — берём первые max_len шагов
            result = arr[:max_len, :]
            mask[:max_len] = 1.0

        else:
            # Padding — добавляем нули до max_len
            pad_len = max_len - seq_len
            padding = np.zeros((pad_len, feat_dim), dtype=np.float32)
            result  = np.concatenate([arr, padding], axis=0)  # [max_len, feat_dim]
            mask[:seq_len] = 1.0  # только реальные кадры = 1

        return result, mask

    def _compute_class_weights(self):
        """
        Считаем веса для каждого класса.

        Зачем: даже после балансировки классы могут быть неравномерны.
        Weighted CrossEntropy loss будет больше штрафовать за ошибки
        на редких классах (sad, anger) — это помогает модели не игнорировать их.

        Формула: weight[i] = total / (n_classes * count[i])
        Редкий класс → маленький count → большой вес → больше внимания
        """
        counts = np.zeros(3)
        for sample in self.split_data.values():
            counts[sample['label']] += 1

        total = counts.sum()
        weights = total / (3 * counts)

        # Нормализуем чтобы среднее = 1.0
        weights = weights / weights.mean()

        print(f"  Веса классов: happy={weights[0]:.3f}, "
              f"sad={weights[1]:.3f}, anger={weights[2]:.3f}")

        return torch.FloatTensor(weights)

    def get_class_weights(self):
        """Возвращает веса классов для передачи в loss функцию."""
        return self.class_weights


# ============================================================
# Функция для создания DataLoader'ов
# ============================================================

def get_dataloaders(pkl_path, batch_size=64, num_workers=0):
    """
    Загружает pkl и создаёт три DataLoader'а: train, val, test.

    Параметры:
        pkl_path    — путь к mosei_clean.pkl
        batch_size  — размер батча (64 или 128 для T4 GPU)
        num_workers — 0 для Colab (избегаем предупреждений)

    Почему shuffle=True только для train:
        На train перемешиваем чтобы модель не запоминала порядок.
        На val/test НЕ перемешиваем — нужна стабильная оценка.
    """
    print(f"\nЗагружаем данные из {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    train_dataset = MoseiDataset(data, split='train')
    val_dataset   = MoseiDataset(data, split='val')
    test_dataset  = MoseiDataset(data, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # перемешиваем каждую эпоху
        num_workers=num_workers,
        drop_last=True,        # последний неполный батч отбрасываем
                               # это стабилизирует BatchNorm в CNN
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Возвращаем также веса классов для loss функции
    class_weights = train_dataset.get_class_weights()

    return train_loader, val_loader, test_loader, class_weights


# ============================================================
# Проверка — запусти этот файл напрямую чтобы убедиться
# что всё работает правильно
# ============================================================

if __name__ == "__main__":

    PKL_PATH = "/Users/dariyaablanova/Desktop/unic_work/Diploma/mosei_clean.pkl"

    print("=" * 50)
    print("Проверяем dataset.py...")
    print("=" * 50)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        PKL_PATH,
        batch_size=64,
        num_workers=0,
    )

    print(f"\nРазмер train loader: {len(train_loader)} батчей")
    print(f"Размер val   loader: {len(val_loader)} батчей")
    print(f"Размер test  loader: {len(test_loader)} батчей")
    print(f"Веса классов: {class_weights}")

    # Берём один батч и проверяем формы тензоров
    print("\nПроверяем формы одного батча...")
    audio, visual, text, labels, mask = next(iter(train_loader))

    print(f"  audio:  {audio.shape}   ← [batch, MAX_SEQ_LEN, 74]")
    print(f"  visual: {visual.shape}  ← [batch, MAX_SEQ_LEN, 713]")
    print(f"  text:   {text.shape}    ← [batch, MAX_SEQ_LEN, text_dim]")
    print(f"  labels: {labels.shape}  ← [batch]")
    print(f"  mask:   {mask.shape}    ← [batch, MAX_SEQ_LEN]")

    print(f"\n  Уникальные лейблы в батче: {labels.unique()}")
    print(f"  Диапазон аудио:  [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"  Диапазон визуал: [{visual.min():.3f}, {visual.max():.3f}]")
