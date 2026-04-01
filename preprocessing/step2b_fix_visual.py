
import h5py
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# -------------------------------------------------------
HDF5_PATH   = "/Users/dariyaablanova/Downloads/mosei.hdf5"
CLEAN_PATH  = "/Users/dariyaablanova/Desktop/unic_work/Diploma/mosei_clean.pkl"  # перезапишем
# -------------------------------------------------------

CLASS_MAP   = {'happiness': 0, 'sadness': 1, 'anger': 2}
CLASS_NAMES = {0: 'happy', 1: 'sad', 2: 'anger'}

# Разумный порог для сырых фич OpenFace перед нормализацией
# Всё что выходит за [-1000, 1000] — это явно битые кадры
VISUAL_RAW_CLIP = 1000.0


def load_existing(path):
    """Загружаем уже готовый pkl чтобы не читать HDF5 заново."""
    print("Загружаем существующий mosei_clean.pkl...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def recompute_normalization(data):
    """
    Пересчитываем нормализацию с обрезкой сырых выбросов.
    """
    print("\nПересчитываем нормализацию визуала...")

    # Шаг 1 — собираем сырые визуальные фичи из train
    # Нам нужно вернуться к сырым данным через HDF5
    # (в pkl уже нормализованные — перечитаем из HDF5)

    print("Читаем сырые визуальные фичи из HDF5...")
    train_ids = set(data['train'].keys())
    val_ids   = set(data['val'].keys())
    test_ids  = set(data['test'].keys())
    all_ids   = train_ids | val_ids | test_ids

    raw_visual = {}
    raw_audio  = {}

    with h5py.File(HDF5_PATH, 'r') as f:
        for seg_id in all_ids:
            try:
                audio  = f['COVAREP'][seg_id]['features'][()].astype(np.float32)
                visual = f['OpenFace_2'][seg_id]['features'][()].astype(np.float32)

                audio  = np.nan_to_num(audio,  nan=0.0, posinf=0.0, neginf=0.0)
                visual = np.nan_to_num(visual, nan=0.0, posinf=0.0, neginf=0.0)

                # Обрезаем сырой визуал ДО скейлера
                visual = np.clip(visual, -VISUAL_RAW_CLIP, VISUAL_RAW_CLIP)

                raw_visual[seg_id] = visual
                raw_audio[seg_id]  = audio

            except Exception:
                continue

    print(f"  Прочитано сегментов: {len(raw_visual)}")

    # Шаг 2 — проверяем диапазон после обрезки
    train_visual_raw = np.vstack([raw_visual[sid] for sid in train_ids if sid in raw_visual])
    train_audio_raw  = np.vstack([raw_audio[sid]  for sid in train_ids if sid in raw_audio])

    print(f"\n  Визуал после clip({VISUAL_RAW_CLIP}):")
    print(f"    min={train_visual_raw.min():.2f}, max={train_visual_raw.max():.2f}")
    print(f"    shape={train_visual_raw.shape}")

    print(f"\n  Аудио:")
    print(f"    min={train_audio_raw.min():.2f}, max={train_audio_raw.max():.2f}")
    print(f"    shape={train_audio_raw.shape}")

    # Шаг 3 — новые скейлеры
    audio_scaler  = StandardScaler().fit(train_audio_raw)
    visual_scaler = StandardScaler().fit(train_visual_raw)

    # Шаг 4 — нормализуем все сплиты
    result = {'train': {}, 'val': {}, 'test': {}}

    for split_name, split_dict in [('train', data['train']),
                                    ('val',   data['val']),
                                    ('test',  data['test'])]:
        for seg_id, sample in split_dict.items():
            if seg_id not in raw_visual:
                continue

            audio_n  = np.clip(audio_scaler.transform(raw_audio[seg_id]),    -5, 5)
            visual_n = np.clip(visual_scaler.transform(raw_visual[seg_id]),   -5, 5)

            result[split_name][seg_id] = {
                'audio':  audio_n.astype(np.float32),
                'visual': visual_n.astype(np.float32),
                'text':   sample['text'],
                'label':  sample['label'],
            }

    # Шаг 5 — проверяем результат
    ex_a = next(iter(result['train'].values()))
    print(f"\n  После новой нормализации (пример):")
    print(f"    аудио  mean={ex_a['audio'].mean():.4f},  std={ex_a['audio'].std():.4f}")
    print(f"    визуал mean={ex_a['visual'].mean():.4f}, std={ex_a['visual'].std():.4f}")

    # Считаем сколько сэмплов в каждом сплите
    for name, spl in result.items():
        counts = defaultdict(int)
        for s in spl.values():
            counts[s['label']] += 1
        counts_str = ', '.join(f"{CLASS_NAMES[k]}={v}" for k, v in sorted(counts.items()))
        print(f"    {name:5s}: {len(spl):5d} сэмплов  ({counts_str})")

    return result, audio_scaler, visual_scaler


if __name__ == "__main__":

    # Загружаем существующий pkl
    data = load_existing(CLEAN_PATH)

    # Пересчитываем нормализацию
    normalized, audio_scaler, visual_scaler = recompute_normalization(data)

    # Перезаписываем pkl
    output = {
        'train':         normalized['train'],
        'val':           normalized['val'],
        'test':          normalized['test'],
        'audio_scaler':  audio_scaler,
        'visual_scaler': visual_scaler,
        'class_names':   CLASS_NAMES,
    }

    with open(CLEAN_PATH, 'wb') as f:
        pickle.dump(output, f)

    print(f"\n{'='*50}")
    print(f"✓ mosei_clean.pkl перезаписан с исправленной нормализацией!")
    print(f"  train: {len(normalized['train'])} сэмплов")
    print(f"  val:   {len(normalized['val'])} сэмплов")
    print(f"  test:  {len(normalized['test'])} сэмплов")
    print(f"{'='*50}")