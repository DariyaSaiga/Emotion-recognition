"""
ШАГ 2 — Читаем, чистим и нормализуем CMU-MOSEI.
Исправленная версия под реальную структуру файла.

Запуск: python3 step2_preprocess.py
"""

import h5py
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# -------------------------------------------------------
# ПОМЕНЯЙ ПУТИ НА СВОИ
# -------------------------------------------------------
HDF5_PATH   = "/Users/Лейла/Downloads/mosei.hdf5"
OUTPUT_PATH = "/Users/Лейла/Desktop/mosei_clean.pkl"
# -------------------------------------------------------

# All Labels: [sentiment, happiness, sadness, anger, surprise, disgust, fear]
# Индексы:         0           1        2       3       4         5      6
CLASS_MAP   = {'happiness': 0, 'sadness': 1, 'anger': 2}
CLASS_NAMES = {0: 'happy', 1: 'sad', 2: 'anger'}


# ============================================================
# ЧАСТЬ 1 — Читаем данные из HDF5
# ============================================================

def read_hdf5(path):
    print("\n[1/4] Читаем HDF5 файл...")
    data = {}
    skipped = 0

    with h5py.File(path, 'r') as f:
        all_ids = list(f['COVAREP'].keys())
        print(f"    Всего сегментов: {len(all_ids)}")

        for seg_id in all_ids:
            try:
                audio  = f['COVAREP'][seg_id]['features'][()].astype(np.float32)
                visual = f['OpenFace_2'][seg_id]['features'][()].astype(np.float32)
                labels = f['All Labels'][seg_id]['features'][()].astype(np.float32).flatten()

                # Текст (words) — если есть
                text = None
                if 'words' in f:
                    try:
                        text = f['words'][seg_id]['features'][()].astype(np.float32)
                    except:
                        text = None

                data[seg_id] = {
                    'audio':  audio,
                    'visual': visual,
                    'labels': labels,
                    'text':   text,
                }

            except Exception as e:
                skipped += 1
                continue

    print(f"    Успешно прочитано: {len(data)}")
    print(f"    Пропущено: {skipped}")
    return data


# ============================================================
# ЧАСТЬ 2 — Определяем лейбл
# ============================================================

def assign_label(labels_flat):
    """
    labels_flat: [7] = [sentiment, happiness, sadness, anger, ...]
    Берём доминирующую из трёх нужных эмоций.
    Порог 0.5 — эмоция должна быть достаточно выражена.
    """
    scores = {
        'happiness': labels_flat[1],
        'sadness':   labels_flat[2],
        'anger':     labels_flat[3],
    }
    dominant = max(scores, key=scores.get)
    if scores[dominant] < 0.3:
        return None
    return CLASS_MAP[dominant]


# ============================================================
# ЧАСТЬ 3 — Чистим данные
# ============================================================

def clean_data(raw_data):
    print("\n[2/4] Чистим данные...")

    clean = {}
    reasons = defaultdict(int)
    label_counts = defaultdict(int)

    for seg_id, sample in raw_data.items():

        label = assign_label(sample['labels'])
        if label is None:
            reasons['слабая эмоция (< 0.5)'] += 1
            continue

        if sample['audio'].shape[0] < 2:
            reasons['аудио слишком короткое'] += 1
            continue

        if sample['visual'].shape[0] < 2:
            reasons['визуал слишком короткий'] += 1
            continue

        audio  = np.nan_to_num(sample['audio'],  nan=0.0, posinf=0.0, neginf=0.0)
        visual = np.nan_to_num(sample['visual'], nan=0.0, posinf=0.0, neginf=0.0)
        text   = np.nan_to_num(sample['text'],   nan=0.0, posinf=0.0, neginf=0.0) \
                 if sample['text'] is not None else None

        clean[seg_id] = {
            'audio':  audio,
            'visual': visual,
            'text':   text,
            'label':  label,
        }
        label_counts[label] += 1

    print(f"    После очистки: {len(clean)} сегментов")
    print(f"    Причины пропуска:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"        {reason}: {count}")
    print(f"    Распределение классов:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"        {CLASS_NAMES[lbl]}: {cnt}")

    return clean, label_counts


# ============================================================
# ЧАСТЬ 4 — Балансировка классов
# ============================================================

def balance_classes(clean, label_counts):
    """
    Ограничиваем самый большой класс: максимум = 1.5 × меньший класс.
    """
    print("\n    Балансируем классы...")

    min_count   = min(label_counts.values())
    max_allowed = int(min_count * 1.5)

    per_class = defaultdict(list)
    for seg_id, sample in clean.items():
        per_class[sample['label']].append(seg_id)

    np.random.seed(42)
    balanced = {}
    for lbl, ids in per_class.items():
        if len(ids) > max_allowed:
            ids = list(np.random.choice(ids, max_allowed, replace=False))
        for sid in ids:
            balanced[sid] = clean[sid]

    new_counts = defaultdict(int)
    for s in balanced.values():
        new_counts[s['label']] += 1

    print(f"    После балансировки: {len(balanced)} сегментов")
    for lbl, cnt in sorted(new_counts.items()):
        print(f"        {CLASS_NAMES[lbl]}: {cnt}")

    return balanced


# ============================================================
# ЧАСТЬ 5 — Разбивка train/val/test по video_id
# ============================================================

def split_data(clean):
    print("\n[3/4] Разбиваем на train/val/test по video_id...")

    video_ids = list(set(seg_id.split('[')[0] for seg_id in clean.keys()))
    np.random.seed(42)
    np.random.shuffle(video_ids)

    n       = len(video_ids)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_videos = set(video_ids[:n_train])
    val_videos   = set(video_ids[n_train:n_train + n_val])
    test_videos  = set(video_ids[n_train + n_val:])

    splits = {'train': {}, 'val': {}, 'test': {}}
    for seg_id, sample in clean.items():
        vid = seg_id.split('[')[0]
        if vid in train_videos:
            splits['train'][seg_id] = sample
        elif vid in val_videos:
            splits['val'][seg_id] = sample
        else:
            splits['test'][seg_id] = sample

    for name, spl in splits.items():
        counts = defaultdict(int)
        for s in spl.values():
            counts[s['label']] += 1
        counts_str = ', '.join(f"{CLASS_NAMES[k]}={v}" for k, v in sorted(counts.items()))
        print(f"    {name:5s}: {len(spl):5d} сегментов  ({counts_str})")

    return splits


# ============================================================
# ЧАСТЬ 6 — Нормализация
# ============================================================

def normalize(splits):
    print("\n[4/4] Нормализуем аудио и визуал...")
    print("    (статистика считается ТОЛЬКО по train)")

    train_audio  = np.vstack([s['audio']  for s in splits['train'].values()])
    train_visual = np.vstack([s['visual'] for s in splits['train'].values()])

    print(f"    Train аудио матрица:  {train_audio.shape}")
    print(f"    Train визуал матрица: {train_visual.shape}")
    print(f"    Аудио до норм:  min={train_audio.min():.2f}, max={train_audio.max():.2f}")
    print(f"    Визуал до норм: min={train_visual.min():.2f}, max={train_visual.max():.2f}")

    audio_scaler  = StandardScaler().fit(train_audio)
    visual_scaler = StandardScaler().fit(train_visual)

    result = {}
    for split_name, split_data in splits.items():
        result[split_name] = {}
        for seg_id, sample in split_data.items():
            audio_n  = np.clip(audio_scaler.transform(sample['audio']),   -5, 5)
            visual_n = np.clip(visual_scaler.transform(sample['visual']), -5, 5)

            result[split_name][seg_id] = {
                'audio':  audio_n.astype(np.float32),
                'visual': visual_n.astype(np.float32),
                'text':   sample['text'],
                'label':  sample['label'],
            }

    # Проверяем после нормализации
    ex = next(iter(result['train'].values()))
    print(f"    Аудио после норм:  mean={ex['audio'].mean():.4f}, std={ex['audio'].std():.4f}")
    print(f"    Визуал после норм: mean={ex['visual'].mean():.4f}, std={ex['visual'].std():.4f}")

    return result, audio_scaler, visual_scaler


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    raw_data = read_hdf5(HDF5_PATH)
    clean, label_counts = clean_data(raw_data)
    clean = balance_classes(clean, label_counts)
    splits = split_data(clean)
    normalized, audio_scaler, visual_scaler = normalize(splits)

    output = {
        'train':         normalized['train'],
        'val':           normalized['val'],
        'test':          normalized['test'],
        'audio_scaler':  audio_scaler,
        'visual_scaler': visual_scaler,
        'class_names':   CLASS_NAMES,
    }

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(output, f)

    print(f"\n{'='*50}")
    print(f"✓ Готово! Файл сохранён: {OUTPUT_PATH}")
    print(f"  train: {len(normalized['train'])} сэмплов")
    print(f"  val:   {len(normalized['val'])} сэмплов")
    print(f"  test:  {len(normalized['test'])} сэмплов")
    print(f"{'='*50}")
    print("\nСкинь этот вывод в чат — переходим к dataset.py!")