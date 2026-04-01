"""
preprocess.py — полный препроцессинг CMU-MOSEI за один запуск

Что делает:
1. Читает все 23,244 сэмпла из HDF5
2. Нормализует аудио и визуал
3. Считает BERT эмбеддинги из полных предложений → [768] вектор
4. Разбивает на train/val/test по video_id
5. Сохраняет готовый pkl ~1.5 GB

Запуск в Colab:
    !python preprocess.py \
        --hdf5_path /content/drive/MyDrive/Diploma2/mosei.hdf5 \
        --output    /content/drive/MyDrive/Diploma2/mosei_v3.pkl

Время: ~35-40 минут (BERT на 23k предложений)
"""

import argparse
import pickle
import numpy as np
import torch
import h5py
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizerFast, BertModel
from collections import defaultdict
from tqdm import tqdm

SKIP        = {'sp', '', 'SP', '-'}
CLASS_MAP   = {'happiness': 0, 'sadness': 1, 'anger': 2}
CLASS_NAMES = {0: 'happy', 1: 'sad', 2: 'anger'}


# ── 1. Читаем аудио / визуал / лейблы ──────────────────────────

def read_hdf5(path):
    print('\n[1/4] Читаем HDF5...')
    data = {}
    with h5py.File(path, 'r') as f:
        all_ids = list(f['COVAREP'].keys())
        for sid in tqdm(all_ids, desc='  читаем'):
            try:
                audio  = f['COVAREP'][sid]['features'][()].astype(np.float32)
                visual = f['OpenFace_2'][sid]['features'][()].astype(np.float32)
                labels = f['All Labels'][sid]['features'][()].astype(np.float32).flatten()

                if audio.shape[0] < 2 or visual.shape[0] < 2:
                    continue

                scores   = {'happiness': labels[1],
                            'sadness':   labels[2],
                            'anger':     labels[3]}
                dominant = max(scores, key=scores.get)

                audio  = np.nan_to_num(audio,  nan=0.0, posinf=0.0, neginf=0.0)
                visual = np.nan_to_num(visual, nan=0.0, posinf=0.0, neginf=0.0)
                visual = np.clip(visual, -1000, 1000)

                data[sid] = {
                    'audio':  audio,
                    'visual': visual,
                    'label':  CLASS_MAP[dominant],
                }
            except Exception:
                continue

    counts = defaultdict(int)
    for s in data.values(): counts[s['label']] += 1
    print(f'  Сэмплов: {len(data)}')
    print(f'  happy={counts[0]}  sad={counts[1]}  anger={counts[2]}')
    return data


# ── 2. Разбивка по video_id ─────────────────────────────────────

def split_data(data):
    print('\n[2/4] Разбиваем train/val/test по video_id...')
    video_ids = list(set(sid.split('[')[0] for sid in data.keys()))
    np.random.seed(42)
    np.random.shuffle(video_ids)
    n       = len(video_ids)
    train_v = set(video_ids[:int(n * 0.70)])
    val_v   = set(video_ids[int(n * 0.70):int(n * 0.85)])

    splits = {'train': {}, 'val': {}, 'test': {}}
    for sid, s in data.items():
        vid = sid.split('[')[0]
        if vid in train_v:   splits['train'][sid] = s
        elif vid in val_v:   splits['val'][sid]   = s
        else:                splits['test'][sid]  = s

    for name, spl in splits.items():
        c = defaultdict(int)
        for s in spl.values(): c[s['label']] += 1
        print(f'  {name}: {len(spl):6d}  '
              f'happy={c[0]:5d}  sad={c[1]:4d}  anger={c[2]:4d}')
    return splits


# ── 3. Нормализация аудио и визуала ────────────────────────────

def normalize(splits):
    print('\n[3/4] Нормализуем аудио и визуал (по train)...')
    train_audio  = np.vstack([s['audio']  for s in splits['train'].values()])
    train_visual = np.vstack([s['visual'] for s in splits['train'].values()])

    audio_scaler  = StandardScaler().fit(train_audio)
    visual_scaler = StandardScaler().fit(train_visual)

    for sname, spl in splits.items():
        for sid, s in spl.items():
            s['audio']  = np.clip(
                audio_scaler.transform(s['audio']),  -5, 5).astype(np.float32)
            s['visual'] = np.clip(
                visual_scaler.transform(s['visual']), -5, 5).astype(np.float32)

    print(f'  аудио  после норм: mean={train_audio.mean():.3f}')
    print(f'  визуал после норм: mean={train_visual.mean():.3f}')
    return splits, audio_scaler, visual_scaler


# ── 4. BERT эмбеддинги ─────────────────────────────────────────

def get_sentence(hdf5_file, sid):
    try:
        raw   = hdf5_file['words'][sid]['features'][()]
        words = [w[0].decode('utf-8').strip().lower()
                 for w in raw
                 if w[0].decode('utf-8').strip() not in SKIP]
        return ' '.join(words) if words else 'unknown'
    except Exception:
        return 'unknown'


def compute_bert(splits, hdf5_path, device, batch_size=64):
    print('\n[4/4] Считаем BERT эмбеддинги...')
    print('  Загружаем bert-base-uncased...')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model     = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    # Собираем все seg_id
    all_ids = []
    for spl in splits.values():
        all_ids.extend(spl.keys())

    # Читаем предложения
    print(f'  Читаем {len(all_ids)} предложений из HDF5...')
    sentences = {}
    with h5py.File(hdf5_path, 'r') as f:
        for sid in tqdm(all_ids, desc='  слова'):
            sentences[sid] = get_sentence(f, sid)

    # Считаем BERT батчами
    print(f'  Считаем BERT для {len(all_ids)} предложений...')
    embeddings = {}

    for i in tqdm(range(0, len(all_ids), batch_size), desc='  BERT'):
        batch_ids  = all_ids[i:i + batch_size]
        batch_sent = [sentences[sid] for sid in batch_ids]

        enc = tokenizer(
            batch_sent,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt',
        )
        with torch.no_grad():
            out    = model(
                input_ids=enc['input_ids'].to(device),
                attention_mask=enc['attention_mask'].to(device),
            )
            # CLS токен = суммарное представление предложения [batch, 768]
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()

        for sid, emb in zip(batch_ids, cls_emb):
            embeddings[sid] = emb.astype(np.float32)  # [768]

    # Записываем в splits
    for spl in splits.values():
        for sid in spl:
            spl[sid]['text'] = embeddings[sid]  # [768]

    ex = next(iter(splits['train'].values()))
    print(f'\n  text shape: {ex["text"].shape}  ← ожидаем (768,)')
    print(f'  text mean:  {ex["text"].mean():.4f}')
    print(f'  text norm:  {np.linalg.norm(ex["text"]):.4f}')

    # Освобождаем память
    del model
    torch.cuda.empty_cache()

    return splits


# ── MAIN ────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    data   = read_hdf5(args.hdf5_path)
    splits = split_data(data)
    splits, audio_scaler, visual_scaler = normalize(splits)
    splits = compute_bert(splits, args.hdf5_path, device)

    print(f'\nСохраняем {args.output}...')
    output = {
        'train':          splits['train'],
        'val':            splits['val'],
        'test':           splits['test'],
        'audio_scaler':   audio_scaler,
        'visual_scaler':  visual_scaler,
        'class_names':    CLASS_NAMES,
    }
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)

    import os
    size = os.path.getsize(args.output) / 1e9
    print(f'✓ Сохранено! Размер: {size:.2f} GB')
    print(f'\ntrain: {len(splits["train"])}')
    print(f'val:   {len(splits["val"])}')
    print(f'test:  {len(splits["test"])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5_path', required=True)
    p.add_argument('--output',    required=True)
    main(p.parse_args())
