"""
add_bert_embeddings_v2.py

Правильный способ считать BERT эмбеддинги:
- Собираем полное предложение из слов HDF5
- Подаём предложение целиком в BERT
- Берём last_hidden_state — контекстуальные эмбеддинги каждого токена
- Выравниваем до MAX_SEQ_LEN=50

Запуск:
    !python add_bert_embeddings_v2.py \
        --hdf5_path /content/drive/MyDrive/Diploma2/mosei.hdf5 \
        --pkl_path  /content/drive/MyDrive/Diploma2/mosei_full.pkl

Время: ~30-40 минут на T4
"""

import argparse
import pickle
import numpy as np
import torch
import h5py
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm

SKIP = {'sp', '', 'SP', '-'}
MAX_SEQ_LEN = 50


def get_sentence(hdf5_file, seg_id):
    """Собираем полное предложение из слов сегмента."""
    try:
        raw   = hdf5_file['words'][seg_id]['features'][()]
        words = [w[0].decode('utf-8').strip().lower()
                 for w in raw
                 if w[0].decode('utf-8').strip() not in SKIP]
        return ' '.join(words) if words else 'unknown'
    except Exception:
        return 'unknown'


def compute_bert_batch(sentences, tokenizer, model, device, max_len=50, batch_size=64):
    """
    Считаем BERT эмбеддинги для списка предложений.

    Правильная схема:
    1. Токенизируем полное предложение (не слова по отдельности)
    2. Получаем last_hidden_state [batch, max_len, 768]
    3. Каждый токен имеет контекст всего предложения

    Возвращает список numpy массивов shape (MAX_SEQ_LEN, 768).
    """
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
        )

        input_ids      = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: [batch, max_len, 768]
            hidden = out.last_hidden_state

        for emb in hidden.cpu().numpy().astype(np.float32):
            all_embeddings.append(emb)  # (max_len, 768)

    return all_embeddings


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nЗагружаем BERT...')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model     = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    print('BERT загружен.')

    print(f'\nЗагружаем {args.pkl_path}...')
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Все seg_id из всех сплитов
    seg_to_split = {}
    for split in ['train', 'val', 'test']:
        for seg_id in data[split]:
            seg_to_split[seg_id] = split

    print(f'Всего сэмплов: {len(seg_to_split)}')

    # Собираем предложения из HDF5
    print(f'\nСобираем предложения из {args.hdf5_path}...')
    seg_ids   = list(seg_to_split.keys())
    sentences = []

    with h5py.File(args.hdf5_path, 'r') as f:
        for seg_id in tqdm(seg_ids, desc='Читаем слова'):
            sentences.append(get_sentence(f, seg_id))

    # Показываем примеры
    print('\nПримеры предложений:')
    for i in range(3):
        print(f'  [{seg_ids[i]}]: "{sentences[i][:80]}"')

    # Проверяем качество
    empty = sum(1 for s in sentences if s == 'unknown')
    print(f'\nПустых предложений: {empty}/{len(sentences)}')

    # Считаем BERT эмбеддинги
    print(f'\nСчитаем BERT эмбеддинги для {len(sentences)} предложений...')
    embeddings = compute_bert_batch(
        sentences, tokenizer, model, device,
        max_len=MAX_SEQ_LEN,
        batch_size=64,
    )
    print(f'Готово! Форма эмбеддинга: {embeddings[0].shape}')  # (50, 768)

    # Проверяем качество эмбеддингов
    ex = embeddings[0]
    print(f'Mean: {ex.mean():.4f}, Std: {ex.std():.4f}')
    print(f'Норма [CLS] токена: {np.linalg.norm(ex[0]):.4f}')

    # Обновляем pkl
    print('\nОбновляем pkl...')
    updated = 0
    for seg_id, emb in zip(seg_ids, embeddings):
        split = seg_to_split[seg_id]
        data[split][seg_id]['text'] = emb
        updated += 1

    print(f'Обновлено: {updated} сэмплов')

    # Проверка
    ex_sample = next(iter(data['train'].values()))
    print(f'\nПроверка:')
    print(f'  text shape: {ex_sample["text"].shape}  ← ожидаем (50, 768)')
    print(f'  text mean:  {ex_sample["text"].mean():.4f}')
    print(f'  text std:   {ex_sample["text"].std():.4f}')

    # Сохраняем
    print(f'\nСохраняем {args.pkl_path}...')
    with open(args.pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f'✓ Готово!')
    print('\nТеперь запускай обучение:')
    print('  !python train.py --model baseline  --epochs 50 ...')
    print('  !python train.py --model bottleneck --epochs 50 ...')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5_path', required=True)
    p.add_argument('--pkl_path',  required=True)
    args = p.parse_args()
    main(args)