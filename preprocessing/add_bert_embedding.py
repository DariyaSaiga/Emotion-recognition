import argparse
import pickle
import numpy as np
import torch
import h5py
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Служебные токены которые нужно игнорировать
SKIP_TOKENS = {'sp', '', 'SP'}


def get_words(hdf5_file, seg_id):

    try:
        raw = hdf5_file['words'][seg_id]['features'][()]
        # raw: numpy array shape (n_words, 1), dtype=|S32 (байтовые строки)

        words_list = []
        for item in raw:
            word = item[0].decode('utf-8').strip().lower()
            if word not in SKIP_TOKENS and len(word) > 0:
                words_list.append(word)

        sentence = ' '.join(words_list)
        return words_list, sentence

    except Exception:
        return [], ''


def compute_bert_embeddings(sentences, tokenizer, bert_model, device,
                             max_length=50, batch_size=32):

    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        # Токенизируем батч
        encoded = tokenizer(
            batch_sentences,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )

        input_ids      = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # last_hidden_state: [batch, max_length, 768]
            hidden = outputs.last_hidden_state

        # Переводим в numpy и сохраняем
        hidden_np = hidden.cpu().numpy().astype(np.float32)
        for emb in hidden_np:
            all_embeddings.append(emb)  # (max_length, 768)

    return all_embeddings


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # --- Загружаем BERT ---
    print("\nЗагружаем BERT tokenizer и модель...")
    tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device)
    bert_model.eval()  # замораживаем — только inference
    print("BERT загружен.")

    # --- Загружаем pkl ---
    print(f"\nЗагружаем {args.pkl_path}...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Собираем все seg_id из всех сплитов
    all_segments = {}
    for split in ['train', 'val', 'test']:
        for seg_id in data[split].keys():
            all_segments[seg_id] = split

    print(f"Всего сэмплов: {len(all_segments)}")

    # --- Читаем слова из HDF5 ---
    print(f"\nЧитаем слова из {args.hdf5_path}...")
    seg_ids   = list(all_segments.keys())
    sentences = []
    valid_ids = []

    with h5py.File(args.hdf5_path, 'r') as f:
        for seg_id in tqdm(seg_ids, desc="Читаем слова"):
            words_list, sentence = get_words(f, seg_id)

            if len(sentence.strip()) == 0:
                # Пустое предложение — ставим заглушку
                sentence = 'unknown'

            sentences.append(sentence)
            valid_ids.append(seg_id)

    print(f"\nПример предложений:")
    for i in range(3):
        print(f"  [{valid_ids[i]}]: '{sentences[i]}'")

    # --- Считаем BERT эмбеддинги ---
    print(f"\nСчитаем BERT эмбеддинги для {len(sentences)} предложений...")

    embeddings = compute_bert_embeddings(
        sentences, tokenizer, bert_model, device,
        max_length=50,
        batch_size=64,  # батч для BERT inference
    )

    print(f"Готово! Форма одного эмбеддинга: {embeddings[0].shape}")
    # Ожидаем: (50, 768)

    # --- Обновляем pkl ---
    print("\nОбновляем mosei_clean.pkl...")

    updated = 0
    for seg_id, emb in zip(valid_ids, embeddings):
        split = all_segments[seg_id]
        data[split][seg_id]['text'] = emb  # заменяем None на реальные эмбеддинги
        updated += 1

    print(f"Обновлено сэмплов: {updated}")

    # Проверяем
    example = next(iter(data['train'].values()))
    print(f"\nПроверка:")
    print(f"  text shape: {example['text'].shape}  ← ожидаем (50, 768)")
    print(f"  text dtype: {example['text'].dtype}")
    print(f"  text mean:  {example['text'].mean():.4f}")

    # --- Сохраняем обновлённый pkl ---
    print(f"\nСохраняем обновлённый pkl...")
    with open(args.pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Готово! pkl обновлён: {args.pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='Путь к HDF5 файлу')
    parser.add_argument('--pkl_path',  type=str, required=True,
                        help='Путь к mosei_clean.pkl (будет обновлён)')
    args = parser.parse_args()
    main(args)