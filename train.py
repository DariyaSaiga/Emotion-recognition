import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer
from tqdm import tqdm

from dataset import get_dataloaders
from baseline import LateFusionBaseline

# ══════════════════════════════════════════════
# НАСТРОЙКИ — меняй только здесь
# ══════════════════════════════════════════════
CONFIG = {
    'data_root':   '/Users/dariyaablanova/Desktop/unic_work/Diploma/model_learning/CMU-MOSEI/CMU-MOSEI_CLEAN_5k',
    'batch_size':  16,
    'epochs':      10,
    'lr':          1e-3,
    'num_workers': 0,
    'num_classes': 7,
    'output_dir':  'results/baseline',
}

# ══════════════════════════════════════════════
# ТОКЕНИЗАТОР BERT
# ══════════════════════════════════════════════
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def collate_fn(batch):
    """
    Собирает батч: токенизирует тексты, складывает тензоры.
    Это нужно потому что BERT принимает токены, а не сырой текст.
    """
    mels    = torch.stack([b['mel'] for b in batch])
    labels  = torch.stack([b['label'] for b in batch])
    texts   = [b['text'] for b in batch]

    # Токенизируем все тексты батча сразу
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    return {
        'mel':            mels,
        'text_inputs':    encoded,   # input_ids + attention_mask
        'label':          labels,
    }

# ══════════════════════════════════════════════
# МЕТРИКИ
# ══════════════════════════════════════════════
def compute_metrics(all_preds, all_labels):
    wa = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return wa, f1

# ══════════════════════════════════════════════
# TRAIN EPOCH
# ══════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc='  train', leave=False):
        mel         = batch['mel'].to(device)
        text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
        labels      = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(text_inputs, mel)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    wa, f1 = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), wa, f1

# ══════════════════════════════════════════════
# EVAL EPOCH
# ══════════════════════════════════════════════
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False):
            mel         = batch['mel'].to(device)
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            labels      = batch['label'].to(device)


            logits = model(text_inputs, mel)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    wa, f1 = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), wa, f1

# ══════════════════════════════════════════════
# ГРАФИКИ
# ══════════════════════════════════════════════
def save_plots(history, output_dir):
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, history['train_loss'], 'b-o', markersize=4, label='Train')
    axes[0].plot(ep, history['val_loss'],   'r-o', markersize=4, label='Val')
    axes[0].set_title('Loss по эпохам')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history['train_wa'], 'b-o', markersize=4, label='Train WA')
    axes[1].plot(ep, history['val_wa'],   'r-o', markersize=4, label='Val WA')
    axes[1].set_title('Weighted Accuracy по эпохам')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('WA')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('Baseline — Late Fusion', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Графики сохранены → {path}")

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")

    # Данные
    print("\nЗагрузка данных...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
    )

    # Модель из baseline.py второго участника
    model = LateFusionBaseline(num_classes=CONFIG['num_classes']).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nМодель: LateFusionBaseline")
    print(f"Обучаемые параметры: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history = {k: [] for k in
               ['train_loss', 'val_loss', 'train_wa', 'val_wa']}

    best_val_wa = 0.0
    best_epoch  = 0
    model_path  = os.path.join(CONFIG['output_dir'], 'best_model.pth')

    print(f"\n{'─'*50}")
    print(f"Обучение: {CONFIG['epochs']} эпох")
    print(f"{'─'*50}\n")

    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"Эпоха {epoch:02d}/{CONFIG['epochs']}")

        tr_loss, tr_wa, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_wa, vl_f1 = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_wa'].append(tr_wa)
        history['val_wa'].append(vl_wa)

        marker = ' ← лучшая' if vl_wa > best_val_wa else ''
        print(f"  Train → loss: {tr_loss:.4f}  WA: {tr_wa:.4f}  F1: {tr_f1:.4f}")
        print(f"  Val   → loss: {vl_loss:.4f}  WA: {vl_wa:.4f}  F1: {vl_f1:.4f}{marker}\n")

        if vl_wa > best_val_wa:
            best_val_wa = vl_wa
            best_epoch  = epoch
            torch.save(model.state_dict(), model_path)

    # Финальный тест
    print('─' * 50)
    print('Финальный тест...')
    model.load_state_dict(torch.load(model_path))
    ts_loss, ts_wa, ts_f1 = eval_epoch(model, test_loader, criterion, device)

    print(f"\n{'═'*50}")
    print(f"  РЕЗУЛЬТАТЫ BASELINE")
    print(f"{'═'*50}")
    print(f"  Weighted Accuracy (WA) : {ts_wa:.4f} ({ts_wa*100:.2f}%)")
    print(f"  Weighted F1-score      : {ts_f1:.4f} ({ts_f1*100:.2f}%)")
    print(f"  Лучшая эпоха           : {best_epoch}")
    print(f"{'═'*50}\n")

    # Сохраняем результаты
    with open(os.path.join(CONFIG['output_dir'], 'results.txt'), 'w') as f:
        f.write("BASELINE — Late Fusion\n")
        f.write("=" * 40 + "\n")
        f.write(f"Weighted Accuracy (WA) : {ts_wa:.4f} ({ts_wa*100:.2f}%)\n")
        f.write(f"Weighted F1-score      : {ts_f1:.4f} ({ts_f1*100:.2f}%)\n")
        f.write(f"Лучшая эпоха           : {best_epoch} / {CONFIG['epochs']}\n")
        f.write(f"Learning rate          : {CONFIG['lr']}\n")
        f.write(f"Batch size             : {CONFIG['batch_size']}\n")

    save_plots(history, CONFIG['output_dir'])
    print("Готово! Результаты в папке:", CONFIG['output_dir'])


if __name__ == '__main__':
    main()
