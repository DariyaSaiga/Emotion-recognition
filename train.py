import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

from dataset import get_dataloaders, CLASS_WEIGHTS, EMOTION_NAMES


# ══════════════════════════════════════════════════════════════════
# АРГУМЕНТЫ
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение модели на CMU-MOSEI')
    parser.add_argument('--model',      type=str, default='baseline',
                        choices=['baseline', 'bottleneck',
                                 'text_only', 'audio_only', 'visual_only'],
                        help='Тип модели')
    parser.add_argument('--csv',        type=str, required=True,
                        help='Путь к cmu_mosei_final.csv')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Папка для сохранения (по умолчанию results/<model>)')
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--seed',       type=int, default=42)
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# ВОСПРОИЗВОДИМОСТЬ
# ══════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════
# МЕТРИКИ
# ══════════════════════════════════════════════════════════════════

def compute_metrics(all_preds, all_labels):
    """Возвращает Weighted Accuracy (WA) и Weighted F1."""
    wa = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return wa, f1


# ══════════════════════════════════════════════════════════════════
# TRAIN / EVAL EPOCH
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc='  train', leave=False):
        text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
        audio_feat  = batch['audio_feat'].to(device)
        visual_feat = batch['visual_feat'].to(device)
        labels      = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(text_inputs, audio_feat, visual_feat)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — стабилизирует обучение
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    wa, f1 = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), wa, f1


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False):
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            audio_feat  = batch['audio_feat'].to(device)
            visual_feat = batch['visual_feat'].to(device)
            labels      = batch['label'].to(device)

            logits = model(text_inputs, audio_feat, visual_feat)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    wa, f1 = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), wa, f1, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ══════════════════════════════════════════════════════════════════

def save_plots(history, model_name, output_dir):
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, history['train_loss'], 'b-o', ms=3, label='Train')
    axes[0].plot(ep, history['val_loss'],   'r-o', ms=3, label='Val')
    axes[0].set_title('Loss по эпохам')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history['train_wa'], 'b-o', ms=3, label='Train WA')
    axes[1].plot(ep, history['val_wa'],   'r-o', ms=3, label='Val WA')
    axes[1].plot(ep, history['train_f1'], 'b--', ms=3, alpha=0.5, label='Train F1')
    axes[1].plot(ep, history['val_f1'],   'r--', ms=3, alpha=0.5, label='Val F1')
    axes[1].set_title('Метрики по эпохам')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'{model_name} — CMU-MOSEI', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Графики → {path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir or os.path.join('results', args.model)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'═'*50}")
    print(f"  Модель:    {args.model.upper()}")
    print(f"  Устройство: {device}")
    print(f"  CSV:       {args.csv}")
    print(f"{'═'*50}\n")

    # ── Данные ────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Модель ────────────────────────────────────────────────────
    if args.model == 'baseline':
        from baseline import LateFusionBaseline
        model = LateFusionBaseline().to(device)
    elif args.model == 'bottleneck':
        from bottleneck import BottleneckTransformer
        model = BottleneckTransformer().to(device)
    elif args.model == 'text_only':
        from baseline import TextOnlyModel
        model = TextOnlyModel().to(device)
    elif args.model == 'audio_only':
        from baseline import AudioOnlyModel
        model = AudioOnlyModel().to(device)
    elif args.model == 'visual_only':
        from baseline import VisualOnlyModel
        model = VisualOnlyModel().to(device)
    else:
        raise ValueError(f"Неизвестная модель: {args.model}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nПараметры всего:     {total:,}")
    print(f"Параметры обучаемых: {trainable:,}\n")

    # ── Loss с весами классов ─────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device)
    )

    # ── Оптимизатор ───────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    # ── История ───────────────────────────────────────────────────
    history = {k: [] for k in
               ['train_loss', 'val_loss', 'train_wa', 'val_wa',
                'train_f1', 'val_f1']}

    best_val_wa = 0.0
    best_epoch  = 0
    model_path  = os.path.join(output_dir, 'best_model.pth')

    print(f"{'─'*50}")
    print(f"Начинаем обучение: {args.epochs} эпох")
    print(f"{'─'*50}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Эпоха {epoch:02d}/{args.epochs}")

        tr_loss, tr_wa, tr_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_wa, vl_f1, _, _ = eval_epoch(
            model, val_loader, criterion, device)

        scheduler.step(vl_wa)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_wa'].append(tr_wa)
        history['val_wa'].append(vl_wa)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(vl_f1)

        marker = ' ← лучшая' if vl_wa > best_val_wa else ''
        print(f"  Train → loss={tr_loss:.4f}  WA={tr_wa:.4f}  F1={tr_f1:.4f}")
        print(f"  Val   → loss={vl_loss:.4f}  WA={vl_wa:.4f}  F1={vl_f1:.4f}{marker}\n")

        if vl_wa > best_val_wa:
            best_val_wa = vl_wa
            best_epoch  = epoch
            torch.save(model.state_dict(), model_path)
        
        

    # ── Финальный тест ────────────────────────────────────────────
    print('─' * 50)
    print('Финальный тест...')
    model.load_state_dict(torch.load(model_path))
    ts_loss, ts_wa, ts_f1, preds, labels_true = eval_epoch(
        model, test_loader, criterion, device)

    emotion_list = [EMOTION_NAMES[i] for i in range(3)]
    report = classification_report(
        labels_true, preds,
        target_names=emotion_list,
        zero_division=0
    )

    print(f"\n{'═'*50}")
    print(f"  РЕЗУЛЬТАТЫ: {args.model.upper()}")
    print(f"{'═'*50}")
    print(f"  Weighted Accuracy (WA): {ts_wa:.4f} ({ts_wa*100:.2f}%)")
    print(f"  Weighted F1-score     : {ts_f1:.4f} ({ts_f1*100:.2f}%)")
    print(f"  Лучшая эпоха          : {best_epoch} / {args.epochs}")
    print(f"\n{report}")

    # Сохраняем результаты
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"МОДЕЛЬ: {args.model.upper()}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Weighted Accuracy (WA): {ts_wa:.4f} ({ts_wa*100:.2f}%)\n")
        f.write(f"Weighted F1-score     : {ts_f1:.4f} ({ts_f1*100:.2f}%)\n")
        f.write(f"Лучшая эпоха          : {best_epoch} / {args.epochs}\n")
        f.write(f"Learning rate         : {args.lr}\n")
        f.write(f"Batch size            : {args.batch_size}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    save_plots(history, args.model.upper(), output_dir)

    print(f"\nРезультаты сохранены в: {output_dir}/")
    print(f"  best_model.pth")
    print(f"  results.txt")
    print(f"  training_curves.png")


if __name__ == '__main__':
    main()
