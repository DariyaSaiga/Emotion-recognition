import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import defaultdict

# Подключаем наши файлы
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders
from models import AudioEncoder, VisualEncoder, TextEncoder, BottleneckModel


# ============================================================
# LATE FUSION BASELINE
# Три независимых классификатора — предсказания усредняются
# ============================================================

class LateFusionBaseline(nn.Module):
    def __init__(self, dropout=0.3):
        super(LateFusionBaseline, self).__init__()

        self.audio_encoder  = AudioEncoder(dropout=dropout)
        self.visual_encoder = VisualEncoder(dropout=dropout)
        self.text_encoder   = TextEncoder(dropout=dropout)

        self.audio_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3)
        )
        self.visual_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3)
        )
        self.text_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3)
        )

    def forward(self, audio, visual, text, mask=None):
        audio_feat  = self.audio_encoder(audio)
        visual_feat = self.visual_encoder(visual, mask)
        text_feat   = self.text_encoder(text)

        audio_logits  = self.audio_classifier(audio_feat)
        visual_logits = self.visual_classifier(visual_feat)
        text_logits   = self.text_classifier(text_feat)

        # Late Fusion = среднее предсказаний трёх модальностей
        return (audio_logits + visual_logits + text_logits) / 3.0


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def compute_accuracy(logits, labels):
    preds   = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def compute_per_class_accuracy(logits, labels, num_classes=3):
    preds  = logits.argmax(dim=-1)
    names  = {0: 'happy', 1: 'sad', 2: 'anger'}
    result = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            result[names[c]] = 0.0
            continue
        result[names[c]] = (preds[mask] == labels[mask]).sum().item() / mask.sum().item()
    return result


# ============================================================
# ОДНА ЭПОХА ОБУЧЕНИЯ
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for audio, visual, text, labels, mask in loader:
        audio, visual, text = audio.to(device), visual.to(device), text.to(device)
        labels, mask        = labels.to(device), mask.to(device)

        optimizer.zero_grad()
        logits = model(audio, visual, text, mask)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — ограничиваем градиенты чтобы обучение не ломалось
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    return (total_loss / len(loader),
            compute_accuracy(all_logits, all_labels),
            compute_per_class_accuracy(all_logits, all_labels))


# ============================================================
# ОЦЕНКА НА VAL / TEST
# ============================================================

def evaluate(model, loader, criterion, device):
    """
    Оцениваем без изменения весов.
    torch.no_grad() — не считаем градиенты, быстрее и меньше памяти.
    model.eval()    — выключаем Dropout.
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for audio, visual, text, labels, mask in loader:
            audio, visual, text = audio.to(device), visual.to(device), text.to(device)
            labels, mask        = labels.to(device), mask.to(device)

            logits     = model(audio, visual, text, mask)
            total_loss += criterion(logits, labels).item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    return (total_loss / len(loader),
            compute_accuracy(all_logits, all_labels),
            compute_per_class_accuracy(all_logits, all_labels))


# ============================================================
# ГРАФИКИ
# ============================================================

def save_plots(history, model_name, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'r-', label='Val Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'],   'r-', label='Val Accuracy')
    ax2.axhline(y=0.70, color='g', linestyle='--', alpha=0.7, label='Цель 70%')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    График сохранён: {path}")


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        args.pkl_path, batch_size=args.batch_size, num_workers=0,
    )

    print(f"\nМодель: {args.model}")
    if args.model == 'baseline':
        model = LateFusionBaseline(dropout=args.dropout)
    else:
        model = BottleneckModel(dropout=args.dropout)

    model = model.to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc   = 0.0
    patience_count = 0
    history        = defaultdict(list)

    print(f"\nЭпох: {args.epochs} | Батч: {args.batch_size} | LR: {args.lr}")
    print("=" * 65)
    print(f"{'Эпоха':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc, train_pc = train_epoch(
            model, train_loader, optimizer, criterion, device)

        val_loss, val_acc, val_pc = evaluate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.1%} | {current_lr:>8.2e}")

        if epoch % 5 == 0:
            print(f"         Train: happy={train_pc['happy']:.1%}  "
                  f"sad={train_pc['sad']:.1%}  anger={train_pc['anger']:.1%}")
            print(f"         Val:   happy={val_pc['happy']:.1%}  "
                  f"sad={val_pc['sad']:.1%}  anger={val_pc['anger']:.1%}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            best_path      = os.path.join(save_dir, f'{args.model}_best.pt')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': val_acc}, best_path)
            print(f"         ★ Лучший val acc: {val_acc:.1%} — сохранено")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"\nEarly stopping на эпохе {epoch}")
            break

    print("=" * 65)

    # Финальный тест
    print("\nОцениваем на test...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    test_loss, test_acc, test_pc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*40}")
    print(f"РЕЗУЛЬТАТЫ: {args.model.upper()}")
    print(f"{'='*40}")
    print(f"  Val  Accuracy: {best_val_acc:.1%}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"  По классам:")
    print(f"    happy: {test_pc['happy']:.1%}")
    print(f"    sad:   {test_pc['sad']:.1%}")
    print(f"    anger: {test_pc['anger']:.1%}")
    print(f"{'='*40}")

    save_plots(history, args.model, save_dir)

    metrics = {
        'model': args.model, 'best_val_acc': best_val_acc,
        'test_acc': test_acc, 'test_per_class': test_pc,
        'history': dict(history),
    }
    with open(os.path.join(save_dir, f'{args.model}_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\nВсе файлы сохранены в: {save_dir}")
    return metrics


# ============================================================
# АРГУМЕНТЫ
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str,   default='bottleneck',
                   choices=['baseline', 'bottleneck'])
    p.add_argument('--pkl_path',   type=str,
                   default='/content/drive/MyDrive/Diploma2/mosei_clean.pkl')
    p.add_argument('--output_dir', type=str,
                   default='/content/drive/MyDrive/Diploma/results')
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--dropout',    type=float, default=0.3)
    p.add_argument('--patience',   type=int,   default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())