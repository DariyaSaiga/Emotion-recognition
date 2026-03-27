"""
predict.py — инференс на обученных моделях

Использование в Colab:

    # Предсказание на тестовой части датасета
    !python predict.py \
        --model baseline \
        --checkpoint results/baseline/best_model.pth \
        --csv /content/drive/MyDrive/Diploma/cmu_mosei_final.csv

    # То же самое для bottleneck
    !python predict.py \
        --model bottleneck \
        --checkpoint results/bottleneck/best_model.pth \
        --csv /content/drive/MyDrive/Diploma/cmu_mosei_final.csv

    # Сравнить обе модели
    !python predict.py --compare \
        --baseline_ckpt  results/baseline/best_model.pth \
        --bottleneck_ckpt results/bottleneck/best_model.pth \
        --csv /content/drive/MyDrive/Diploma/cmu_mosei_final.csv
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import seaborn as sns

from dataset import get_dataloaders, CLASS_WEIGHTS, EMOTION_NAMES, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════
# АРГУМЕНТЫ
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, default='baseline',
                        choices=['baseline', 'bottleneck'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Путь к .pth файлу')
    parser.add_argument('--csv',        type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default=None)
    # Режим сравнения двух моделей
    parser.add_argument('--compare',    action='store_true',
                        help='Сравнить baseline vs bottleneck')
    parser.add_argument('--baseline_ckpt',   type=str, default='results/baseline/best_model.pth')
    parser.add_argument('--bottleneck_ckpt', type=str, default='results/bottleneck/best_model.pth')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛИ
# ══════════════════════════════════════════════════════════════════

def load_model(model_type: str, checkpoint_path: str, device):
    """Загружает модель из checkpoint."""
    if model_type == 'baseline':
        from baseline import LateFusionBaseline
        model = LateFusionBaseline()
    elif model_type == 'bottleneck':
        from bottleneck import BottleneckTransformer
        model = BottleneckTransformer()
    else:
        raise ValueError(f"Неизвестная модель: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  ✓ Загружена модель {model_type.upper()} из {checkpoint_path}")
    return model


# ══════════════════════════════════════════════════════════════════
# ИНФЕРЕНС
# ══════════════════════════════════════════════════════════════════

def run_inference(model, loader, device):
    """Прогоняет модель через весь dataloader, возвращает предсказания и метки."""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            audio_feat  = batch['audio_feat'].to(device)
            visual_feat = batch['visual_feat'].to(device)
            labels      = batch['label']

            logits = model(text_inputs, audio_feat, visual_feat)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


# ══════════════════════════════════════════════════════════════════
# МЕТРИКИ И ГРАФИКИ
# ══════════════════════════════════════════════════════════════════

def print_metrics(model_name, preds, labels):
    """Выводит полный отчёт о метриках."""
    wa = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    emotion_list = [EMOTION_NAMES[i] for i in range(NUM_CLASSES)]

    print(f"\n{'═'*50}")
    print(f"  {model_name.upper()} — Результаты на тесте")
    print(f"{'═'*50}")
    print(f"  Weighted Accuracy (WA): {wa:.4f} ({wa*100:.2f}%)")
    print(f"  Weighted F1-score     : {f1:.4f} ({f1*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds,
                                 target_names=emotion_list, zero_division=0))
    return wa, f1


def save_confusion_matrix(preds, labels, model_name, output_dir):
    """Сохраняет матрицу ошибок."""
    emotion_list = [EMOTION_NAMES[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_list, yticklabels=emotion_list, ax=ax)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Истинное')
    ax.set_title(f'Confusion Matrix — {model_name.upper()}')
    plt.tight_layout()

    path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Матрица ошибок → {path}")


def save_comparison_chart(results: dict, output_dir: str):
    """Сравнительный график двух моделей."""
    models   = list(results.keys())
    wa_vals  = [results[m]['wa'] for m in models]
    f1_vals  = [results[m]['f1'] for m in models]

    x    = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, [v*100 for v in wa_vals], width,
                   label='WA (%)', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, [v*100 for v in f1_vals], width,
                   label='F1 (%)', color='coral', alpha=0.85)

    # Подписи на барах
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontsize=12)
    ax.set_ylabel('Score (%)')
    ax.set_title('Сравнение моделей на тесте CMU-MOSEI\n(3 класса: happy / sad / anger)')
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  График сравнения → {path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Загружаем данные (нужен только test split)
    _, _, test_loader = get_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # ── Режим сравнения двух моделей ─────────────────────────────
    if args.compare:
        output_dir = args.output_dir or 'results/comparison'
        os.makedirs(output_dir, exist_ok=True)

        results = {}
        for model_type, ckpt in [
            ('baseline',   args.baseline_ckpt),
            ('bottleneck', args.bottleneck_ckpt)
        ]:
            print(f"\nЗагружаем {model_type}...")
            model = load_model(model_type, ckpt, device)
            preds, labels, _ = run_inference(model, test_loader, device)
            wa, f1 = print_metrics(model_type, preds, labels)
            results[model_type] = {'wa': wa, 'f1': f1}
            save_confusion_matrix(preds, labels, model_type, output_dir)

        save_comparison_chart(results, output_dir)

        # Итоговая таблица
        print(f"\n{'─'*40}")
        print(f"{'Модель':<15} {'WA':>8} {'F1':>8}")
        print(f"{'─'*40}")
        for m, r in results.items():
            print(f"{m.upper():<15} {r['wa']*100:>7.2f}% {r['f1']*100:>7.2f}%")
        print(f"{'─'*40}")

        winner = max(results, key=lambda m: results[m]['f1'])
        print(f"\n🏆 Лучшая модель по F1: {winner.upper()}")

    # ── Режим одной модели ────────────────────────────────────────
    else:
        output_dir = args.output_dir or f'results/{args.model}'
        os.makedirs(output_dir, exist_ok=True)

        ckpt = args.checkpoint or f'results/{args.model}/best_model.pth'
        model = load_model(args.model, ckpt, device)
        preds, labels, probs = run_inference(model, test_loader, device)

        print_metrics(args.model, preds, labels)
        save_confusion_matrix(preds, labels, args.model, output_dir)

        # Сохраняем предсказания в CSV
        pred_df = pd.DataFrame({
            'true_label':     labels,
            'true_emotion':   [EMOTION_NAMES[l] for l in labels],
            'pred_label':     preds,
            'pred_emotion':   [EMOTION_NAMES[p] for p in preds],
            'prob_happy':     probs[:, 0],
            'prob_sad':       probs[:, 1],
            'prob_anger':     probs[:, 2],
        })
        pred_path = os.path.join(output_dir, 'predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"  Предсказания → {pred_path}")


if __name__ == '__main__':
    main()
