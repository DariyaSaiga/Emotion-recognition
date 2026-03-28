"""
predict.py — инференс и сравнение всех моделей

Использование в Colab:

    BASE=/content/drive/MyDrive/Diploma/checkpoints
    CSV=/content/drive/MyDrive/Diploma/cmu_mosei_final.csv

    # Сравнить ВСЕ обученные модели (автоматически находит готовые чекпоинты)
    !python predict.py --compare_all --ckpt_dir $BASE --csv $CSV

    # Одна конкретная модель
    !python predict.py --model baseline --ckpt $BASE/baseline/best_model.pth --csv $CSV
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

from dataset import get_dataloaders, EMOTION_NAMES, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════

# Порядок и отображаемые имена
MODEL_CONFIG = {
    'text_only':   {'label': 'Text\n(BERT)',        'color': '#5B9BD5', 'group': 'unimodal'},
    'audio_only':  {'label': 'Audio\n(1D CNN)',     'color': '#70AD47', 'group': 'unimodal'},
    'visual_only': {'label': 'Visual\n(BiLSTM)',    'color': '#FFC000', 'group': 'unimodal'},
    'baseline':    {'label': 'Late Fusion\n(Baseline)', 'color': '#ED7D31', 'group': 'multimodal'},
    'bottleneck':  {'label': 'Bottleneck\nTransformer', 'color': '#C00000', 'group': 'multimodal'},
}


# ══════════════════════════════════════════════════════════════════
# АРГУМЕНТЫ
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='Инференс и сравнение моделей')
    parser.add_argument('--csv',         type=str, required=True,
                        help='Путь к cmu_mosei_final.csv')
    parser.add_argument('--batch_size',  type=int, default=32)

    # Режим одной модели
    parser.add_argument('--model',       type=str, default=None,
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument('--ckpt',        type=str, default=None,
                        help='Путь к .pth файлу одной модели')

    # Режим сравнения всех
    parser.add_argument('--compare_all', action='store_true',
                        help='Сравнить все доступные обученные модели')
    parser.add_argument('--ckpt_dir',    type=str,
                        default='/content/drive/MyDrive/Diploma/checkpoints',
                        help='Папка где лежат подпапки с best_model.pth')
    parser.add_argument('--output_dir',  type=str, default=None,
                        help='Куда сохранять графики (по умолчанию = --ckpt_dir)')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛИ
# ══════════════════════════════════════════════════════════════════

def load_model(model_type: str, checkpoint_path: str, device):
    if model_type == 'baseline':
        from baseline import LateFusionBaseline
        model = LateFusionBaseline()
    elif model_type == 'bottleneck':
        from bottleneck import BottleneckTransformer
        model = BottleneckTransformer()
    elif model_type == 'text_only':
        from baseline import TextOnlyModel
        model = TextOnlyModel()
    elif model_type == 'audio_only':
        from baseline import AudioOnlyModel
        model = AudioOnlyModel()
    elif model_type == 'visual_only':
        from baseline import VisualOnlyModel
        model = VisualOnlyModel()
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"  ✓ {model_type:<14} ← {checkpoint_path}")
    return model


# ══════════════════════════════════════════════════════════════════
# ИНФЕРЕНС
# ══════════════════════════════════════════════════════════════════

def run_inference(model, loader, device):
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

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ══════════════════════════════════════════════════════════════════
# ГРАФИКИ И ОТЧЁТЫ
# ══════════════════════════════════════════════════════════════════

def save_confusion_matrix(preds, labels, model_name, output_dir):
    """Матрица ошибок для одной модели."""
    try:
        import seaborn as sns
        use_sns = True
    except ImportError:
        use_sns = False

    emotion_list = [EMOTION_NAMES[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # нормализованная

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',    'Абсолютные значения'),
        (axes[1], cm_norm, '.2f',  'Нормализованная (по строкам)'),
    ]:
        if use_sns:
            import seaborn as sns
            sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                        xticklabels=emotion_list, yticklabels=emotion_list, ax=ax)
        else:
            im = ax.imshow(data, cmap='Blues')
            for i in range(len(emotion_list)):
                for j in range(len(emotion_list)):
                    ax.text(j, i, format(data[i, j], fmt),
                            ha='center', va='center', fontsize=12)
            ax.set_xticks(range(len(emotion_list)))
            ax.set_xticklabels(emotion_list)
            ax.set_yticks(range(len(emotion_list)))
            ax.set_yticklabels(emotion_list)
        ax.set_xlabel('Предсказано')
        ax.set_ylabel('Истинное')
        ax.set_title(title)

    fig.suptitle(f'Confusion Matrix — {MODEL_CONFIG[model_name]["label"].replace(chr(10), " ")}',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f'cm_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Confusion matrix → {path}")


def save_comparison_chart(results: dict, output_dir: str):
    """
    Итоговый сравнительный график всех моделей для диплома.
    Горизонтальные бары, сгруппированы по Unimodal / Multimodal.
    """
    models = [m for m in MODEL_CONFIG if m in results]
    labels = [MODEL_CONFIG[m]['label'] for m in models]
    colors = [MODEL_CONFIG[m]['color'] for m in models]
    wa_vals = [results[m]['wa'] * 100 for m in models]
    f1_vals = [results[m]['f1'] * 100 for m in models]

    y = np.arange(len(models))
    h = 0.35

    fig, ax = plt.subplots(figsize=(10, max(5, len(models) * 1.2)))

    bars_wa = ax.barh(y + h/2, wa_vals, h, label='Weighted Accuracy',
                      color=colors, alpha=0.85)
    bars_f1 = ax.barh(y - h/2, f1_vals, h, label='Weighted F1',
                      color=colors, alpha=0.50, hatch='//')

    # Подписи значений
    for bar in list(bars_wa) + list(bars_f1):
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%', va='center', ha='left', fontsize=9)

    # Разделитель unimodal / multimodal
    unimodal_count = sum(1 for m in models if MODEL_CONFIG[m]['group'] == 'unimodal')
    if unimodal_count > 0 and unimodal_count < len(models):
        ax.axhline(y=unimodal_count - 0.5, color='gray',
                   linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(ax.get_xlim()[1] * 0.02 if ax.get_xlim()[1] > 0 else 1,
                unimodal_count - 0.5 + 0.1,
                'Мультимодальные ↑', fontsize=8, color='gray')
        ax.text(ax.get_xlim()[1] * 0.02 if ax.get_xlim()[1] > 0 else 1,
                unimodal_count - 0.5 - 0.4,
                'Одномодальные ↓', fontsize=8, color='gray')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Score (%)', fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_title('Сравнение моделей — CMU-MOSEI\n(3 класса: happy / sad / anger)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()  # первая модель сверху
    plt.tight_layout()

    path = os.path.join(output_dir, 'comparison_all_models.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  График сравнения → {path}")


def save_per_class_chart(results: dict, output_dir: str):
    """График F1 по каждому классу для всех моделей."""
    models = [m for m in MODEL_CONFIG if m in results and 'per_class_f1' in results[m]]
    if not models:
        return

    emotions = [EMOTION_NAMES[i] for i in range(NUM_CLASSES)]
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        vals = [results[m]['per_class_f1'][e] for e in emotions]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, [v * 100 for v in vals], width,
                      label=MODEL_CONFIG[m]['label'].replace('\n', ' '),
                      color=MODEL_CONFIG[m]['color'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions], fontsize=12)
    ax.set_ylabel('F1-score (%)')
    ax.set_title('F1 по классам эмоций', fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'comparison_per_class_f1.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  F1 по классам      → {path}")


def print_final_table(results: dict):
    """Красивая итоговая таблица в консоль."""
    models = [m for m in MODEL_CONFIG if m in results]

    print(f"\n{'═'*62}")
    print(f"  ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ — CMU-MOSEI")
    print(f"{'═'*62}")
    print(f"  {'Модель':<22} {'WA':>8}  {'F1':>8}   Тип")
    print(f"  {'─'*56}")

    prev_group = None
    for m in models:
        r   = results[m]
        cfg = MODEL_CONFIG[m]
        # Разделитель между группами
        if prev_group and cfg['group'] != prev_group:
            print(f"  {'─'*56}")
        name  = cfg['label'].replace('\n', ' ')
        group = 'Uni' if cfg['group'] == 'unimodal' else 'Multi'
        marker = ' ◄ ЛУЧШАЯ' if m == max(results, key=lambda x: results[x]['f1']) else ''
        print(f"  {name:<22} {r['wa']*100:>7.2f}%  {r['f1']*100:>7.2f}%   {group}{marker}")
        prev_group = cfg['group']

    print(f"{'═'*62}")

    # Прирост Bottleneck над Baseline
    if 'baseline' in results and 'bottleneck' in results:
        delta_wa = (results['bottleneck']['wa'] - results['baseline']['wa']) * 100
        delta_f1 = (results['bottleneck']['f1'] - results['baseline']['f1']) * 100
        sign_wa  = '+' if delta_wa >= 0 else ''
        sign_f1  = '+' if delta_f1 >= 0 else ''
        print(f"\n  Bottleneck vs Late Fusion:")
        print(f"    WA:  {sign_wa}{delta_wa:.2f}%  |  F1:  {sign_f1}{delta_f1:.2f}%")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}\n")

    # Загружаем данные (только test)
    _, _, test_loader = get_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # ── Режим: сравнить ВСЕ доступные модели ─────────────────────
    if args.compare_all:
        output_dir = args.output_dir or args.ckpt_dir
        os.makedirs(output_dir, exist_ok=True)

        # Автоматически ищем готовые чекпоинты
        available = {}
        for m in MODEL_CONFIG:
            ckpt = os.path.join(args.ckpt_dir, m, 'best_model.pth')
            if os.path.exists(ckpt):
                available[m] = ckpt

        if not available:
            print(f"❌ Чекпоинты не найдены в: {args.ckpt_dir}")
            print("   Убедись что папки называются: baseline/, bottleneck/, text_only/ и т.д.")
            return

        print(f"Найдено моделей: {len(available)}")
        for m, p in available.items():
            print(f"  {m}: {p}")

        results = {}
        print("\nЗапускаем инференс...")
        for model_type, ckpt_path in available.items():
            model = load_model(model_type, ckpt_path, device)
            preds, labels, _ = run_inference(model, test_loader, device)

            wa = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)

            # F1 по каждому классу
            f1_per = f1_score(labels, preds, average=None, zero_division=0)
            per_class = {EMOTION_NAMES[i]: f1_per[i] for i in range(NUM_CLASSES)}

            results[model_type] = {'wa': wa, 'f1': f1, 'per_class_f1': per_class,
                                   'preds': preds, 'labels': labels}

            # Confusion matrix для каждой модели
            save_confusion_matrix(preds, labels, model_type, output_dir)

        # Итоговые графики и таблица
        save_comparison_chart(results, output_dir)
        save_per_class_chart(results, output_dir)
        print_final_table(results)

        # Сохраняем сводную таблицу в CSV
        rows = []
        for m, r in results.items():
            row = {'model': m, 'WA': round(r['wa']*100, 2), 'F1': round(r['f1']*100, 2)}
            row.update({f'F1_{k}': round(v*100, 2) for k, v in r['per_class_f1'].items()})
            rows.append(row)
        summary_path = os.path.join(output_dir, 'results_summary.csv')
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\n  Сводная таблица    → {summary_path}")

    # ── Режим: одна модель ────────────────────────────────────────
    else:
        if not args.model:
            print("Укажи --model или используй --compare_all")
            return

        output_dir = args.output_dir or os.path.join(args.ckpt_dir, args.model)
        os.makedirs(output_dir, exist_ok=True)

        ckpt = args.ckpt or os.path.join(args.ckpt_dir, args.model, 'best_model.pth')
        model = load_model(args.model, ckpt, device)
        preds, labels, probs = run_inference(model, test_loader, device)

        wa = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        emotion_list = [EMOTION_NAMES[i] for i in range(NUM_CLASSES)]

        print(f"\n{'═'*50}")
        print(f"  {args.model.upper()} — Результаты на тесте")
        print(f"{'═'*50}")
        print(f"  Weighted Accuracy (WA): {wa:.4f} ({wa*100:.2f}%)")
        print(f"  Weighted F1-score     : {f1:.4f} ({f1*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(labels, preds,
                                    target_names=emotion_list, zero_division=0))

        save_confusion_matrix(preds, labels, args.model, output_dir)

        pred_df = pd.DataFrame({
            'true_label':   labels,
            'true_emotion': [EMOTION_NAMES[l] for l in labels],
            'pred_label':   preds,
            'pred_emotion': [EMOTION_NAMES[p] for p in preds],
            'prob_happy':   probs[:, 0],
            'prob_sad':     probs[:, 1],
            'prob_anger':   probs[:, 2],
        })
        pred_path = os.path.join(output_dir, f'predictions_{args.model}.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"  Предсказания → {pred_path}")


if __name__ == '__main__':
    main()
