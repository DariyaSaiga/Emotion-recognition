import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders
from models  import LateFusionBaseline, BottleneckModel


def accuracy(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()


def per_class_acc(logits, labels):
    preds  = logits.argmax(-1)
    names  = {0: 'happy', 1: 'sad', 2: 'anger'}
    result = {}
    for c in range(3):
        m = (labels == c)
        result[names[c]] = (preds[m] == labels[m]).float().mean().item() \
                           if m.sum() > 0 else 0.0
    return result


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for audio, visual, text, labels, mask in loader:
            audio  = audio.to(device)
            visual = visual.to(device)
            text   = text.to(device)
            labels = labels.to(device)
            mask   = mask.to(device)

            logits = model(audio, visual, text, mask)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return (total_loss / len(loader),
            accuracy(all_logits, all_labels),
            per_class_acc(all_logits, all_labels))


def save_plots(history, name, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'],   'r-', label='Val')
    ax1.set_title(f'{name} — Loss')
    ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid()

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'],   'r-', label='Val')
    ax2.axhline(0.70, color='g', linestyle='--', alpha=0.7, label='70%')
    ax2.set_title(f'{name} — Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid()

    plt.tight_layout()
    path = os.path.join(save_dir, f'{name}_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График: {path}")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    save_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        args.pkl_path, args.batch_size)

    model = LateFusionBaseline(args.dropout) if args.model == 'baseline' \
            else BottleneckModel(args.dropout)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel: {args.model} | Params: {params:,}')
    print(f'Epochs: {args.epochs} | Batch: {args.batch_size} | '
          f'LR: {args.lr} | Dropout: {args.dropout}')

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val, patience_cnt = 0.0, 0
    history = defaultdict(list)

    print('\n' + '=' * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7}")
    print('-' * 60)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_pc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True)
        vl_loss, vl_acc, vl_pc = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False)

        scheduler.step(vl_loss)
        lr_now = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.1%} | "
              f"{vl_loss:>8.4f} | {vl_acc:>6.1%}  lr={lr_now:.0e}")

        if epoch % 5 == 0:
            print(f"       Train: happy={tr_pc['happy']:.1%} "
                  f"sad={tr_pc['sad']:.1%} anger={tr_pc['anger']:.1%}")
            print(f"       Val:   happy={vl_pc['happy']:.1%} "
                  f"sad={vl_pc['sad']:.1%} anger={vl_pc['anger']:.1%}")

        if vl_acc > best_val:
            best_val, patience_cnt = vl_acc, 0
            best_path = os.path.join(save_dir, f'{args.model}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_acc': vl_acc
            }, best_path)
            print(f"       ★ Best val: {vl_acc:.1%} saved")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break

    print('=' * 60)

    # Финальный тест
    print('\nTest evaluation...')
    model.load_state_dict(
        torch.load(best_path, map_location=device)['model_state'])
    _, ts_acc, ts_pc = run_epoch(
        model, test_loader, optimizer, criterion, device, train=False)

    print(f'\n{"="*45}')
    print(f'RESULTS: {args.model.upper()}')
    print(f'{"="*45}')
    print(f'  Val  Accuracy : {best_val:.1%}')
    print(f'  Test Accuracy : {ts_acc:.1%}')
    print(f'  happy: {ts_pc["happy"]:.1%} | '
          f'sad: {ts_pc["sad"]:.1%} | '
          f'anger: {ts_pc["anger"]:.1%}')
    print(f'{"="*45}')

    save_plots(history, args.model, save_dir)

    with open(os.path.join(save_dir, f'{args.model}_metrics.pkl'), 'wb') as f:
        pickle.dump({
            'model': args.model,
            'best_val': best_val,
            'test_acc': ts_acc,
            'test_pc':  ts_pc,
            'history':  dict(history)
        }, f)

    print(f'Saved: {save_dir}')
    return ts_acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      default='bottleneck',
                   choices=['baseline', 'bottleneck'])
    p.add_argument('--pkl_path',   default='/content/drive/MyDrive/Diploma2/mosei_full.pkl')
    p.add_argument('--output_dir', default='/content/drive/MyDrive/Diploma2/results_full')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--dropout',    type=float, default=0.3)
    p.add_argument('--patience',   type=int,   default=10)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
