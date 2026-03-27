"""
baseline.py — Late Fusion Baseline для CMU-MOSEI

Архитектура:
    Text   (BERT → [CLS] 768-dim)  → Linear → 256-dim → ReLU → Dropout
    Audio  (COVAREP 74-dim)         →  MLP   → 256-dim → ReLU → Dropout
    Visual (Facet42 35-dim)         →  MLP   → 256-dim → ReLU → Dropout
                                          ↓
                              Concat → 768-dim
                                          ↓
                              FC Classifier → 3 класса

Сравнение с Bottleneck Transformer (bottleneck.py):
    Baseline: простое конкатенирование, нет взаимодействия между модальностями
    Bottleneck: информация проходит через общие bottleneck-токены
"""

import torch
import torch.nn as nn
from transformers import BertModel

from dataset import AUDIO_DIM, VISUAL_DIM, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════
# ЭНКОДЕРЫ
# ══════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    BERT → [CLS] → Linear → 256-dim.
    Веса BERT заморожены (frozen) — быстрее обучение.
    """

    def __init__(self, output_dim: int = 256,
                 bert_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)

        # Замораживаем — не обучаем BERT в baseline
        for param in self.bert.parameters():
            param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.proj(cls)                   # (B, 256)


class AudioMLPEncoder(nn.Module):
    """
    MLP на pre-extracted COVAREP аудио признаках (74-dim).
    74 → 128 → 256
    """

    def __init__(self, input_dim: int = AUDIO_DIM,
                 output_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """x: (B, 74)"""
        return self.mlp(x)   # (B, 256)


class VisualMLPEncoder(nn.Module):
    """
    MLP на pre-extracted Facet42 визуальных признаках (35-dim).
    35 → 128 → 256
    """

    def __init__(self, input_dim: int = VISUAL_DIM,
                 output_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """x: (B, 35)"""
        return self.mlp(x)   # (B, 256)


# ══════════════════════════════════════════════════════════════════
# LATE FUSION BASELINE
# ══════════════════════════════════════════════════════════════════

class LateFusionBaseline(nn.Module):
    """
    Baseline модель: поздняя фьюжн (Late Fusion).

    Каждая модальность кодируется независимо,
    затем признаки конкатенируются и классифицируются.

    Параметры:
        num_classes : кол-во классов (3: happy/sad/anger)
        feat_dim    : размерность каждого энкодера (256)
        bert_name   : BERT модель
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 feat_dim: int = 256,
                 bert_name: str = 'bert-base-uncased'):
        super().__init__()

        self.text_encoder   = TextEncoder(output_dim=feat_dim, bert_name=bert_name)
        self.audio_encoder  = AudioMLPEncoder(output_dim=feat_dim)
        self.visual_encoder = VisualMLPEncoder(output_dim=feat_dim)

        fusion_dim = feat_dim * 3   # 768 = 256 + 256 + 256

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, text_inputs, audio_feat, visual_feat):
        """
        text_inputs : dict {input_ids: (B,L), attention_mask: (B,L)}
        audio_feat  : (B, 74)
        visual_feat : (B, 35)
        Returns:
            logits : (B, num_classes)
        """
        text_feat   = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )                                          # (B, 256)
        audio_feat  = self.audio_encoder(audio_feat)   # (B, 256)
        visual_feat = self.visual_encoder(visual_feat) # (B, 256)

        fused  = torch.cat([text_feat, audio_feat, visual_feat], dim=1)  # (B, 768)
        logits = self.classifier(fused)                                   # (B, 3)
        return logits


# ══════════════════════════════════════════════════════════════════
# БЫСТРАЯ ПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = LateFusionBaseline().to(device)
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметры всего:     {total:,}")
    print(f"Параметры обучаемых: {trainable:,}")

    # Тестовый прогон
    B = 4
    text_inputs = {
        'input_ids':      torch.randint(0, 1000, (B, 128)).to(device),
        'attention_mask': torch.ones(B, 128, dtype=torch.long).to(device),
    }
    audio  = torch.randn(B, AUDIO_DIM).to(device)
    visual = torch.randn(B, VISUAL_DIM).to(device)

    logits = model(text_inputs, audio, visual)
    print(f"Logits shape: {logits.shape}")   # (4, 3)
    print("✓ LateFusionBaseline работает корректно!")
