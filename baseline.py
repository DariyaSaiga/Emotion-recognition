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


class AudioCNNEncoder(nn.Module):
    """
    1D CNN энкодер на COVAREP признаках (74-dim).
    
    Вход:  (B, 74)
    Выход: (B, 256)
    
    Архитектура:
        (B, 74) → reshape → (B, 1, 74)   # как 1D сигнал
        → Conv1d(1→32, k=3) → BN → ReLU
        → Conv1d(32→64, k=3) → BN → ReLU
        → Conv1d(64→128, k=3) → BN → ReLU
        → AdaptiveAvgPool → Flatten
        → Linear(128→256) → ReLU → Dropout
    """

    def __init__(self, input_dim: int = 74, output_dim: int = 256):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Блок 1: (B, 1, 74) → (B, 32, 72)
            nn.Conv1d(1, 32, kernel_size=3, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # Блок 2: (B, 32, 72) → (B, 64, 70)
            nn.Conv1d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Блок 3: (B, 64, 70) → (B, 128, 68)
            nn.Conv1d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Усредняем по временному измерению → (B, 128, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),                    # (B, 128)
            nn.Linear(128, output_dim),      # (B, 256)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """x: (B, 74)"""
        x = x.unsqueeze(1)       # (B, 1, 74) — добавляем канал
        x = self.conv_layers(x)  # (B, 128, 68)
        x = self.pool(x)         # (B, 128, 1)
        x = self.head(x)         # (B, 256)
        return x


class VisualBiLSTMEncoder(nn.Module):
    """
    BiLSTM энкодер на Facet42 визуальных признаках (35-dim).

    Вход:  (B, 35) — усреднённые по времени признаки лица
    Выход: (B, 256)

    Так как у нас признаки уже усреднены по времени,
    мы разбиваем вектор на 7 шагов по 5 признаков —
    это позволяет BiLSTM видеть структуру признаков.

    35 → reshape (7, 5) → BiLSTM(5, 64) → последний hidden → Linear → 256
    """

    def __init__(self, input_dim: int = VISUAL_DIM,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = 256):
        super().__init__()

        self.seq_len  = 7                          # разбиваем на 7 шагов
        self.step_dim = input_dim // self.seq_len  # 713 // 7 = 101

        self.bilstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,   # ← это и есть Bi
            dropout=0.3,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),  # *2 потому что bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """x: (B, 713)"""
        B = x.size(0)
        # 713 // 7 = 101, но 7*101 = 707 — обрезаем 6 лишних значений
        x = x[:, :self.seq_len * self.step_dim]   # (B, 707)
        x = x.view(B, self.seq_len, self.step_dim)

        # BiLSTM → out: (B, 7, 128), hidden: (num_layers*2, B, 64)
        _, (hidden, _) = self.bilstm(x)

        # Берём последний слой: forward[-2] и backward[-1]
        forward_h  = hidden[-2]                                  # (B, 64)
        backward_h = hidden[-1]                                  # (B, 64)
        combined   = torch.cat([forward_h, backward_h], dim=1)  # (B, 128)

        return self.head(combined)  # (B, 256)


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
        self.audio_encoder = AudioCNNEncoder(output_dim=feat_dim)
        self.visual_encoder = VisualBiLSTMEncoder(output_dim=feat_dim)

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
# UNIMODAL МОДЕЛИ — для ablation study
# ══════════════════════════════════════════════════════════════════

class TextOnlyModel(nn.Module):
    """Только BERT — текстовая модальность"""
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=256):
        super().__init__()
        self.text_encoder = TextEncoder(output_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, text_inputs, audio_feat, visual_feat):
        feat = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )
        return self.classifier(feat)


class AudioOnlyModel(nn.Module):
    """Только 1D CNN — аудио модальность"""
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=256):
        super().__init__()
        self.audio_encoder = AudioCNNEncoder(output_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, text_inputs, audio_feat, visual_feat):
        feat = self.audio_encoder(audio_feat)
        return self.classifier(feat)


class VisualOnlyModel(nn.Module):
    """Только BiLSTM — визуальная модальность"""
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=256):
        super().__init__()
        self.visual_encoder = VisualBiLSTMEncoder(output_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, text_inputs, audio_feat, visual_feat):
        feat = self.visual_encoder(visual_feat)
        return self.classifier(feat)


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
