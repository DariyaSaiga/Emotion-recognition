import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


# ══════════════════════════════════════════════════════
# 1.  ТВОЙ ЭНКОДЕР — CNN для аудио      (Ablanova)
# ══════════════════════════════════════════════════════

class AudioCNNEncoder(nn.Module):
    """
    CNN энкодер для мел-спектрограмм.

    Вход:  (B, 1, 128, 128)  — мел-спектрограмма
    Выход: (B, 256)          — вектор аудио-признаков

    Архитектура (5 блоков):
        Conv2d(k=3, p=1) → BatchNorm2d → ReLU → MaxPool2d(2,2)
    После 5 блоков: 128 → 4 по пространству, 256 каналов.
    AdaptiveAvgPool → Flatten → Linear(256) → ReLU → Dropout
    """

    def __init__(self, output_dim=256):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Блок 1: (1,128,128) → (32,64,64)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 2: (32,64,64) → (64,32,32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 3: (64,32,32) → (128,16,16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 4: (128,16,16) → (256,8,8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 5: (256,8,8) → (256,4,4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Усредняем по пространству → (B, 256, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),               # (B, 256)
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """x: (B, 1, 128, 128)"""
        x = self.conv_blocks(x)   # (B, 256, 4, 4)
        x = self.pool(x)          # (B, 256, 1, 1)
        x = self.head(x)          # (B, 256)
        return x


# ══════════════════════════════════════════════════════
# 2.  ЭНКОДЕР ВТОРОГО УЧАСТНИКА — BERT  (Alpieva)
# ══════════════════════════════════════════════════════

class TextBERTEncoder(nn.Module):
    """
    BERT энкодер для текста.

    Вход:  список строк ['I feel great', ...]
    Выход: (B, 256)  — сжатый вектор текста

    Используем bert-base-uncased pretrained, веса заморожены (frozen).
    [CLS] токен (768-dim) → Linear → 256-dim.
    """

    def __init__(self, output_dim=256, bert_name='bert-base-uncased'):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.bert = BertModel.from_pretrained(bert_name)

        # Замораживаем BERT — не обучаем его веса
        for param in self.bert.parameters():
            param.requires_grad = False

        # Проекция 768 → output_dim
        self.head = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, texts, device):
        """texts: list[str]"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        with torch.no_grad():
            out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        cls = out.last_hidden_state[:, 0, :]  # (B, 768) — [CLS] токен
        return self.head(cls)                  # (B, 256)


# ══════════════════════════════════════════════════════
# 3.  ЭНКОДЕР ВТОРОГО УЧАСТНИКА — ResNet-18  (Alpieva)
#     ЗАГЛУШКА — используется нулевой вектор,
#     пока нет папки с кадрами лиц.
#     Когда кадры появятся — раскомментируй класс ниже.
# ══════════════════════════════════════════════════════

class VisualResNetEncoder(nn.Module):
    """
    ResNet-18 энкодер для кадров лица.

    Вход:  (B, 3, 224, 224)
    Выход: (B, 256)

    Pretrained ResNet-18, последний FC убран, веса frozen.
    """

    def __init__(self, output_dim=256):
        super().__init__()

        resnet = tv_models.resnet18(pretrained=True)
        # Убираем последний classifier слой
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """x: (B, 3, 224, 224)"""
        with torch.no_grad():
            x = self.backbone(x)   # (B, 512, 1, 1)
        return self.head(x)        # (B, 256)


# ══════════════════════════════════════════════════════
# 4.  BASELINE МОДЕЛЬ — Late Fusion
# ══════════════════════════════════════════════════════

class BaselineLateFusion(nn.Module):
    """
    Baseline: Late Fusion из трёх модальностей.

    Схема:
        Audio  (wav)   → CNN Encoder   → 256-dim
        Text   (str)   → BERT Encoder  → 256-dim
        Visual (frame) → ResNet Encoder → 256-dim  ← пока заглушка
                                         ↓
                              concat → 768-dim
                                         ↓
                              FC Classifier → 7 классов

    Параметры:
        num_classes : кол-во классов эмоций (7)
        feat_dim    : размерность каждого энкодера (256)
        use_visual  : True = используем ResNet, False = нули вместо видео
    """

    EMOTION_NAMES = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear', 'neutral']

    def __init__(self, num_classes=7, feat_dim=256, use_visual=False):
        super().__init__()
        
        self.use_visual = use_visual
        self.feat_dim   = feat_dim

        self.audio_encoder  = AudioCNNEncoder(output_dim=feat_dim)
        self.text_encoder   = TextBERTEncoder(output_dim=feat_dim)
        self.bottleneck_fusion = AttentionBottleneck(
        feat_dim=feat_dim,
        num_tokens=4
)

        # вход в классификатор теперь = 256 (а не 512/768)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        if use_visual:
            self.visual_encoder = VisualResNetEncoder(output_dim=feat_dim)

        # Размерность после concat
        fusion_dim = feat_dim * 3 if use_visual else feat_dim * 2  # 768 или 512

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        

    def forward(self, mel, texts, device, frames=None):
        """
        mel    : (B, 1, 128, 128)
        texts  : list[str]
        device : torch.device
        frames : (B, 3, 224, 224) или None
        """
        audio_feat = self.audio_encoder(mel)           # (B, 256)
        text_feat  = self.text_encoder(texts, device)  # (B, 256)

        # собираем модальности как sequence
        if self.use_visual and frames is not None:
            visual_feat = self.visual_encoder(frames)
            features = torch.stack([audio_feat, text_feat, visual_feat], dim=1)  # (B, 3, 256)
        else:
            features = torch.stack([audio_feat, text_feat], dim=1)  # (B, 2, 256)

        # bottleneck fusion
        fused = self.bottleneck_fusion(features)  # (B, 256)

        # классификация
        logits = self.classifier(fused)
        return logits
    



class AttentionBottleneck(nn.Module):
    """
    Attention Bottleneck для объединения модальностей.
    """

    def __init__(self, feat_dim=256, num_tokens=4, num_heads=4):
        super().__init__()

        self.num_tokens = num_tokens
        self.feat_dim = feat_dim

        # bottleneck токены (обучаемые)
        self.bottleneck = nn.Parameter(
            torch.randn(1, num_tokens, feat_dim)
        )

        # Multihead Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, features):
        """
        features: (B, M, 256)
        M = количество модальностей (2 или 3)
        """

        B = features.size(0)

        # повторяем bottleneck под batch
        bottleneck = self.bottleneck.repeat(B, 1, 1)  # (B, N, 256)

        # attention: bottleneck "смотрит" на модальности
        attended, _ = self.attn(
            query=bottleneck,
            key=features,
            value=features
        )

        out = self.norm(attended + bottleneck)  # residual

        # усредняем bottleneck токены → один вектор
        out = out.mean(dim=1)  # (B, 256)

        return out
