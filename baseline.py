import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertModel
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Text Encoder (BERT)
# =========================
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # freeze BERT for baseline
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return cls_embedding  # (B, 768)


# =========================
# Audio Encoder (CNN) — Ablanova
# =========================
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        # x: (B, 1, 64, 128)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return x  # (B, 128)


# =========================
# Visual Encoder (ResNet-18) — Alpieva
# Пока отключён — нет кадров лиц
# =========================
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x  # (B, 512)


# =========================
# Late Fusion Baseline
# Сейчас: Audio + Text (без Visual — нет кадров)
# Потом:  Audio + Text + Visual (добавим когда будут кадры)
# =========================
class LateFusionBaseline(nn.Module):
    def __init__(self, num_classes=7, use_visual=False):
        super().__init__()

        self.use_visual = use_visual
        self.text_encoder  = TextEncoder()
        self.audio_encoder = AudioEncoder()

        if use_visual:
            self.visual_encoder = VisualEncoder()
            fusion_dim = 768 + 128 + 512  # text + audio + visual
        else:
            fusion_dim = 768 + 128         # text + audio только

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_inputs, audio_inputs, visual_inputs=None):
        text_feat  = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"]
        )                                    # (B, 768)
        audio_feat = self.audio_encoder(audio_inputs)  # (B, 128)

        if self.use_visual and visual_inputs is not None:
            visual_feat = self.visual_encoder(visual_inputs)  # (B, 512)
            fused = torch.cat([text_feat, audio_feat, visual_feat], dim=1)
        else:
            fused = torch.cat([text_feat, audio_feat], dim=1)  # (B, 896)

        logits = self.classifier(fused)
        return logits
