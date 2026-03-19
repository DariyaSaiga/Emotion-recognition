import torch
import torch.nn as nn
from transformers import BertModel


class AttentionBottleneckModel(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=256):
        super().__init__()

        # 🔹 TEXT: BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, hidden_dim)

        # 🔹 AUDIO: простой encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, hidden_dim),
            nn.ReLU()
        )

        # 🔹 ATTENTION BOTTLENECK
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # 🔹 CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, text_inputs, mel):

        # ===== TEXT =====
        bert_out = self.bert(**text_inputs)
        h_text = bert_out.last_hidden_state[:, 0, :]   # CLS токен
        h_text = self.text_fc(h_text)

        # ===== AUDIO =====
        mel = mel.unsqueeze(1)  # (B, 1, H, W)
        h_audio = self.audio_encoder(mel)

        # ===== ATTENTION BOTTLENECK =====
        # делаем sequence из 2 токенов (text + audio)
        h_text = h_text.unsqueeze(1)   # (B,1,D)
        h_audio = h_audio.unsqueeze(1) # (B,1,D)

        combined = torch.cat([h_text, h_audio], dim=1)  # (B,2,D)

        attn_out, _ = self.attention(combined, combined, combined)

        # "узкое место" — усредняем
        h = attn_out.mean(dim=1)

        # ===== CLASSIFIER =====
        logits = self.classifier(h)

        return logits