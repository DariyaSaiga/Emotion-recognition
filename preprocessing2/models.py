import torch
import torch.nn as nn


# ============================================================
# BASELINE — простой Late Fusion (намеренно простой)
# mean pooling + linear на каждую модальность
# ============================================================

class LateFusionBaseline(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.audio_proj  = nn.Sequential(
            nn.Linear(74,  128), nn.ReLU(), nn.Dropout(dropout))
        self.visual_proj = nn.Sequential(
            nn.Linear(713, 128), nn.ReLU(), nn.Dropout(dropout))
        self.text_proj   = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(dropout))

        self.audio_cls  = nn.Linear(128, 3)
        self.visual_cls = nn.Linear(128, 3)
        self.text_cls   = nn.Linear(128, 3)

    def forward(self, audio, visual, text, mask=None):
        # audio:  [B, 50, 74]  → mean → [B, 74]
        # visual: [B, 50, 713] → mean → [B, 713]
        # text:   [B, 768]
        af = self.audio_proj(audio.mean(dim=1))
        vf = self.visual_proj(visual.mean(dim=1))
        tf = self.text_proj(text)
        return (self.audio_cls(af) +
                self.visual_cls(vf) +
                self.text_cls(tf)) / 3.0


# ============================================================
# ENCODERS для Bottleneck
# Аудио и визуал → полные последовательности [B, 50, 128]
# Текст → один вектор [B, 256] через Linear
# ============================================================

class AudioCNN(nn.Module):
    """1D-CNN. [B, 50, 74] → [B, 50, 128]"""
    def __init__(self, input_dim=74, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)  # [B, 50, 128]


class VisualBiLSTM(nn.Module):
    """BiLSTM. [B, 50, 713] → [B, 50, 128]"""
    def __init__(self, input_dim=713, proj_dim=256, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x):
        out, _ = self.lstm(self.proj(x))
        return self.fc(out)  # [B, 50, 128]


class TextEncoder(nn.Module):
    """BERT CLS вектор → [B, 128]"""
    def __init__(self, bert_dim=768, output_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, output_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)  # [B, 128]


# ============================================================
# BOTTLENECK ATTENTION FUSION
# Latent tokens смотрят на:
#   - полные последовательности аудио [B, 50, 128]
#   - полные последовательности визуала [B, 50, 128]
#   - текстовый вектор [B, 128] (добавляется как один шаг)
# 16 токенов как в статье DBA
# ============================================================

class BottleneckFusion(nn.Module):
    def __init__(self, seq_dim=128, bottleneck_dim=64,
                 num_tokens=16, num_heads=4, dropout=0.3):
        super().__init__()

        # 16 обучаемых latent tokens
        self.tokens = nn.Parameter(
            torch.randn(num_tokens, bottleneck_dim) * 0.02)

        self.proj_audio  = nn.Linear(seq_dim, bottleneck_dim)
        self.proj_visual = nn.Linear(seq_dim, bottleneck_dim)
        self.proj_text   = nn.Linear(seq_dim, bottleneck_dim)

        self.attn_audio  = nn.MultiheadAttention(
            bottleneck_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_visual = nn.MultiheadAttention(
            bottleneck_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_text   = nn.MultiheadAttention(
            bottleneck_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.norm2 = nn.LayerNorm(bottleneck_dim)
        self.norm3 = nn.LayerNorm(bottleneck_dim)
        self.norm4 = nn.LayerNorm(bottleneck_dim)

        self.ff = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 4),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 4, bottleneck_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, audio_seq, visual_seq, text_vec):
        # audio_seq:  [B, 50, 128]
        # visual_seq: [B, 50, 128]
        # text_vec:   [B, 128] → добавляем ось → [B, 1, 128]
        B = audio_seq.size(0)

        a = self.proj_audio(audio_seq)              # [B, 50, 64]
        v = self.proj_visual(visual_seq)            # [B, 50, 64]
        t = self.proj_text(text_vec.unsqueeze(1))   # [B, 1, 64]

        tok = self.tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 16, 64]

        out, _ = self.attn_audio(tok, a, a)
        tok = self.norm1(tok + self.drop(out))

        out, _ = self.attn_visual(tok, v, v)
        tok = self.norm2(tok + self.drop(out))

        out, _ = self.attn_text(tok, t, t)
        tok = self.norm3(tok + self.drop(out))

        tok = self.norm4(tok + self.drop(self.ff(tok)))

        return tok.mean(dim=1)  # [B, 64]


# ============================================================
# BOTTLENECK MODEL
# ============================================================

class BottleneckModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.audio_enc  = AudioCNN(dropout=dropout)
        self.visual_enc = VisualBiLSTM(dropout=dropout)
        self.text_enc   = TextEncoder(dropout=dropout)
        self.fusion     = BottleneckFusion(
            seq_dim=128, bottleneck_dim=64,
            num_tokens=16, num_heads=4, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 3)
        )

    def forward(self, audio, visual, text, mask=None):
        af = self.audio_enc(audio)    # [B, 50, 128]
        vf = self.visual_enc(visual)  # [B, 50, 128]
        tf = self.text_enc(text)      # [B, 128]
        return self.classifier(self.fusion(af, vf, tf))


# ============================================================
# Проверка
# ============================================================

if __name__ == '__main__':
    B = 4
    audio  = torch.randn(B, 50, 74)
    visual = torch.randn(B, 50, 713)
    text   = torch.randn(B, 768)   # теперь [B, 768] — не последовательность

    for name, model in [('Baseline',   LateFusionBaseline()),
                         ('Bottleneck', BottleneckModel())]:
        out = model(audio, visual, text)
        p   = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(f'{name}: выход={out.shape}, параметров={p:,}')
        assert out.shape == (B, 3)
    print('OK')
