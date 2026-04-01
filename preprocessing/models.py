import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """1D-CNN для аудио. Вход: [B, 50, 74]"""

    def __init__(self, input_dim=74, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, return_seq=False):
        # x: [B, 50, 74]
        x = self.encoder(x.transpose(1, 2))  # [B, 128, 50]
        if return_seq:
            return x.transpose(1, 2)          # [B, 50, 128] — последовательность
        return self.fc(self.pool(x).squeeze(-1))  # [B, 128] — один вектор


class VisualEncoder(nn.Module):
    """BiLSTM для видео. Вход: [B, 50, 713]"""

    def __init__(self, input_dim=713, proj_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(proj_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc   = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.seq_fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, mask=None, return_seq=False):
        # x: [B, 50, 713]
        out, (h, _) = self.lstm(self.proj(x))
        if return_seq:
            return self.seq_fc(out)             # [B, 50, 128] — все шаги
        return self.fc(torch.cat([h[-2], h[-1]], dim=-1))  # [B, 128] — финал


class TextEncoder(nn.Module):
    """Проекция BERT эмбеддингов. Вход: [B, 50, 768]"""

    def __init__(self, bert_dim=768, output_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x, return_seq=False):
        # x: [B, 50, 768]
        projected = self.proj(x)               # [B, 50, 128]
        if return_seq:
            return projected                   # [B, 50, 128] — вся последовательность
        return projected.mean(dim=1)           # [B, 128] — среднее



class LateFusionBaseline(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.audio_enc  = AudioEncoder(dropout=dropout)
        self.visual_enc = VisualEncoder(dropout=dropout)
        self.text_enc   = TextEncoder(output_dim=128, dropout=dropout)

        # Три независимых классификатора
        self.audio_cls  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3)
        )
        self.visual_cls = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3)
        )
        self.text_cls   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3)
        )

    def forward(self, audio, visual, text, mask=None):
        af = self.audio_enc(audio)
        vf = self.visual_enc(visual)
        tf = self.text_enc(text)
        return (self.audio_cls(af) +
                self.visual_cls(vf) +
                self.text_cls(tf)) / 3.0



class BottleneckFusion(nn.Module):
    def __init__(self, seq_dim=128, bottleneck_dim=64,
                 num_tokens=4, num_heads=4, dropout=0.3):
        super().__init__()

        # Обучаемые latent tokens — посредники между модальностями
        self.tokens = nn.Parameter(
            torch.randn(num_tokens, bottleneck_dim) * 0.02
        )

        # Проекции модальностей в bottleneck пространство
        self.proj_audio  = nn.Linear(seq_dim, bottleneck_dim)
        self.proj_visual = nn.Linear(seq_dim, bottleneck_dim)
        self.proj_text   = nn.Linear(seq_dim, bottleneck_dim)

        # Cross-Attention: токены смотрят на каждую модальность
        # Query = latent tokens
        # Key = Value = последовательность модальности [50 шагов]
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

    def forward(self, audio_seq, visual_seq, text_seq):
        # audio_seq:  [B, 50, 128]
        # visual_seq: [B, 50, 128]
        # text_seq:   [B, 50, 128]
        B = audio_seq.size(0)

        # Проецируем в bottleneck пространство
        a = self.proj_audio(audio_seq)   # [B, 50, 64]
        v = self.proj_visual(visual_seq) # [B, 50, 64]
        t = self.proj_text(text_seq)     # [B, 50, 64]

        # Расширяем latent tokens на батч: [4, 64] → [B, 4, 64]
        tok = self.tokens.unsqueeze(0).expand(B, -1, -1)

        # Токены смотрят на ВСЕ 50 шагов каждой модальности
        out, _ = self.attn_audio(tok, a, a)
        tok = self.norm1(tok + self.drop(out))

        out, _ = self.attn_visual(tok, v, v)
        tok = self.norm2(tok + self.drop(out))

        out, _ = self.attn_text(tok, t, t)
        tok = self.norm3(tok + self.drop(out))

        tok = self.norm4(tok + self.drop(self.ff(tok)))

        # Усредняем по токенам → финальный вектор
        return tok.mean(dim=1)  # [B, 64]


class BottleneckModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.audio_enc  = AudioEncoder(dropout=dropout)
        self.visual_enc = VisualEncoder(dropout=dropout)
        self.text_enc   = TextEncoder(output_dim=128, dropout=dropout)

        self.fusion = BottleneckFusion(
            seq_dim=128,
            bottleneck_dim=64,
            num_tokens=4,
            num_heads=4,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 3)
        )

    def forward(self, audio, visual, text, mask=None):
        # Получаем ПОЛНЫЕ последовательности от энкодеров
        af = self.audio_enc(audio,  return_seq=True)   # [B, 50, 128]
        vf = self.visual_enc(visual, return_seq=True)  # [B, 50, 128]
        tf = self.text_enc(text,    return_seq=True)   # [B, 50, 128]

        fused = self.fusion(af, vf, tf)                # [B, 64]
        return self.classifier(fused)                   # [B, 3]


# ============================================================
# Проверка
# ============================================================

if __name__ == '__main__':
    B, S = 4, 50
    audio  = torch.randn(B, S, 74)
    visual = torch.randn(B, S, 713)
    text   = torch.randn(B, S, 768)

    for name, model in [('Baseline',   LateFusionBaseline()),
                         ('Bottleneck', BottleneckModel())]:
        out = model(audio, visual, text)
        p   = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(f'{name}: выход={out.shape}, параметров={p:,}')
        assert out.shape == (B, 3), 'ОШИБКА!'
    print('OK')