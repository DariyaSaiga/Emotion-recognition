import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
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

    def forward(self, x):
        x = self.encoder(x.transpose(1, 2))
        return self.fc(self.pool(x).squeeze(-1))


class VisualEncoder(nn.Module):
    def __init__(self, input_dim=713, proj_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(proj_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc   = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, x, mask=None):
        _, (h, _) = self.lstm(self.proj(x))
        return self.fc(torch.cat([h[-2], h[-1]], dim=-1))


class TextEncoder(nn.Module):
    def __init__(self, bert_dim=768, output_dim=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.proj(x.mean(dim=1))


class LateFusionBaseline(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.audio_enc  = AudioEncoder(dropout=dropout)
        self.visual_enc = VisualEncoder(dropout=dropout)
        self.text_enc   = TextEncoder(dropout=dropout)

        self.audio_cls  = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(64, 3))
        self.visual_cls = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(64, 3))
        self.text_cls   = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(64, 3))

    def forward(self, audio, visual, text, mask=None):
        return (self.audio_cls(self.audio_enc(audio)) +
                self.visual_cls(self.visual_enc(visual, mask)) +
                self.text_cls(self.text_enc(text))) / 3.0


class BottleneckFusion(nn.Module):
    def __init__(self, text_dim=256, audio_dim=128, visual_dim=128,
                 bottleneck_dim=128, num_tokens=8, num_heads=8, dropout=0.3):
        super().__init__()
        self.proj_text   = nn.Linear(text_dim,   bottleneck_dim)
        self.proj_audio  = nn.Linear(audio_dim,  bottleneck_dim)
        self.proj_visual = nn.Linear(visual_dim, bottleneck_dim)

        self.tokens = nn.Parameter(torch.randn(num_tokens, bottleneck_dim) * 0.02)

        self.attn_text   = nn.MultiheadAttention(bottleneck_dim, num_heads,
                                                  dropout=dropout, batch_first=True)
        self.attn_audio  = nn.MultiheadAttention(bottleneck_dim, num_heads,
                                                  dropout=dropout, batch_first=True)
        self.attn_visual = nn.MultiheadAttention(bottleneck_dim, num_heads,
                                                  dropout=dropout, batch_first=True)

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

    def forward(self, text, audio, visual):
        B = text.size(0)
        t = self.proj_text(text).unsqueeze(1)
        a = self.proj_audio(audio).unsqueeze(1)
        v = self.proj_visual(visual).unsqueeze(1)

        tok = self.tokens.unsqueeze(0).expand(B, -1, -1)

        out, _ = self.attn_text(tok, t, t)
        tok = self.norm1(tok + self.drop(out))
        out, _ = self.attn_audio(tok, a, a)
        tok = self.norm2(tok + self.drop(out))
        out, _ = self.attn_visual(tok, v, v)
        tok = self.norm3(tok + self.drop(out))
        tok = self.norm4(tok + self.drop(self.ff(tok)))

        return tok.mean(dim=1)


class BottleneckModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.audio_enc  = AudioEncoder(dropout=dropout)
        self.visual_enc = VisualEncoder(dropout=dropout)
        self.text_enc   = TextEncoder(dropout=dropout)
        self.fusion     = BottleneckFusion(
            bottleneck_dim=64, num_tokens=4, num_heads=4, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 3)
        )

    def forward(self, audio, visual, text, mask=None):
        af = self.audio_enc(audio)
        vf = self.visual_enc(visual, mask)
        tf = self.text_enc(text)
        return self.classifier(self.fusion(tf, af, vf))


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