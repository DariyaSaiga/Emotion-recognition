import math
import torch
import torch.nn as nn
from transformers import BertModel

from dataset import AUDIO_DIM, VISUAL_DIM, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════
# 1. ПРОЕКЦИЯ МОДАЛЬНОСТЕЙ В ОБЩЕЕ ПРОСТРАНСТВО
# ══════════════════════════════════════════════════════════════════

class VisualBiLSTMProjector(nn.Module):
    """
    BiLSTM проектор для визуальных Facet42 признаков (35-dim) → d_model.

    Разбиваем вектор (B, 35) на последовательность (B, 7, 5),
    пропускаем через BiLSTM, берём последний hidden state,
    проецируем в d_model и оборачиваем в токен (B, 1, d_model).
    """

    def __init__(self, d_model: int,
                 input_dim: int = VISUAL_DIM,
                 hidden_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()

        self.seq_len  = 7
        self.step_dim = input_dim // self.seq_len  # 713 // 7 = 101

        self.bilstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        # Learnable modality embedding (как в ModalityProjector)
        self.modality_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        """x: (B, 713) → (B, 1, d_model)"""
        B = x.size(0)
        # 713 // 7 = 101, но 7*101 = 707 — обрезаем 6 лишних значений
        x = x[:, :self.seq_len * self.step_dim]     # (B, 707)
        x = x.view(B, self.seq_len, self.step_dim)  # (B, 7, 101)

        _, (hidden, _) = self.bilstm(x)              # hidden: (layers*2, B, 128)
        forward_h  = hidden[-2]                       # (B, 128)
        backward_h = hidden[-1]                       # (B, 128)
        combined   = torch.cat([forward_h, backward_h], dim=1)  # (B, 256)

        out = self.proj(combined).unsqueeze(1)       # (B, 1, d_model)
        out = out + self.modality_emb                # + modality embedding
        return out


class ModalityProjector(nn.Module):
    """
    Проецирует признаки одной модальности в d_model-мерное пространство.
    Добавляет learnable positional/modality embedding.
    """

    def __init__(self, input_dim: int, d_model: int, num_tokens: int = 1):
        """
        input_dim  : размерность входных признаков
        d_model    : размерность модели Transformer
        num_tokens : количество токенов на модальность (1 для pre-extracted feat)
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        # Learnable modality embedding
        self.modality_emb = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    def forward(self, x):
        """x: (B, input_dim) → (B, num_tokens, d_model)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)                     # (B, 1, input_dim)
        x = self.proj(x)                           # (B, num_tokens, d_model)
        x = x + self.modality_emb                  # + modality embedding
        return x


# ══════════════════════════════════════════════════════════════════
# 2. BERT ПРОЕКЦИЯ
# ══════════════════════════════════════════════════════════════════

class TextProjector(nn.Module):
    """
    BERT → несколько первых token-ов (CLS + context) → d_model.
    Берём первые `num_tokens` токенов из BERT для более богатого представления.
    """

    def __init__(self, d_model: int, num_tokens: int = 4,
                 bert_name: str = 'bert-base-uncased'):
        super().__init__()
        self.num_tokens = num_tokens
        self.bert = BertModel.from_pretrained(bert_name)

        # Замораживаем BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.modality_emb = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        # Берём первые num_tokens токенов
        tokens = out.last_hidden_state[:, :self.num_tokens, :]  # (B, T, 768)
        tokens = self.proj(tokens)                              # (B, T, d_model)
        tokens = tokens + self.modality_emb                    # + modality emb
        return tokens


# ══════════════════════════════════════════════════════════════════
# 3. BOTTLENECK FUSION LAYER
# ══════════════════════════════════════════════════════════════════

class BottleneckFusionLayer(nn.Module):
    """
    Один слой Bottleneck Fusion.

    Схема для одного слоя:
        Для каждой модальности M:
            [tokens_M | bottleneck] → MultiHeadAttention → update tokens_M
        Bottleneck = среднее обновлений от всех модальностей

    Параметры:
        d_model     : размерность
        nhead       : количество голов attention
        num_modalities : количество модальностей (3)
        dropout     : dropout
    """

    def __init__(self, d_model: int, nhead: int = 4,
                 num_modalities: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities

        # Отдельный Self-Attention для каждой модальности
        # (каждая модальность видит свои токены + bottleneck)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_modalities)
        ])

        # Feed-Forward для каждой модальности
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_modalities)
        ])

        # LayerNorm
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model)
                                    for _ in range(num_modalities)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model)
                                    for _ in range(num_modalities)])
        self.norm_bottleneck = nn.LayerNorm(d_model)

    def forward(self, modality_tokens: list, bottleneck: torch.Tensor):
        """
        modality_tokens : list of (B, T_i, d_model)  — токены каждой модальности
        bottleneck      : (B, B_n, d_model)           — bottleneck токены

        Returns:
            updated_tokens     : list of (B, T_i, d_model)
            updated_bottleneck : (B, B_n, d_model)
        """
        updated_tokens = []
        bottleneck_updates = []

        for i, (tokens, attn, ffn, n1, n2) in enumerate(zip(
                modality_tokens,
                self.attn_layers,
                self.ffn_layers,
                self.norm1,
                self.norm2)):

            # Конкатенируем токены модальности с bottleneck
            # [tokens_i | bottleneck]: (B, T_i + B_n, d_model)
            combined = torch.cat([tokens, bottleneck], dim=1)

            # Self-Attention: каждый токен attention к tokens_i + bottleneck
            attn_out, _ = attn(combined, combined, combined)

            # Обновляем только токены модальности (не bottleneck)
            T_i = tokens.size(1)
            tokens_updated  = n1(tokens + attn_out[:, :T_i, :])
            tokens_updated  = n2(tokens_updated + ffn(tokens_updated))

            # Bottleneck часть из этой модальности
            bn_contribution = attn_out[:, T_i:, :]
            bottleneck_updates.append(bn_contribution)
            updated_tokens.append(tokens_updated)

        # Обновляем bottleneck = среднее вкладов от всех модальностей
        bn_stack = torch.stack(bottleneck_updates, dim=0)   # (M, B, B_n, d)
        bn_mean  = bn_stack.mean(dim=0)                     # (B, B_n, d)
        bottleneck_new = self.norm_bottleneck(bottleneck + bn_mean)

        return updated_tokens, bottleneck_new

class AudioCNNProjector(nn.Module):
    """
    1D CNN проектор для COVAREP аудио признаков (74-dim) → d_model.
    Тот же CNN что в baseline, но выход адаптирован для Bottleneck.
    """
    def __init__(self, input_dim: int = 74, d_model: int = 128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        """x: (B, 74) → (B, 1, d_model)"""
        x = x.unsqueeze(1)        # (B, 1, 74)
        x = self.conv_layers(x)   # (B, 128, 68)
        x = self.pool(x)          # (B, 128, 1)
        x = self.head(x)          # (B, d_model)
        return x.unsqueeze(1)     # (B, 1, d_model) — один токен
    
# ══════════════════════════════════════════════════════════════════
# 4. BOTTLENECK TRANSFORMER — ГЛАВНАЯ МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════

class BottleneckTransformer(nn.Module):
    """
    Bottleneck Transformer для распознавания эмоций из 3 модальностей.

    Параметры:
        num_classes      : 3 (happy / sad / anger)
        d_model          : размерность внутреннего пространства (128)
        num_layers       : количество Bottleneck Fusion слоёв (4)
        nhead            : количество голов attention (4)
        num_bottlenecks  : количество bottleneck токенов (4)
        text_tokens      : количество BERT токенов для текста (4)
        dropout          : dropout
        bert_name        : BERT модель
    """

    def __init__(self,
                 num_classes: int     = NUM_CLASSES,
                 d_model: int         = 128,
                 num_layers: int      = 4,
                 nhead: int           = 4,
                 num_bottlenecks: int = 4,
                 text_tokens: int     = 4,
                 dropout: float       = 0.1,
                 bert_name: str       = 'bert-base-uncased'):
        super().__init__()

        self.d_model         = d_model
        self.num_bottlenecks = num_bottlenecks

        # ── Проекторы модальностей ────────────────────────────────
        self.text_projector   = TextProjector(
            d_model=d_model, num_tokens=text_tokens, bert_name=bert_name
        )
        self.audio_projector = AudioCNNProjector(
            input_dim=AUDIO_DIM, d_model=d_model
        )
        self.visual_projector = VisualBiLSTMProjector(
            d_model=d_model
        )

        # ── Learnable Bottleneck токены ───────────────────────────
        # Инициализируем как learnable параметры (не зависят от входа)
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottlenecks, d_model) * 0.02
        )

        # ── Bottleneck Fusion слои ────────────────────────────────
        self.fusion_layers = nn.ModuleList([
            BottleneckFusionLayer(
                d_model=d_model, nhead=nhead,
                num_modalities=3, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # ── Классификатор ────────────────────────────────────────
        # Используем bottleneck токены для классификации
        classifier_in = d_model * num_bottlenecks

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, text_inputs, audio_feat, visual_feat):
        """
        text_inputs : dict {input_ids: (B,L), attention_mask: (B,L)}
        audio_feat  : (B, 74)
        visual_feat : (B, 35)

        Returns:
            logits : (B, num_classes)
        """
        B = audio_feat.size(0)

        # ── Проекция в d_model ────────────────────────────────────
        text_tokens   = self.text_projector(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )                                          # (B, text_tokens, d_model)
        audio_tokens  = self.audio_projector(audio_feat)   # (B, 1, d_model)
        visual_tokens = self.visual_projector(visual_feat) # (B, 1, d_model)

        # ── Инициализируем bottleneck токены ──────────────────────
        # Expand до размера батча
        bottleneck = self.bottleneck_tokens.expand(B, -1, -1)  # (B, B_n, d_model)

        # ── Последовательные Bottleneck Fusion слои ───────────────
        modality_tokens = [text_tokens, audio_tokens, visual_tokens]

        for layer in self.fusion_layers:
            modality_tokens, bottleneck = layer(modality_tokens, bottleneck)

        # ── Классификация по bottleneck токенам ──────────────────
        # Flatten bottleneck: (B, B_n * d_model)
        bn_flat = bottleneck.reshape(B, -1)        # (B, num_bottlenecks * d_model)
        logits  = self.classifier(bn_flat)         # (B, num_classes)
        return logits


# ══════════════════════════════════════════════════════════════════
# БЫСТРАЯ ПРОВЕРКА
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = BottleneckTransformer(
        d_model=128,
        num_layers=4,
        nhead=4,
        num_bottlenecks=4,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
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
    print(f"\nLogits shape: {logits.shape}")   # (4, 3)
    print(f"Probabilities: {torch.softmax(logits, dim=-1)}")
    print("\n✓ BottleneckTransformer работает корректно!")

    # Проверяем совместимость с baseline (одинаковый интерфейс)
    from baseline import LateFusionBaseline
    baseline = LateFusionBaseline().to(device)
    bl_logits = baseline(text_inputs, audio, visual)
    print(f"✓ Baseline logits shape: {bl_logits.shape}")
    print("\n✓ Оба models имеют одинаковый интерфейс — train.py работает с обоими!")
