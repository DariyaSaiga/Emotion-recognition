"""
models.py — все модели для мультимодального распознавания эмоций

Содержит:
    AudioEncoder    — 1D-CNN для аудио (COVAREP, 74 фичи)
    VisualEncoder   — BiLSTM для видео (OpenFace, 713 фич)
    TextEncoder     — замороженный BERT для текста (768-dim)
    BaselineModel   — конкатенация трёх энкодеров (для сравнения)
    BottleneckModel — Attention Bottleneck Fusion (основная модель)
"""

import torch
import torch.nn as nn
from transformers import BertModel


# ============================================================
# 1. AUDIO ENCODER — 1D-CNN
# ============================================================

class AudioEncoder(nn.Module):
    """
    Что делает: извлекает признаки из аудио последовательности.

    Почему 1D-CNN:
        COVAREP фичи — это временной ряд. CNN хорошо находит
        локальные паттерны (например резкое изменение интонации).
        Он быстрее LSTM и хорошо работает на аудио.

    Архитектура:
        Вход [batch, seq_len, 74]
        → переставляем оси → [batch, 74, seq_len]  (CNN ждёт каналы на второй позиции)
        → Conv1d слой 1: 74 → 128 каналов
        → BatchNorm + ReLU + Dropout
        → Conv1d слой 2: 128 → 128 каналов
        → BatchNorm + ReLU + Dropout
        → Global Average Pooling: усредняем по времени → [batch, 128]
        → Linear: 128 → 128
        Выход [batch, 128]

    BatchNorm — нормализует активации внутри слоя.
        Это стабилизирует обучение и позволяет использовать
        больший learning rate. Без него CNN часто "взрывается"
        или "умирает" в первые эпохи.

    Dropout(0.3) — случайно отключает 30% нейронов во время обучения.
        Это регуляризация — модель не может полагаться на один нейрон
        и вынуждена учить более устойчивые признаки.
        Помогает против переобучения.
    """

    def __init__(self, input_dim=74, hidden_dim=128, dropout=0.3):
        super(AudioEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Слой 1
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            # kernel_size=3: смотрим на 3 соседних временных шага
            # padding=1: сохраняем длину последовательности
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Слой 2
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Global Average Pooling — усредняем по времени
        # Превращает [batch, 128, seq_len] → [batch, 128]
        # Это делает модель независимой от длины входа
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Финальный линейный слой
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [batch, seq_len, 74]

        x = x.transpose(1, 2)          # → [batch, 74, seq_len]
        x = self.encoder(x)             # → [batch, 128, seq_len]
        x = self.pool(x)                # → [batch, 128, 1]
        x = x.squeeze(-1)               # → [batch, 128]
        x = self.fc(x)                  # → [batch, 128]
        return x


# ============================================================
# 2. VISUAL ENCODER — BiLSTM
# ============================================================

class VisualEncoder(nn.Module):
    """
    Что делает: извлекает признаки из последовательности кадров лица.

    Почему BiLSTM:
        Мимика разворачивается во времени — улыбка нарастает
        и спадает. LSTM хранит память о прошлых кадрах.
        Bi (двунаправленный) — читает и вперёд и назад,
        что даёт полный контекст выражения лица.

    Почему сначала уменьшаем 713 → 256:
        713 признаков это много. LSTM с таким входом будет
        очень медленным и будет переобучаться. Линейный слой
        сжимает до 256 — оставляем самое важное.

    Архитектура:
        Вход [batch, seq_len, 713]
        → Linear проекция: 713 → 256
        → Dropout
        → BiLSTM: 256 → 128 (в каждом направлении)
        → берём последний скрытый вектор обоих направлений
        → concatenate → [batch, 256]
        → Linear: 256 → 128
        Выход [batch, 128]
    """

    def __init__(self, input_dim=713, proj_dim=256, hidden_dim=128, dropout=0.3):
        super(VisualEncoder, self).__init__()

        # Проекционный слой: уменьшаем размерность перед LSTM
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # BiLSTM: bidirectional=True означает два LSTM — вперёд и назад
        # hidden_size=hidden_dim в каждом направлении
        # → итого выход 2 * hidden_dim = 256
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=2,           # два слоя LSTM для большей глубины
            batch_first=True,       # первая ось = batch (не seq)
            bidirectional=True,
            dropout=dropout,        # dropout между слоями LSTM
        )

        # Финальный слой: 256 → 128
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, mask=None):
        # x: [batch, seq_len, 713]

        x = self.projection(x)          # → [batch, seq_len, 256]

        # LSTM обрабатывает последовательность
        out, (hidden, _) = self.lstm(x)
        # hidden: [4, batch, 128] — 4 = 2 слоя × 2 направления

        # Берём последний слой, оба направления
        # hidden[-2]: последний слой, прямое направление
        # hidden[-1]: последний слой, обратное направление
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        # → [batch, 256]

        out = self.fc(last_hidden)      # → [batch, 128]
        return out


# ============================================================
# 3. TEXT ENCODER — замороженный BERT
# ============================================================

class TextEncoder(nn.Module):
    """
    Что делает: превращает текст в вектор смысла через BERT.

    Почему BERT заморожен:
        BERT уже обучен на огромном корпусе (Wikipedia + книги).
        Если его размораживать — он начнёт переобучаться на наших
        6374 сэмплах и "забудет" всё что знал. Мы это проверяли —
        train accuracy 99%, val падает. Поэтому только frozen.

    Почему [CLS] токен:
        BERT добавляет специальный токен [CLS] в начало каждой фразы.
        Во время предобучения этот токен учится быть "суммарным
        представлением" всего предложения. Именно он нам и нужен.

    Архитектура:
        Вход [batch, seq_len, 768]  ← уже посчитанные эмбеддинги из HDF5
        → берём первый токен (CLS): [batch, 768]
        → Linear проекция: 768 → 256
        Выход [batch, 256]

    Примечание: в нашем датасете текст уже в виде эмбеддингов (768-dim)
    из HDF5 файла. Поэтому BERT используем только для проекции,
    а не для полного прохода токенов.
    """

    def __init__(self, bert_dim=768, output_dim=256, dropout=0.3):
        super(TextEncoder, self).__init__()

        # Проекция: уменьшаем 768 → 256 чтобы совпадало с другими энкодерами
        # по смысловой ёмкости
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [batch, seq_len, 768]

        # Берём первый временной шаг — это [CLS] токен
        cls_token = x[:, 0, :]          # → [batch, 768]

        out = self.projection(cls_token) # → [batch, 256]
        return out


# ============================================================
# 4. BASELINE MODEL — простая конкатенация
# ============================================================

class BaselineModel(nn.Module):
    """
    Что делает: объединяет три энкодера через простую конкатенацию.

    Почему это baseline:
        Это самый простой способ слить три модальности.
        Просто склеиваем векторы рядом и подаём в классификатор.
        Никакого взаимодействия между модальностями — каждая
        просто добавляет свои признаки.

    Архитектура:
        audio_feat:  [batch, 128]
        visual_feat: [batch, 128]
        text_feat:   [batch, 256]
        → concatenate → [batch, 512]
        → Linear(512, 256) + ReLU + Dropout
        → Linear(256, 3)   ← 3 класса: happy, sad, anger
        Выход [batch, 3]  ← logits (сырые оценки до softmax)
    """

    def __init__(self, dropout=0.3):
        super(BaselineModel, self).__init__()

        self.audio_encoder  = AudioEncoder()
        self.visual_encoder = VisualEncoder()
        self.text_encoder   = TextEncoder()

        # 128 + 128 + 256 = 512
        fusion_dim = 128 + 128 + 256

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, audio, visual, text, mask=None):
        audio_feat  = self.audio_encoder(audio)         # [batch, 128]
        visual_feat = self.visual_encoder(visual, mask) # [batch, 128]
        text_feat   = self.text_encoder(text)           # [batch, 256]

        # Конкатенируем по последней оси
        fused = torch.cat([audio_feat, visual_feat, text_feat], dim=-1)
        # → [batch, 512]

        logits = self.classifier(fused)                 # → [batch, 3]
        return logits


# ============================================================
# 5. BOTTLENECK MODEL — Attention Bottleneck Fusion
# ============================================================

class BottleneckFusion(nn.Module):
    """
    Что делает: позволяет модальностям "общаться" друг с другом
    через маленькие обучаемые токены (latent tokens).

    Идея простыми словами:
        В baseline модели audio, visual и text просто складываются рядом.
        Они не взаимодействуют. Но в реальности эмоции — это
        согласованность: злой голос + нахмуренное лицо + грубые слова.

        Bottleneck создаёт N маленьких "посредников" (latent tokens).
        Каждый посредник смотрит на все три модальности через Attention
        и собирает нужную информацию. Потом классификатор смотрит
        на этих посредников.

        Это как редактор газеты: вместо того чтобы читать все три
        источника сразу — он читает краткие сводки (latent tokens)
        которые уже выделили главное из каждого источника.

    Attention механизм:
        Посредник задаёт вопрос (Query): "что важно для этой эмоции?"
        Модальность отвечает (Key, Value): "вот мои признаки"
        Attention считает насколько каждый признак важен для вопроса
        и берёт взвешенную сумму.

    Параметры:
        num_tokens  — сколько посредников (4 достаточно)
        bottleneck_dim — размер посредника (64)
        num_heads   — сколько независимых вопросов задаёт каждый посредник
    """

    def __init__(self, text_dim=256, audio_dim=128, visual_dim=128,
                 bottleneck_dim=64, num_tokens=4, num_heads=4, dropout=0.3):
        super(BottleneckFusion, self).__init__()

        self.num_tokens    = num_tokens
        self.bottleneck_dim = bottleneck_dim

        # Проецируем все модальности в одно пространство (bottleneck_dim)
        # Чтобы посредники могли сравнивать их между собой
        self.proj_text   = nn.Linear(text_dim,   bottleneck_dim)
        self.proj_audio  = nn.Linear(audio_dim,  bottleneck_dim)
        self.proj_visual = nn.Linear(visual_dim, bottleneck_dim)

        # Latent tokens — обучаемые параметры посредников
        # Инициализируются случайно, обучаются вместе с моделью
        self.latent_tokens = nn.Parameter(
            torch.randn(num_tokens, bottleneck_dim)
        )

        # Attention слои: посредники смотрят на каждую модальность
        # MultiheadAttention(embed_dim, num_heads)
        # Query = latent tokens, Key = Value = фичи модальности
        self.attn_text   = nn.MultiheadAttention(bottleneck_dim, num_heads,
                                                  dropout=dropout, batch_first=True)
        self.attn_audio  = nn.MultiheadAttention(bottleneck_dim, num_heads,
                                                  dropout=dropout, batch_first=True)
        self.attn_visual = nn.MultiheadAttention(bottleneck_dim, num_heads,
                                                  dropout=dropout, batch_first=True)

        # Layer Norm после каждого attention — стабилизирует обучение
        self.norm_text   = nn.LayerNorm(bottleneck_dim)
        self.norm_audio  = nn.LayerNorm(bottleneck_dim)
        self.norm_visual = nn.LayerNorm(bottleneck_dim)

        # Feed-forward: обрабатываем собранную информацию
        self.ff = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )
        self.norm_ff = nn.LayerNorm(bottleneck_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat, visual_feat):
        # text_feat:   [batch, 256] → проецируем в [batch, 64]
        # audio_feat:  [batch, 128] → проецируем в [batch, 64]
        # visual_feat: [batch, 128] → проецируем в [batch, 64]

        batch_size = text_feat.size(0)

        # Проецируем в общее пространство и добавляем ось времени
        # unsqueeze(1): [batch, 64] → [batch, 1, 64]
        t = self.proj_text(text_feat).unsqueeze(1)     # [batch, 1, 64]
        a = self.proj_audio(audio_feat).unsqueeze(1)   # [batch, 1, 64]
        v = self.proj_visual(visual_feat).unsqueeze(1) # [batch, 1, 64]

        # Расширяем latent tokens на весь батч
        # latent_tokens: [num_tokens, 64] → [batch, num_tokens, 64]
        tokens = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Attention: посредники смотрят на каждую модальность
        # Query = tokens (посредники задают вопрос)
        # Key = Value = фичи модальности (модальность отвечает)
        tokens_t, _ = self.attn_text(tokens, t, t)
        tokens   = self.norm_text(tokens + tokens_t)   # residual connection

        tokens_a, _ = self.attn_audio(tokens, a, a)
        tokens   = self.norm_audio(tokens + tokens_a)

        tokens_v, _ = self.attn_visual(tokens, v, v)
        tokens   = self.norm_visual(tokens + tokens_v)

        # Feed-forward обработка
        tokens_ff = self.ff(tokens)
        tokens    = self.norm_ff(tokens + tokens_ff)   # residual connection

        # Усредняем по посредникам: [batch, num_tokens, 64] → [batch, 64]
        fused = tokens.mean(dim=1)

        return fused


class BottleneckModel(nn.Module):
    """
    Полная модель с Bottleneck Attention Fusion.

    Архитектура:
        audio  → AudioEncoder  → [batch, 128]
        visual → VisualEncoder → [batch, 128]  → BottleneckFusion → [batch, 64]
        text   → TextEncoder   → [batch, 256]
        → Linear(64, 3)
        Выход [batch, 3]
    """

    def __init__(self, dropout=0.3):
        super(BottleneckModel, self).__init__()

        self.audio_encoder  = AudioEncoder(dropout=dropout)
        self.visual_encoder = VisualEncoder(dropout=dropout)
        self.text_encoder   = TextEncoder(dropout=dropout)

        self.bottleneck = BottleneckFusion(
            text_dim=256,
            audio_dim=128,
            visual_dim=128,
            bottleneck_dim=64,
            num_tokens=4,
            num_heads=4,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, audio, visual, text, mask=None):
        audio_feat  = self.audio_encoder(audio)         # [batch, 128]
        visual_feat = self.visual_encoder(visual, mask) # [batch, 128]
        text_feat   = self.text_encoder(text)           # [batch, 256]

        fused  = self.bottleneck(text_feat, audio_feat, visual_feat)  # [batch, 64]
        logits = self.classifier(fused)                               # [batch, 3]
        return logits


# ============================================================
# Проверка — запусти напрямую
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Проверяем все модели...")
    print("=" * 50)

    batch_size = 4
    seq_len    = 50

    # Создаём случайные тензоры как будто это реальные данные
    audio  = torch.randn(batch_size, seq_len, 74)
    visual = torch.randn(batch_size, seq_len, 713)
    text   = torch.randn(batch_size, seq_len, 768)
    mask   = torch.ones(batch_size, seq_len)

    print("\n[1] AudioEncoder:")
    model = AudioEncoder()
    out   = model(audio)
    print(f"    вход:  {audio.shape}")
    print(f"    выход: {out.shape}  ← ожидаем [4, 128]")
    assert out.shape == (batch_size, 128), "ОШИБКА форма неправильная!"
    print("    ✓ OK")

    print("\n[2] VisualEncoder:")
    model = VisualEncoder()
    out   = model(visual)
    print(f"    вход:  {visual.shape}")
    print(f"    выход: {out.shape}  ← ожидаем [4, 128]")
    assert out.shape == (batch_size, 128), "ОШИБКА форма неправильная!"
    print("    ✓ OK")

    print("\n[3] TextEncoder:")
    model = TextEncoder()
    out   = model(text)
    print(f"    вход:  {text.shape}")
    print(f"    выход: {out.shape}  ← ожидаем [4, 256]")
    assert out.shape == (batch_size, 256), "ОШИБКА форма неправильная!"
    print("    ✓ OK")

    print("\n[4] BaselineModel:")
    model  = BaselineModel()
    logits = model(audio, visual, text, mask)
    print(f"    выход: {logits.shape}  ← ожидаем [4, 3]")
    assert logits.shape == (batch_size, 3), "ОШИБКА форма неправильная!"
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Параметров: {params:,}")
    print("    ✓ OK")

    print("\n[5] BottleneckModel:")
    model  = BottleneckModel()
    logits = model(audio, visual, text, mask)
    print(f"    выход: {logits.shape}  ← ожидаем [4, 3]")
    assert logits.shape == (batch_size, 3), "ОШИБКА форма неправильная!"
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Параметров: {params:,}")
    print("    ✓ OK")

    print("\n" + "=" * 50)
    print("✓ Все модели работают корректно!")
    print("Скинь вывод в чат — переходим к train.py!")
    print("=" * 50)