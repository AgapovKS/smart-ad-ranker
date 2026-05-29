# Smart Ad Ranker

Pet-проект по задачам programmatic-рекламы: предсказание CTR, оптимизация ставок под целевой CPA и симуляция RTB-аукциона.

**[Live Demo на Hugging Face Spaces](https://huggingface.co/spaces/ksagapov/Smart_Ad_Ranker)**

---

## Что это

Три инструмента в одном Gradio-приложении:

| Инструмент | Что делает |
|---|---|
| **CTR Oracle** | Предсказывает P(click) по контексту показа (позиция баннера, тип устройства, час суток, выходной день) |
| **Auction Simulator** | Симулирует RTB-аукцион: кампания пользователя против N конкурентов с разными бюджетами и стратегиями — показывает win rate, потраченный бюджет и CPA |
| **Bid Optimizer** | Находит оптимальную CPM-ставку под целевой CPA через аналитическую кривую win rate |

---

## Архитектура

### CTR-предсказание — DeepFM + Platt Scaling

```
Категориальные признаки (27 полей)
        ↓
   Embedding Layer (dim=16)
   ┌────────────────────────┐
   │  FM Component          │  ←  попарные взаимодействия признаков
   │  MLP Component         │  ←  [256, 256] + BatchNorm + Dropout
   └────────────────────────┘
        ↓
   Platt Scaling  (a · logit + b → σ → P(click))
```

Данные: датасет [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction) (~40M показов мобильной рекламы).

Platt Scaling критичен для CPA-таргетинга: сырые логиты DeepFM не являются калиброванными вероятностями, и без этого шага расчёт ставки по формуле `bid = target_CPA × CTR × CVR × 1000` даёт смещённый результат.

### Bid Optimizer — аналитическая оптимизация под CPA

Рыночное распределение ставок аппроксимируется log-normal, что даёт явную кривую win rate:

```
win_rate(bid) = LogNormal_CDF(bid; μ, σ)

max_CPM = target_CPA × CTR × CVR × 1000

opt_bid = max { bid : (bid / 1000) / (win_rate(bid) × CTR × CVR) ≤ target_CPA }
```

### Auction Simulator — RTB second-price аукцион

Симуляция с тремя архетипами конкурентов:

| Тип | Бюджет | Ставка относительно рынка |
|---|---|---|
| Лоу-спенд | $30–90 | 0.45–0.85× |
| Обычный | $80–220 | 0.85–1.20× |
| Агрессивный | $160–500 | 1.20–2.50× |

Каждый участник использует pacing (замедление при исчерпании бюджета) и lognormal-шум на ставках.

---

## Структура репозитория

```
smart-ad-ranker/
├── configs/          # Конфиги обучения и параметры модели
├── demo/             # Исходники Gradio-приложения (локальный запуск)
├── notebooks/        # Jupyter-ноутбуки: EDA, обучение, анализ
├── src/              # Python-пакет: модели, датасеты, утилиты
├── requirements.txt  # Зависимости
└── README.md
```

Веса модели хранятся на HF Hub: [`ksagapov/smart-ad-ranker-weights`](https://huggingface.co/ksagapov/smart-ad-ranker-weights)
Файлы: `deepfm.pt`, `vocab.pkl`

---

## Быстрый старт

```bash
git clone https://github.com/AgapovKS/smart-ad-ranker
cd smart-ad-ranker
pip install -r requirements.txt
python demo/app.py
```

Веса загружаются автоматически с HF Hub при первом запуске. Если недоступны — приложение стартует в демо-режиме со случайными весами.

---

## Зависимости

| Библиотека | Назначение |
|---|---|
| `torch >= 2.2` | DeepFM |
| `transformers >= 4.40` | утилиты |
| `pandas >= 2.0` | обработка данных |
| `scikit-learn >= 1.4` | утилиты ML |
| `scipy >= 1.12` | log-normal модель рынка |
| `gradio >= 4.29` | веб-интерфейс |
| `plotly >= 5.20` | визуализация |

---

## Статус

- [x] DeepFM + Platt Scaling на Avazu
- [x] Bid Optimizer (log-normal win rate)
- [x] RTB Auction Simulator
- [x] Gradio-приложение на HF Spaces
- [ ] Добавить визуализацию кривой win rate в Bid Optimizer
- [ ] Segment Analyzer: CTR по комбинациям признаков (banner_pos × device_type)
