"""
app.py для Hugging Face Spaces
Деплой: загрузить этот файл + requirements.txt в Space
Веса модели загружаются с HF Hub при старте
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import lognorm
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import gradio as gr

# ── Конфиг ────────────────────────────────────────────────────────────────────
HF_REPO = "ksagapov/smart-ad-ranker-weights"
DEMO_MODE = False


# ── Модели ────────────────────────────────────────────────────────────────────
class FMLayer(nn.Module):
    def forward(self, x):
        sq_sum = x.sum(dim=1) ** 2
        sum_sq = (x ** 2).sum(dim=1)
        return 0.5 * (sq_sum - sum_sq).sum(dim=1, keepdim=True)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.2):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden=(256, 256), dropout=0.2):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = nn.Embedding(sum(field_dims), embed_dim, padding_idx=0)
        self.register_buffer(
            "offsets",
            torch.tensor([0, *torch.cumsum(torch.tensor(field_dims[:-1]), 0).tolist()])
        )
        self.fm = FMLayer()
        self.mlp = MLP(self.num_fields * embed_dim, list(hidden), dropout)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x + self.offsets
        emb = self.embedding(x)
        return (self.fm(emb) + self.mlp(emb.view(x.size(0), -1)) + self.bias).squeeze(1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class PlattCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return torch.sigmoid(self.a * logits + self.b)


@dataclass
class Advertiser:
    name: str
    kind: str
    daily_budget: float
    base_bid: float
    spent: float = 0.0
    wins: int = 0
    clicks: int = 0
    conversions: int = 0
    seen: int = 0

    @property
    def remaining(self):
        return max(0.0, self.daily_budget - self.spent)

    def effective_bid(self, rng):
        if self.remaining <= 0:
            return 0.0

        # Лёгкая адаптация под остаток бюджета
        pacing = min(1.0, self.remaining / self.daily_budget * 2.0)

        # Небольшой шум на каждом аукционе, чтобы ставки не были слишком ровные
        noise = rng.lognormal(mean=0.0, sigma=0.10)

        return self.base_bid * pacing * noise


# ── Загрузка весов и инициализация ────────────────────────────────────────────
print("Загружаю модели и индексы...")

try:
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename="deepfm.pt")
    vocab_path = hf_hub_download(repo_id=HF_REPO, filename="vocab.pkl")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = DeepFM(ckpt["field_dims"], embed_dim=16, hidden=(256, 256))
    model.load_state_dict(ckpt["state"], strict=False)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    ALL_FEAT_COLS = ckpt["all_feat_cols"]
    calibrator = PlattCalibrator()
    model.eval()
    print(f"Все ресурсы загружены. Признаков DeepFM: {len(ALL_FEAT_COLS)}")

except Exception as e:
    print(f"Веса не найдены ({e}), запускаю в демо-режиме")
    DEMO_MODE = True
    field_dims = [10] * 27
    ALL_FEAT_COLS = [f"f{i}" for i in range(27)]
    model = DeepFM(field_dims, embed_dim=16, hidden=(256, 256))
    calibrator = PlattCalibrator()
    model.eval()


# ── Вспомогательные функции математики и симуляции ───────────────────────────
def win_rate_curve(bid, market_avg=2.0, market_std=1.0):
    sigma = market_std / market_avg
    mu = np.log(market_avg) - 0.5 * sigma**2
    return lognorm.cdf(bid, s=sigma, scale=np.exp(mu))


def make_competitors(n_comp, market_avg, rng):
    archetypes = [
        {
            "kind": "Лоу-спенд",
            "budget_range": (30, 90),
            "bid_mult_range": (0.45, 0.85),
            "kind_prob": 0.30,
        },
        {
            "kind": "Обычный",
            "budget_range": (80, 220),
            "bid_mult_range": (0.85, 1.20),
            "kind_prob": 0.50,
        },
        {
            "kind": "Агрессивный",
            "budget_range": (160, 500),
            "bid_mult_range": (1.20, 2.50),
            "kind_prob": 0.20,
        },
    ]

    probs = np.array([a["kind_prob"] for a in archetypes], dtype=float)
    probs = probs / probs.sum()

    advertisers = []
    for i in range(int(n_comp)):
        idx = rng.choice(len(archetypes), p=probs)
        arch = archetypes[idx]

        daily_budget = rng.uniform(*arch["budget_range"]) * rng.lognormal(0, 0.18)
        base_bid = market_avg * rng.uniform(*arch["bid_mult_range"]) * rng.lognormal(0, 0.12)

        advertisers.append(
            Advertiser(
                name=f"Конкурент {i + 1}",
                kind=arch["kind"],
                daily_budget=float(max(10.0, daily_budget)),
                base_bid=float(max(0.03, base_bid)),
            )
        )

    return advertisers


def run_simulation(advertisers, n_auctions=30_000, floor=0.1,
                   avg_ctr=0.005, avg_cvr=0.02, seed=None):
    rng = np.random.default_rng(seed)

    for a in advertisers:
        a.spent = a.wins = a.clicks = a.conversions = a.seen = 0

    history = []
    for _ in range(int(n_auctions)):
        bids = {}

        for a in advertisers:
            a.seen += 1
            b = a.effective_bid(rng)
            if b >= floor:
                bids[a.name] = b

        if not bids:
            continue

        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        winner_name = sorted_bids[0][0]
        price_paid = sorted_bids[1][1] if len(sorted_bids) > 1 else floor

        winner = next(a for a in advertisers if a.name == winner_name)
        winner.spent += price_paid / 1000.0
        winner.wins += 1

        clicked = rng.random() < avg_ctr
        converted = clicked and (rng.random() < avg_cvr)
        if clicked:
            winner.clicks += 1
        if converted:
            winner.conversions += 1

        history.append({"winner": winner_name, "click": int(clicked)})

    return history


# ── Логика Gradio функций ─────────────────────────────────────────────────────
def predict_ctr_ui(banner_pos, device_type, hour_of_day, is_weekend):
    hour_int = int(hour_of_day)
    is_peak_computed = 1 if hour_int in [8, 9, 12, 13, 18, 19, 20, 21] else 0

    features = torch.zeros(1, len(ALL_FEAT_COLS), dtype=torch.long)
    feat_map = {
        "banner_pos": int(banner_pos),
        "device_type": int(device_type),
        "hour_of_day": hour_int,
        "is_weekend": int(is_weekend),
        "is_peak": is_peak_computed,
    }

    for i, col in enumerate(ALL_FEAT_COLS):
        if col in feat_map:
            features[0, i] = feat_map[col]

    with torch.no_grad():
        logit = model(features).item()
        raw = torch.sigmoid(torch.tensor(logit)).item()
        cal = calibrator(torch.tensor([[logit]])).item()

    bar_len = int(round(cal * 10))
    bar_len = max(0, min(10, bar_len))
    bar = "█" * bar_len + "░" * (10 - bar_len)
    verdict = "Высокий CTR" if cal > 0.01 else "Низкий CTR"

    mode_note = "\n\n> Внимание: демо-режим — модель не обучена, веса случайные" if DEMO_MODE else ""

    return (
        f"### {verdict}\n\n"
        f"| | |\n|---|---|\n"
        f"| Сырое предсказание | `{raw:.5f}` |\n"
        f"| После Platt Scaling | `{cal:.5f}` |\n\n"
        f"**CTR:** {bar} {cal*100:.3f}%"
        + mode_note
    )


def run_auction_ui(your_bid, your_budget, n_comp, market_avg, n_auctions):
    # Чуть случайности в составе рынка каждый запуск
    rng = np.random.default_rng()

    your_bid = float(your_bid)
    your_budget = float(your_budget)
    market_avg = float(market_avg)

    advs = [Advertiser("Вы", "Ваш аккаунт", your_budget, your_bid)]
    advs.extend(make_competitors(int(n_comp), market_avg, rng))

    run_simulation(advs, n_auctions=int(n_auctions), seed=None)

    you = next(a for a in advs if a.name == "Вы")
    wr = you.wins / max(you.seen, 1)
    cpa = you.spent / you.conversions if you.conversions > 0 else float("inf")
    cpa_s = f"${cpa:.2f}" if cpa < 1e6 else "inf (нет конверсий)"

    pct = you.spent / you.daily_budget * 100
    pct = max(0.0, min(100.0, pct))
    burn_bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

    rows = "\n".join(
        f"| {a.name} | {a.kind} | ${a.daily_budget:.0f} | ${a.spent:.2f} | "
        f"{a.wins:,} | {a.wins / max(a.seen, 1):.1%} | "
        f"{'${:.2f}'.format(a.spent / a.conversions) if a.conversions else 'inf'} |"
        for a in advs
    )

    return (
        f"## Итоги кампании\n\n"
        f"**Бюджет:** {burn_bar} {pct:.1f}% использовано (${you.spent:.2f} / ${you.daily_budget:.0f})\n\n"
        f"| Метрика | Значение |\n|---|---|\n"
        f"| Побед в аукционах | {you.wins:,} |\n"
        f"| Кликов | {you.clicks:,} |\n"
        f"| Конверсий | {you.conversions:,} |\n"
        f"| Win Rate | {wr:.2%} |\n"
        f"| CPA | {cpa_s} |\n\n"
        f"## Все участники\n\n"
        f"| Участник | Тип | Бюджет | Потрачено | Побед | Win% | CPA |\n"
        f"|---|---|---|---|---|---|---|\n"
        f"{rows}"
    )


def optimize_bid_ui(target_cpa, ctr, cvr, market_avg, market_std):
    target_cpa = float(target_cpa)
    ctr = float(ctr)
    cvr = float(cvr)
    market_avg = float(market_avg)
    market_std = float(market_std)

    max_cpm = target_cpa * ctr * cvr * 1000
    bids = np.linspace(0.01, max_cpm * 1.5, 300)
    wrs = win_rate_curve(bids, market_avg, market_std)
    exp_cpa = np.where(wrs > 0, (bids / 1000) / (wrs * ctr * cvr + 1e-12), np.inf)

    feasible = bids[exp_cpa <= target_cpa]
    warning = ""

    if len(feasible) == 0:
        opt_bid = bids[np.nanargmin(np.where(np.isinf(exp_cpa), np.nan, exp_cpa))]
        min_cpa = float(np.nanmin(exp_cpa))
        warning = (
            f"\n\n> Внимание: Target CPA ${target_cpa:.0f} недостижим при данных параметрах рынка. "
            f"Минимальный CPA = **${min_cpa:.2f}**"
        )
    else:
        opt_bid = feasible[-1]

    opt_wr = win_rate_curve(opt_bid, market_avg, market_std)
    opt_cpa = (opt_bid / 1000) / (opt_wr * ctr * cvr + 1e-12)

    n_pts = 30
    sample = np.linspace(0.01, max_cpm * 1.5, n_pts)
    wr_s = win_rate_curve(sample, market_avg, market_std)

    chart = "Win Rate по ставкам:\n```\n"
    for b, w in zip(sample[::3], wr_s[::3]):
        bar = "█" * int(w * 20)
        mark = " <- оптимум" if abs(b - opt_bid) < (sample[1] - sample[0]) * 1.5 else ""
        chart += f"${b:4.2f}  {bar:<20} {w:.0%}{mark}\n"
    chart += "```"

    return (
        f"## Результат оптимизации\n\n"
        f"| Параметр | Значение |\n|---|---|\n"
        f"| Оптимальная ставка | **${opt_bid:.3f} CPM** |\n"
        f"| Макс. допустимая CPM | ${max_cpm:.3f} |\n"
        f"| Ожидаемый Win Rate | {opt_wr:.2%} |\n"
        f"| Ожидаемый CPA | ${opt_cpa:.2f} |"
        + warning
        + f"\n\n{chart}"
    )


# ── Построение интерфейса ─────────────────────────────────────────────────────
with gr.Blocks(
    title="Smart Ad Ranker",
    theme=gr.themes.Soft(primary_hue="violet"),
    css=".gradio-container { max-width: 900px; margin: auto; }"
) as demo:

    gr.Markdown(
        "# Smart Ad Ranker\n"
        "CTR-предсказание, RTB-аукцион, оптимизация ставок\n\n"
        "> Pet-проект: DeepFM + Platt Scaling на датасете Avazu (40M показов mobile рекламы)"
    )

    with gr.Tab("CTR Oracle"):
        gr.Markdown("Выбери параметры рекламного показа — получи предсказание вероятности клика.")
        with gr.Row():
            with gr.Column(scale=1):
                bp = gr.Slider(0, 7, step=1, value=0, label="Позиция баннера")
                dt = gr.Dropdown(
                    choices=[
                        ("Десктоп/Прочее", 0),
                        ("Смартфон", 1),
                        ("Планшет", 4),
                        ("Гаджеты/Другое", 5),
                    ],
                    value=1,
                    label="Тип устройства",
                )
                hod = gr.Slider(0, 23, step=1, value=12, label="Час суток")
                wk = gr.Checkbox(label="Выходной день")
                ctr_btn = gr.Button("Предсказать", variant="primary")
            with gr.Column(scale=1):
                ctr_out = gr.Markdown("*Нажми кнопку для предсказания*")

        ctr_btn.click(predict_ctr_ui, inputs=[bp, dt, hod, wk], outputs=ctr_out)


    with gr.Tab("Auction Simulator"):
        gr.Markdown("Запусти RTB-аукцион и посмотри, как расходуется бюджет против конкурентов.")
        with gr.Row():
            with gr.Column(scale=1):
                ybid = gr.Slider(0.1, 20.0, value=2.5, step=0.1, label="Твоя ставка (CPM $)")
                ybdg = gr.Number(value=100.0, label="Дневной бюджет ($)")
                ncomp = gr.Slider(1, 10, step=1, value=5, label="Число конкурентов")
                mkta = gr.Slider(0.5, 10.0, value=2.0, step=0.1, label="Средняя ставка рынка (CPM $)")
                nauc = gr.Slider(5000, 100000, step=5000, value=30000, label="Аукционов в день")
                auction_btn = gr.Button("Запустить", variant="primary")
            with gr.Column(scale=1):
                sim_out = gr.Markdown("*Задай параметры и нажми кнопку*")

        auction_btn.click(run_auction_ui, inputs=[ybid, ybdg, ncomp, mkta, nauc], outputs=sim_out)

    with gr.Tab("Bid Optimizer"):
        gr.Markdown("Найди оптимальную CPM-ставку для достижения целевого CPA.")
        with gr.Row():
            with gr.Column(scale=1):
                tcpa = gr.Slider(5, 200, value=30, label="Target CPA ($)")
                tctr = gr.Slider(0.001, 0.05, step=0.001, value=0.005, label="Ожидаемый CTR")
                tcvr = gr.Slider(0.001, 0.10, step=0.001, value=0.02, label="Ожидаемый CVR")
                mavg = gr.Slider(0.5, 10.0, value=2.0, step=0.1, label="Средний CPM рынка ($)")
                mstd = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Разброс CPM рынка ($)")
                opt_btn = gr.Button("Оптимизировать", variant="primary")
            with gr.Column(scale=1):
                opt_out = gr.Markdown("*Задай параметры и нажми кнопку*")

        opt_btn.click(optimize_bid_ui, inputs=[tcpa, tctr, tcvr, mavg, mstd], outputs=opt_out)


if __name__ == "__main__":
    demo.launch()
