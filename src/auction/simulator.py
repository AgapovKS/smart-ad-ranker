"""
RTB Auction Simulator.

Implements:
  - Second-price auction (Vickrey)
  - First-price auction
  - Budget pacing (uniform throttling)
  - Bid landscape estimation (log-normal win rate curve)
  - CPA / ROAS estimation given CTR predictions
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from collections import defaultdict


@dataclass
class Advertiser:
    name: str
    daily_budget: float          # USD
    base_bid: float              # max CPM bid
    target_cpa: Optional[float] = None
    target_roas: Optional[float] = None
    # Runtime state
    spent: float = 0.0
    wins: int = 0
    clicks: int = 0
    conversions: int = 0
    impressions_seen: int = 0

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.daily_budget - self.spent)

    @property
    def pacing_factor(self) -> float:
        """Uniform budget pacing: throttle bid if ahead of schedule."""
        return min(1.0, self.remaining_budget / (self.daily_budget + 1e-9) * 24)

    def effective_bid(self) -> float:
        return self.base_bid * self.pacing_factor

    def reset(self):
        self.spent = self.wins = self.clicks = self.conversions = self.impressions_seen = 0
        self.spent = 0.0


@dataclass
class AuctionResult:
    winner: Optional[str]
    winning_price: float          # CPM paid (2nd price)
    clearing_price: float         # 1st price
    all_bids: dict[str, float]


class AuctionSimulator:
    """
    Simulates a stream of RTB auctions for one day.

    Each auction:
      1. Each advertiser submits effective_bid (with pacing)
      2. Winner = highest bidder
      3. Paid price = 2nd highest (Vickrey) or highest (1st price)
      4. Winner gets impression; click/conversion sampled from CTR/CVR
    """

    def __init__(
        self,
        advertisers: list[Advertiser],
        auction_type: Literal["second_price", "first_price"] = "second_price",
        floor_price: float = 0.1,         # minimum CPM in USD
        avg_ctr: float = 0.005,
        avg_cvr: float = 0.02,
        avg_order_value: float = 50.0,
        seed: int = 42,
    ):
        self.advertisers = {a.name: a for a in advertisers}
        self.auction_type = auction_type
        self.floor_price = floor_price
        self.avg_ctr = avg_ctr
        self.avg_cvr = avg_cvr
        self.avg_order_value = avg_order_value
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def run_auction(self, impression_ctr: Optional[float] = None) -> AuctionResult:
        ctr = impression_ctr or self.avg_ctr
        bids = {}
        for name, adv in self.advertisers.items():
            adv.impressions_seen += 1
            if adv.remaining_budget <= 0:
                continue
            bid = adv.effective_bid()
            # Add random noise to model private value uncertainty
            bid *= self.rng.lognormal(0, 0.05)
            bids[name] = round(bid, 4)

        eligible = {n: b for n, b in bids.items() if b >= self.floor_price}
        if not eligible:
            return AuctionResult(None, 0.0, 0.0, bids)

        sorted_bids = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        winner_name, clearing_price = sorted_bids[0]

        if self.auction_type == "second_price":
            winning_price = sorted_bids[1][1] if len(sorted_bids) > 1 else self.floor_price
        else:
            winning_price = clearing_price

        # Cost in USD (CPM → per impression: /1000)
        cost = winning_price / 1000.0
        winner = self.advertisers[winner_name]
        winner.spent += cost
        winner.wins += 1

        # Sample click & conversion
        clicked = self.rng.random() < ctr
        if clicked:
            winner.clicks += 1
            converted = self.rng.random() < self.avg_cvr
            if converted:
                winner.conversions += 1

        result = AuctionResult(winner_name, winning_price, clearing_price, bids)
        self.history.append({
            "winner": winner_name,
            "price_paid": winning_price,
            "click": int(clicked),
            "conversion": int(clicked and converted if clicked else False),
            "cost": cost,
        })
        return result

    def run_day(self, n_auctions: int = 100_000, ctrs: Optional[np.ndarray] = None) -> dict:
        for adv in self.advertisers.values():
            adv.reset()
        self.history.clear()

        for i in range(n_auctions):
            ctr = ctrs[i] if ctrs is not None else None
            self.run_auction(ctr)

        return self.summary()

    def summary(self) -> dict:
        result = {}
        for name, adv in self.advertisers.items():
            cpa = (adv.spent / adv.conversions) if adv.conversions > 0 else float("inf")
            result[name] = {
                "spent": round(adv.spent, 2),
                "budget": adv.daily_budget,
                "budget_utilization": round(adv.spent / adv.daily_budget * 100, 1),
                "wins": adv.wins,
                "clicks": adv.clicks,
                "conversions": adv.conversions,
                "ctr": round(adv.clicks / max(adv.wins, 1), 4),
                "cvr": round(adv.conversions / max(adv.clicks, 1), 4),
                "cpa": round(cpa, 2),
                "win_rate": round(adv.wins / max(adv.impressions_seen, 1), 4),
            }
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Bid landscape: given a bid, estimate win probability
# ──────────────────────────────────────────────────────────────────────────────

def estimate_win_rate(bid: float, market_avg: float = 2.0, market_std: float = 1.0) -> float:
    """
    Approximate win probability using log-normal market price distribution.
    P(win) = P(market_price < bid) = CDF_lognorm(bid)
    """
    from scipy.stats import lognorm
    sigma = market_std / market_avg
    mu = np.log(market_avg) - 0.5 * sigma**2
    return lognorm.cdf(bid, s=sigma, scale=np.exp(mu))


def optimal_bid_for_cpa(
    target_cpa: float,
    cvr: float,
    ctr: float,
    market_avg: float = 2.0,
    market_std: float = 1.0,
    grid_size: int = 200,
) -> dict:
    """
    Find bid that maximizes conversions subject to CPA ≤ target_cpa.
    Uses a sweep over bid values.

    CPA = CPM / (CTR * CVR * 1000)
    => max CPM = target_cpa * CTR * CVR * 1000
    """
    max_cpm = target_cpa * ctr * cvr * 1000
    bids = np.linspace(0.01, max_cpm * 1.5, grid_size)
    win_rates = np.array([estimate_win_rate(b, market_avg, market_std) for b in bids])
    expected_conversions = win_rates * ctr * cvr
    cpa_at_bid = np.where(expected_conversions > 0, (bids / 1000) / expected_conversions, np.inf)

    # Best bid: maximize conversions subject to CPA constraint
    feasible = bids[cpa_at_bid <= target_cpa]
    if len(feasible) == 0:
        return {"optimal_bid": 0.0, "expected_cpa": float("inf"), "win_rate": 0.0}

    opt_bid = feasible[-1]  # highest feasible bid → most conversions
    opt_wr = estimate_win_rate(opt_bid, market_avg, market_std)
    opt_cpa = (opt_bid / 1000) / (opt_wr * ctr * cvr + 1e-12)

    return {
        "optimal_bid_cpm": round(opt_bid, 3),
        "max_allowed_cpm": round(max_cpm, 3),
        "expected_win_rate": round(opt_wr, 4),
        "expected_cpa": round(opt_cpa, 2),
        "bid_landscape": {
            "bids": bids.tolist(),
            "win_rates": win_rates.tolist(),
            "cpa_at_bid": np.where(np.isinf(cpa_at_bid), -1, cpa_at_bid).tolist(),
        },
    }
