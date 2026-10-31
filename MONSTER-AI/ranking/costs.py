"""Transaction-cost model for NSE intraday equity (blueprint section 15 "Costs").

Statutory/brokerage components are configurable schedule values (they change over time and by broker); slippage and
impact are liquidity based estimates. The model is used (a) to reject candidates whose round-trip cost is a large share
of the expected move and (b) in every backtest that claims trading results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    brokerage_flat: float = 20.0  # INR per executed order, capped by brokerage_pct of the trade value
    brokerage_pct: float = 0.0003
    stt_sell_pct: float = 0.00025  # intraday: sell side only
    exch_txn_pct: float = 0.0000297
    sebi_pct: float = 0.000001
    stamp_buy_pct: float = 0.00003  # buy side only
    gst_pct: float = 0.18  # on brokerage + exchange + SEBI charges

    def brokerage(self, value: float) -> float:
        return min(self.brokerage_flat, self.brokerage_pct * value)

    def fees(self, buy_value: float, sell_value: float) -> float:
        """Total INR cost of a round trip (buy then sell, or sell then buy) excluding slippage and impact."""
        brokerage = self.brokerage(buy_value) + self.brokerage(sell_value)
        exchange = self.exch_txn_pct * (buy_value + sell_value)
        sebi = self.sebi_pct * (buy_value + sell_value)
        gst = self.gst_pct * (brokerage + exchange + sebi)
        return brokerage + exchange + sebi + gst + self.stt_sell_pct * sell_value + self.stamp_buy_pct * buy_value

    def round_trip_pct(self, notional: float) -> float:
        """Statutory + brokerage cost as a percentage of the traded notional (per round trip)."""
        if notional <= 0:
            return float("nan")
        return self.fees(notional, notional) / notional * 100.0


def slippage_bps(turnover_cr: float | None) -> float:
    """Half-spread + short-term slippage per side in basis points from median daily traded value (INR crore)."""
    if turnover_cr is None or turnover_cr != turnover_cr:
        return 25.0
    for floor, bps in ((1000.0, 1.5), (500.0, 2.5), (200.0, 4.0), (100.0, 6.0), (50.0, 9.0), (25.0, 14.0)):
        if turnover_cr >= floor:
            return bps
    return 25.0


def impact_bps(notional: float, adv_value: float | None) -> float:
    """Square-root market-impact estimate per side (bps), capped; negligible for retail-sized orders in liquid names."""
    if not adv_value or adv_value <= 0 or notional <= 0:
        return 10.0
    return min(50.0, 12.0 * math.sqrt(notional / adv_value))


def estimate_round_trip_cost_pct(
    turnover_cr: float | None, *, notional: float = 200_000.0, model: CostModel | None = None
) -> float:
    """All-in round-trip cost (% of notional): fees + 2 x slippage + 2 x impact."""
    model = model or CostModel()
    adv = turnover_cr * 1e7 if turnover_cr and turnover_cr == turnover_cr else None
    per_side = slippage_bps(turnover_cr) + impact_bps(notional, adv)
    return model.round_trip_pct(notional) + 2.0 * per_side / 100.0
