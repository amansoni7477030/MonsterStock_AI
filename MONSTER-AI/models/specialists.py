"""Specialist model definitions (blueprint section 13): each specialist sees one family of evidence and predicts the
primary targets from it; a meta-model then combines them. Similarity is a pass-through (no fitted model).

The news/event specialist is limited to scheduled company events (board-meeting/results calendar) until enough live
point-in-time snapshots exist to learn from article-level event features (see Settings.min_news_model_days).
"""

from __future__ import annotations

from dataclasses import dataclass

SCALE = ("t_atr14_pct", "t_range_mean20", "t_range_med20", "t_rv20", "t_range_prev")


@dataclass(frozen=True)
class SpecialistSpec:
    name: str
    prefixes: tuple[str, ...]
    extra: tuple[str, ...] = ()
    targets: tuple[str, ...] = ("unusual", "lr")  # unusual: P(vol-adjusted large move); lr: log range; up: close > open
    stages: tuple[str, ...] = ("pre", "final")
    requires: tuple[str, ...] = ()  # rows lacking these inputs get no output from this specialist
    min_rows: int = 6000
    description: str = ""

    def columns(self, available: list[str]) -> list[str]:
        chosen = [c for c in available if c.startswith(self.prefixes) or c in self.extra]
        return list(dict.fromkeys(chosen))


SPECIALISTS: tuple[SpecialistSpec, ...] = (
    SpecialistSpec("technical", ("t_", "s_"), ("u_beta250", "u_range_mean250", "u_sector_code", "u_is_fno"),
                   description="price, volatility, volume, relative strength and sector context"),
    SpecialistSpec("global", ("g_",), (*SCALE, "t_beta60", "t_corr60", "u_beta250", "u_sector_code"),
                   description="global-to-India transmission: overnight factor moves x trailing sensitivities"),
    SpecialistSpec("regime", ("m_",), (*SCALE, "c_dow", "c_dom", "c_month_end", "c_month_start", "c_gap_prev", "c_gap_next",
                                       "u_sector_code"), description="market trend / volatility / risk / expiry regime"),
    SpecialistSpec("fno", ("f_",), (*SCALE, "c_expiry_today", "c_dte_cal", "c_post_expiry"), requires=("f_iv_atm",),
                   description="derivatives positioning, implied volatility and expected move"),
    SpecialistSpec("premarket", ("p_",), (*SCALE, "m_volreg", "m_vix"), stages=("final",),
                   description="pre-open gap and relative gap versus market/sector"),
    SpecialistSpec("events", ("c_results", "c_days_"), (*SCALE, "c_dow"), requires=("c_results_today",),
                   description="scheduled company catalysts (results/board meetings)"),
    SpecialistSpec("direction", ("t_", "s_", "g_", "f_", "p_", "x_sim_up", "m_"), ("u_sector_code",), targets=("up",),
                   min_rows=10000, description="secondary: close-vs-open direction"),
)

SPECIALIST_BY_NAME = {s.name: s for s in SPECIALISTS}

# Context columns handed to the meta-model next to the specialist outputs.
META_CONTEXT = (
    "t_atr14_pct", "t_range_mean20", "t_range_med20", "t_rv20", "t_range_prev", "m_volreg", "m_trend", "m_risk",
    "u_sector_code", "u_is_fno", "t_log_tv", "c_dow", "m_expiry_share", "x_sim_hit2", "x_sim_hit3", "x_sim_unusual",
    "x_sim_range_mean", "x_sim_range_q90", "x_sim_dist_mean", "p_gap_abs", "p_gap_atr",
)


def specialists_for(stage: str) -> list[SpecialistSpec]:
    return [s for s in SPECIALISTS if stage in s.stages]
