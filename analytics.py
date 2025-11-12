
# analytics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd


# ---------- Utilities ----------

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize minimal columns and dtypes."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "amount", "category", "type"])

    t = df.copy()

    # date
    if "date" not in t.columns:
        t["date"] = pd.Timestamp.today().date()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])

    # amount
    if "amount" not in t.columns:
        t["amount"] = 0.0
    t["amount"] = pd.to_numeric(t["amount"], errors="coerce").fillna(0.0)

    # type
    if "type" not in t.columns:
        t["type"] = "expense"
    t["type"] = t["type"].astype(str).str.lower().str.strip()
    t.loc[~t["type"].isin(["income", "expense"]), "type"] = "expense"

    # category
    if "category" not in t.columns:
        t["category"] = "uncategorized"
    t["category"] = (
        t["category"]
        .fillna("uncategorized")
        .astype(str)             # <-- avoids Timestamp/other non-strings
        .str.strip()
        .replace({"": "uncategorized"})
        .str.lower()
    )

    # Ensure sorted by date
    t = t.sort_values("date").reset_index(drop=True)
    return t


def _daterange_days(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    return list(days)


# ---------- Public API ----------

def no_spend_streak(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Current and longest streak of days with ZERO expense.
    """
    t = _sanitize_df(df)
    if t.empty:
        return 0, 0

    # Daily expense sum
    exp = (
        t.loc[t["type"] == "expense"]
        .groupby(t["date"].dt.date)["amount"]
        .sum()
    )

    start = t["date"].min().normalize()
    end = pd.Timestamp.today().normalize()
    days = pd.date_range(start, end, freq="D")

    daily_zero = []
    for d in days:
        spent = float(exp.get(d.date(), 0.0))
        daily_zero.append(spent == 0.0)

    # Compute runs
    longest = run = 0
    for flag in daily_zero:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    # Current run from the end
    curr = 0
    for flag in reversed(daily_zero):
        if flag:
            curr += 1
        else:
            break

    return int(curr), int(longest)


def detect_trend_spikes(df: pd.DataFrame, window: str = "30D") -> List[Dict[str, float]]:
    """
    Compare expense per-category in the last `window` vs the previous `window`.
    Returns sorted list of spikes: [{'category': 'x', 'delta': 123.0, 'pct': 45.6}, ...]
    """
    t = _sanitize_df(df)
    if t.empty:
        return []

    end = t["date"].max().normalize()
    win = pd.Timedelta(window)
    mid = end - win
    start_prev = mid - win

    exp = t[t["type"] == "expense"][["date", "category", "amount"]].copy()

    recent = exp[(exp["date"] > mid) & (exp["date"] <= end)]
    prev = exp[(exp["date"] > start_prev) & (exp["date"] <= mid)]

    g_recent = recent.groupby("category")["amount"].sum()
    g_prev = prev.groupby("category")["amount"].sum()

    cats = set(g_recent.index).union(g_prev.index)
    out: List[Dict[str, float]] = []
    for c in cats:
        r = float(g_recent.get(c, 0.0))
        p = float(g_prev.get(c, 0.0))
        delta = r - p
        pct = (delta / p * 100.0) if p > 0 else (100.0 if r > 0 else 0.0)
        out.append({"category": c, "delta": round(delta, 2), "pct": round(pct, 1)})

    out.sort(key=lambda x: (x["delta"], x["pct"]), reverse=True)
    return out[:10]


def forecast_next_month(df: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """
    Simple per-category expense forecast using daily mean over lookback window,
    scaled to next month (30 days).
    """
    t = _sanitize_df(df)
    if t.empty:
        return pd.DataFrame(columns=["category", "forecasted_expense"])

    end = t["date"].max().normalize()
    start = max(t["date"].min().normalize(), end - pd.Timedelta(days=lookback_days - 1))

    exp = t[(t["type"] == "expense") & (t["date"].between(start, end))].copy()
    if exp.empty:
        return pd.DataFrame(columns=["category", "forecasted_expense"])

    exp["day"] = exp["date"].dt.date
    daily = exp.groupby(["category", "day"])["amount"].sum().reset_index()
    daily_mean = daily.groupby("category")["amount"].mean()

    next_month_days = 30  # keep constant for UX
    fc = (daily_mean * next_month_days).round(2).sort_values(ascending=False)
    return fc.reset_index().rename(columns={"amount": "forecasted_expense"})


def auto_allocate_budget(df: pd.DataFrame, savings_target_pct: float = 0.15) -> Dict[str, float]:
    """
    Allocate a budget per category:
    - Reserve `savings_target_pct` of income for savings (if income present).
    - Allocate remaining expense budget by historical expense proportions.
    """
    t = _sanitize_df(df)
    if t.empty:
        return {}

    total_income = float(t.loc[t["type"] == "income", "amount"].sum())
    total_expense = float(t.loc[t["type"] == "expense", "amount"].sum())

    savings_pool = round(max(0.0, total_income * float(savings_target_pct)), 0) if total_income > 0 else 0.0
    expense_budget_pool = max(0.0, total_expense - savings_pool)

    exp = t.loc[t["type"] == "expense", ["category", "amount"]]
    if exp.empty or expense_budget_pool <= 0:
        out: Dict[str, float] = {}
        if savings_pool:
            out["__meta__savings_reserved__"] = float(savings_pool)
        return out

    by_cat = exp.groupby("category", dropna=False)["amount"].sum()
    denom = float(by_cat.sum())
    if denom <= 0:
        out: Dict[str, float] = {}
        if savings_pool:
            out["__meta__savings_reserved__"] = float(savings_pool)
        return out

    proportions = (by_cat / denom).fillna(0.0)

    budget_allocation: Dict[str, float] = {}
    for cat, p in proportions.items():
        name = str(cat).title()  # <-- fixes Timestamp/float categories
        budget_allocation[name] = round(float(p) * expense_budget_pool, 0)

    # Sort largest-first
    budget_allocation = dict(sorted(budget_allocation.items(), key=lambda kv: kv[1], reverse=True))
    if savings_pool:
        budget_allocation["__meta__savings_reserved__"] = float(savings_pool)
    return budget_allocation


def compute_fin_health_score(df: pd.DataFrame, budgets: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """
    Composite score (0–100) with factors:
      - savings_rate: income>0 ? (income-expense)/income : 0
      - spend_ratio:   1 - min(1, expense/(income+1e-9))
      - budget_adherence: 1 - mean(max(0, (actual-budget)/max(1,budget)))
      - no-spend streaks
    """
    t = _sanitize_df(df)
    income = float(t.loc[t["type"] == "income", "amount"].sum())
    expense = float(t.loc[t["type"] == "expense", "amount"].sum())
    net = income - expense

    # Savings rate (clamped 0..1)
    savings_rate = 0.0 if income <= 0 else max(0.0, min(1.0, net / income))

    # Spend ratio (lower expense/income is better)
    spend_ratio = 1.0 - (expense / income if income > 0 else 1.0)
    spend_ratio = max(0.0, min(1.0, spend_ratio))

    # Budget adherence
    adherence = 1.0
    details: Dict[str, Dict[str, float]] = {}
    if budgets:
        # Normalize budget category keys to lowercase for matching
        bnorm = {str(k).lower(): float(v) for k, v in budgets.items() if not str(k).startswith("__meta__")}
        actual = (
            t[t["type"] == "expense"]
            .groupby("category")["amount"]
            .sum()
            .to_dict()
        )
        penalties = []
        for k, b in bnorm.items():
            a = float(actual.get(k, 0.0))
            # Over-budget penalty ratio [0..inf)
            over = max(0.0, (a - b) / (b if b > 0 else max(1.0, a)))
            penalties.append(over)
            details[k] = {"budget": b, "actual": a, "over_ratio": round(over, 3)}
        mean_over = float(np.mean(penalties)) if penalties else 0.0
        adherence = max(0.0, 1.0 - min(1.0, mean_over))

    # No-spend streaks
    curr_ns, longest_ns = no_spend_streak(t)

    # Weighted score
    # why: stable, budget-aware, and not too punitive when income is tiny.
    w_savings = 0.45
    w_spend = 0.25
    w_adherence = 0.30
    raw = (w_savings * savings_rate) + (w_spend * spend_ratio) + (w_adherence * adherence)

    # Streak bonus (small)
    streak_bonus = min(5.0, curr_ns * 0.2 + longest_ns * 0.05)
    score = int(round(max(0.0, min(100.0, raw * 100.0 + streak_bonus)), 0))

    return {
        "score": score,
        "income": round(income, 2),
        "expense": round(expense, 2),
        "net": round(net, 2),
        "factors": {
            "savings_rate": round(savings_rate, 3),
            "spend_ratio": round(spend_ratio, 3),
            "budget_adherence": round(adherence, 3),
            "no_spend_streak": int(curr_ns),
            "longest_no_spend": int(longest_ns),
            "budget_details": details,  # per-category
        },
    }
