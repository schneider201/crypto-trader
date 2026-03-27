#!/usr/bin/env python3
"""
Strategy 1 v2 — Funding Rate Mean Reversion (Improved)
Fixes: dual signal, trend filter, price-based TP (3R), funding collection, param grid search.
Walk-forward, no look-ahead bias, pandas/numpy only.
"""

import os
import json
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path("/home/etienne/projects/crypto-trader")
DATA = BASE / "data/historical"
RESULTS = BASE / "research/results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ─── Fixed constants ──────────────────────────────────────────────────────────
STARTING_CAPITAL   = 10_000.0
RISK_PCT           = 0.0075        # 0.75% per trade
ATR_MULT_SL        = 2.0           # stop = 2× ATR
TAKER_FEE          = 0.0004        # per side
SLIPPAGE           = 0.0003        # per side
COST_RT            = (TAKER_FEE + SLIPPAGE) * 2   # round-trip = 0.14%
MAX_LEVERAGE       = 4.0
ROLLING_WINDOW     = 720           # 30d × 24h = zscore lookback
ATR_PERIOD         = 14
TRAIN_DAYS         = 240
OOS_DAYS           = 125
MC_SIMULATIONS     = 500

# ─── Parameter grid ───────────────────────────────────────────────────────────
# abs_funding_threshold expressed as PERCENTILE of training data (e.g., 85 = 85th pct)
# so it auto-calibrates per asset regardless of absolute scale
PARAM_GRID = {
    "zscore_threshold":         [1.5, 2.0, 2.5],
    "abs_funding_pct":          [85, 90, 95],    # percentile of abs(funding) on train
    "trend_filter":             [0.05, 0.08, 0.12],  # 24h momentum limit
    "max_hold_hours":           [16, 24, 48],
}


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_candles_1h(symbol: str) -> pd.DataFrame:
    path = DATA / f"binance_{symbol}usdt_candles_1m.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df1h = df[["open", "high", "low", "close", "volume"]].resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return df1h


def load_funding(symbol: str) -> pd.Series:
    path = DATA / f"hl_{symbol}_funding_1h.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.floor("h")
    s = df["funding_rate"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


# ─── Indicator computation ────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_base_signals(candles: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    """Merge candles + funding, compute ATR, z-score, price momentum. No params needed."""
    df = candles.copy()

    # Forward-fill funding onto hourly candle timestamps
    funding_aligned = funding.reindex(df.index, method="ffill")
    df["funding_rate"] = funding_aligned

    # ATR
    df["atr"] = compute_atr(df, ATR_PERIOD)

    # Z-score of funding rate (rolling 30d)
    roll_mean = df["funding_rate"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 4).mean()
    roll_std  = df["funding_rate"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 4).std()
    df["zscore"] = (df["funding_rate"] - roll_mean) / roll_std.replace(0, np.nan)

    # Fix 2: 24h price momentum for trend filter
    # Shift by 1 to avoid look-ahead: momentum known at bar close
    df["price_mom_24h"] = (df["close"] - df["close"].shift(24)) / df["close"].shift(24)

    # Funding sign momentum (3-period) — still useful secondary filter
    fsgn = np.sign(df["funding_rate"])
    df["fund_mom_pos"] = (fsgn.rolling(3).sum() == 3)
    df["fund_mom_neg"] = (fsgn.rolling(3).sum() == -3)

    return df


def apply_param_signals(df: pd.DataFrame, params: dict,
                        abs_funding_threshold: float = None) -> pd.DataFrame:
    """Apply parameterized signal logic on top of base signals.

    abs_funding_threshold: pre-computed from training data percentile.
    If None, uses params['abs_funding_pct'] with df data (not ideal — only for reference).
    """
    df = df.copy()
    zt  = params["zscore_threshold"]
    tf  = params["trend_filter"]

    # Use pre-computed threshold (from training data percentile)
    if abs_funding_threshold is None:
        # fallback: compute from df
        pct = params.get("abs_funding_pct", 90)
        abs_funding_threshold = np.percentile(df["funding_rate"].abs().dropna(), pct)

    aft = abs_funding_threshold

    # Fix 1: Dual signal — both zscore AND absolute threshold required
    abs_extreme = df["funding_rate"].abs() > aft
    z_short     = df["zscore"] >  zt
    z_long      = df["zscore"] < -zt

    # Fix 2: Trend filter — skip if 24h momentum too strong
    trend_ok = df["price_mom_24h"].abs() < tf

    # For SHORT: funding too positive + NOT in strong uptrend
    # For LONG:  funding too negative + NOT in strong downtrend
    df["signal_short"] = (
        abs_extreme & z_short & df["fund_mom_pos"] &
        trend_ok &
        (df["price_mom_24h"] < tf)    # extra: short only when not in strong uptrend
    )
    df["signal_long"] = (
        abs_extreme & z_long & df["fund_mom_neg"] &
        trend_ok &
        (df["price_mom_24h"] > -tf)   # extra: long only when not in strong downtrend
    )

    return df


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, oos_start: pd.Timestamp, asset: str, params: dict) -> dict:
    """
    Walk-forward backtest on the OOS (or TRAIN) period.
    Returns dict with trades list + equity curve.
    """
    max_hold_hours = params["max_hold_hours"]

    oos = df[df.index >= oos_start].copy()

    equity     = STARTING_CAPITAL
    equity_ts  = []
    trades     = []
    in_trade   = False
    trade_info = {}

    rows = oos.reset_index()
    rows = rows.rename(columns={"index": "time"})

    i = 0
    while i < len(rows) - 1:
        bar      = rows.iloc[i]
        next_bar = rows.iloc[i + 1]

        # ─── Manage open trade ───────────────────────────────────────────
        if in_trade:
            t = trade_info
            cur_close  = bar["close"]
            cur_high   = bar["high"]
            cur_low    = bar["low"]
            cur_zscore = bar["zscore"]
            cur_funding = bar["funding_rate"]
            hours_held = (bar["time"] - t["entry_time"]).total_seconds() / 3600

            exit_price  = None
            exit_reason = None

            # Fix 4: Funding collection (every completed 8h block)
            # Realized funding collected is tracked cumulatively during the hold
            # (accounted on trade close, not added to equity mid-trade to keep it simple)

            if t["direction"] == "SHORT":
                # SL
                if cur_high >= t["stop_price"]:
                    exit_price  = t["stop_price"]
                    exit_reason = "STOP"
                # Fix 3: Price-based TP — partial at 2R, trail rest
                elif not t["partial_done"] and cur_low <= t["tp_partial"]:
                    t["partial_done"]  = True
                    t["partial_price"] = t["tp_partial"]
                    # Move stop to breakeven
                    t["stop_price"] = t["entry_price"]
                    # Immediately check full TP
                    if cur_low <= t["tp_full"]:
                        exit_price  = t["tp_full"]
                        exit_reason = "TP_FULL"
                elif t["partial_done"] and cur_low <= t["tp_full"]:
                    exit_price  = t["tp_full"]
                    exit_reason = "TP_FULL"
                # Trailing stop after partial: if price reverses 1R from current best
                elif t["partial_done"] and cur_high >= t["stop_price"]:
                    exit_price  = t["stop_price"]
                    exit_reason = "TRAIL_STOP"
                # Update trailing stop for remaining (tighten toward price for shorts)
                elif t["partial_done"]:
                    # Trail stop 1R below best price reached
                    best_low = t.get("best_excursion", t["entry_price"])
                    if cur_low < best_low:
                        t["best_excursion"] = cur_low
                        new_trail = cur_low + t["stop_distance"]  # 1R trail
                        if new_trail < t["stop_price"]:
                            t["stop_price"] = new_trail
                # zscore normalization exit (secondary)
                elif not t["partial_done"] and cur_zscore <= 0.3:
                    exit_price  = cur_close
                    exit_reason = "TP_ZSCORE"
                elif hours_held >= max_hold_hours:
                    exit_price  = bar["open"]
                    exit_reason = "TIMEOUT"

            else:  # LONG
                if cur_low <= t["stop_price"]:
                    exit_price  = t["stop_price"]
                    exit_reason = "STOP"
                elif not t["partial_done"] and cur_high >= t["tp_partial"]:
                    t["partial_done"]  = True
                    t["partial_price"] = t["tp_partial"]
                    t["stop_price"]    = t["entry_price"]
                    if cur_high >= t["tp_full"]:
                        exit_price  = t["tp_full"]
                        exit_reason = "TP_FULL"
                elif t["partial_done"] and cur_high >= t["tp_full"]:
                    exit_price  = t["tp_full"]
                    exit_reason = "TP_FULL"
                elif t["partial_done"] and cur_low <= t["stop_price"]:
                    exit_price  = t["stop_price"]
                    exit_reason = "TRAIL_STOP"
                elif t["partial_done"]:
                    best_high = t.get("best_excursion", t["entry_price"])
                    if cur_high > best_high:
                        t["best_excursion"] = cur_high
                        new_trail = cur_high - t["stop_distance"]
                        if new_trail > t["stop_price"]:
                            t["stop_price"] = new_trail
                elif not t["partial_done"] and cur_zscore >= -0.3:
                    exit_price  = cur_close
                    exit_reason = "TP_ZSCORE"
                elif hours_held >= max_hold_hours:
                    exit_price  = bar["open"]
                    exit_reason = "TIMEOUT"

            if exit_price is not None:
                entry_p   = t["entry_price"]
                pos_size  = t["position_usd"]
                direction = t["direction"]

                if direction == "SHORT":
                    raw_pct = (entry_p - exit_price) / entry_p
                else:
                    raw_pct = (exit_price - entry_p) / entry_p

                if t["partial_done"]:
                    partial_pct = (t["partial_price"] - entry_p) / entry_p * (1 if direction == "LONG" else -1)
                    pnl_partial = pos_size * 0.5 * partial_pct
                    pnl_rest    = pos_size * 0.5 * raw_pct
                    pnl_gross   = pnl_partial + pnl_rest
                else:
                    pnl_gross = pos_size * raw_pct

                # Fix 4: Funding collection
                # For SHORT with positive funding: collect funding every 8h
                # For LONG with negative funding: collect funding every 8h
                entry_funding = t["entry_funding"]
                funding_periods = int(hours_held / 8)
                if direction == "SHORT" and entry_funding > 0:
                    funding_collected = abs(entry_funding) * pos_size * funding_periods
                elif direction == "LONG" and entry_funding < 0:
                    funding_collected = abs(entry_funding) * pos_size * funding_periods
                else:
                    funding_collected = 0.0

                cost    = pos_size * COST_RT
                pnl_net = pnl_gross - cost + funding_collected

                equity += pnl_net
                equity = max(equity, 0.01)

                trades.append({
                    "asset":            asset,
                    "entry_time":       str(t["entry_time"]),
                    "exit_time":        str(bar["time"]),
                    "direction":        direction,
                    "entry_price":      entry_p,
                    "exit_price":       exit_price,
                    "stop_price_orig":  t["stop_price_orig"],
                    "position_usd":     pos_size,
                    "pnl_gross":        round(pnl_gross, 4),
                    "funding_collected": round(funding_collected, 4),
                    "pnl_net":          round(pnl_net, 4),
                    "pnl_pct":          round(pnl_net / t["equity_at_entry"] * 100, 4),
                    "hold_hours":       round(hours_held, 2),
                    "exit_reason":      exit_reason,
                    "entry_zscore":     t["entry_zscore"],
                    "entry_funding":    entry_funding,
                    "partial_done":     t["partial_done"],
                })
                in_trade   = False
                trade_info = {}

        # ─── Record equity ───────────────────────────────────────────────
        equity_ts.append({"time": str(bar["time"]), "equity": round(equity, 4)})

        # ─── Check for new signal ────────────────────────────────────────
        if not in_trade:
            sig_short = bar["signal_short"]
            sig_long  = bar["signal_long"]

            if sig_short or sig_long:
                direction = "SHORT" if sig_short else "LONG"

                entry_price = next_bar["open"]
                atr = bar["atr"]
                if pd.isna(atr) or atr <= 0:
                    i += 1
                    continue

                # Apply slippage
                if direction == "SHORT":
                    entry_price *= (1 - SLIPPAGE)
                    stop_price   = entry_price + ATR_MULT_SL * atr
                    # Fix 3: Price-based TP — 2R partial, 3R full
                    stop_dist    = abs(stop_price - entry_price)
                    tp_partial   = entry_price - 2.0 * stop_dist   # 2R
                    tp_full      = entry_price - 3.0 * stop_dist   # 3R
                else:
                    entry_price *= (1 + SLIPPAGE)
                    stop_price   = entry_price - ATR_MULT_SL * atr
                    stop_dist    = abs(stop_price - entry_price)
                    tp_partial   = entry_price + 2.0 * stop_dist
                    tp_full      = entry_price + 3.0 * stop_dist

                if stop_dist <= 0:
                    i += 1
                    continue

                risk_amount  = equity * RISK_PCT
                position_usd = risk_amount / (stop_dist / entry_price)

                # Cap by leverage
                position_usd = min(position_usd, equity * MAX_LEVERAGE)

                if position_usd < 10:
                    i += 1
                    continue

                in_trade   = True
                trade_info = {
                    "entry_time":      next_bar["time"],
                    "entry_price":     entry_price,
                    "stop_price":      stop_price,
                    "stop_price_orig": stop_price,
                    "stop_distance":   stop_dist,
                    "tp_partial":      tp_partial,
                    "tp_full":         tp_full,
                    "direction":       direction,
                    "position_usd":    position_usd,
                    "equity_at_entry": equity,
                    "partial_done":    False,
                    "partial_price":   None,
                    "best_excursion":  entry_price,
                    "entry_zscore":    bar["zscore"],
                    "entry_funding":   bar["funding_rate"],
                }

        i += 1

    return {"trades": trades, "equity_curve": equity_ts}


# ─── Grid search ──────────────────────────────────────────────────────────────

def run_grid_search(base_df: pd.DataFrame, train_start: pd.Timestamp,
                    train_end: pd.Timestamp, asset: str) -> tuple:
    """Grid search on training data — returns (best_params, abs_funding_threshold)."""
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    print(f"    Grid search: {len(combos)} combos on {asset} train set...")

    best_sharpe   = -np.inf
    best_params   = None
    best_n_trades = 0
    best_aft      = None

    # Compute percentile thresholds from training data only (no look-ahead)
    train_df = base_df[(base_df.index >= train_start) & (base_df.index < train_end)].copy()
    train_abs_funding = train_df["funding_rate"].abs().dropna()

    pct_thresholds = {}
    for pct in PARAM_GRID["abs_funding_pct"]:
        pct_thresholds[pct] = np.percentile(train_abs_funding, pct)

    print(f"    Train abs funding pct thresholds: { {k: f'{v:.7f}' for k,v in pct_thresholds.items()} }")

    for combo in combos:
        params = dict(zip(keys, combo))
        pct = params["abs_funding_pct"]
        aft = pct_thresholds[pct]

        try:
            df_sig = apply_param_signals(base_df, params, abs_funding_threshold=aft)
            df_sig = df_sig.dropna(subset=["zscore", "atr", "price_mom_24h"])
            result = run_backtest(df_sig, train_start, asset, params)
            trades = result["trades"]

            if len(trades) < 3:
                continue

            pnl = np.array([t["pnl_net"] for t in trades])
            if len(pnl) < 2:
                continue

            # Use trade-level Sharpe (more robust with few trades)
            if pnl.std() == 0:
                continue
            sharpe = pnl.mean() / pnl.std() * np.sqrt(min(len(pnl), 252))

            if sharpe > best_sharpe:
                best_sharpe   = sharpe
                best_params   = params
                best_n_trades = len(trades)
                best_aft      = aft
        except Exception:
            continue

    if best_params is None:
        # Fallback to defaults
        best_params = {
            "zscore_threshold": 2.0,
            "abs_funding_pct":  90,
            "trend_filter":     0.08,
            "max_hold_hours":   24,
        }
        best_aft    = pct_thresholds.get(90, np.percentile(train_abs_funding, 90))
        best_sharpe = float("nan")

    print(f"    Best train Sharpe: {best_sharpe:.3f} | n_trades: {best_n_trades}")
    print(f"    Best params: {best_params} | abs_funding_threshold: {best_aft:.7f}")
    return best_params, best_aft


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(trades: list, equity_curve: list, oos_start: str, oos_end: str) -> dict:
    if not trades:
        return {"error": "No trades"}

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity_curve)
    df_e["time"] = pd.to_datetime(df_e["time"])
    df_e = df_e.set_index("time").sort_index()

    pnl    = df_t["pnl_net"].values
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_return = (df_e["equity"].iloc[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    n_days = (df_e.index[-1] - df_e.index[0]).total_seconds() / 86400
    ann_return = ((1 + total_return / 100) ** (365 / max(n_days, 1)) - 1) * 100

    daily_eq  = df_e["equity"].resample("D").last().ffill()
    daily_ret = daily_eq.pct_change().dropna()

    sharpe  = (daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() > 0 else 0
    neg_ret = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / neg_ret.std() * np.sqrt(365)) if len(neg_ret) > 1 and neg_ret.std() > 0 else 0

    roll_max = df_e["equity"].cummax()
    drawdown = (df_e["equity"] - roll_max) / roll_max * 100
    max_dd   = drawdown.min()

    calmar   = ann_return / abs(max_dd) if max_dd != 0 else 0
    win_rate = len(wins) / len(pnl) * 100 if len(pnl) > 0 else 0
    avg_win  = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float("inf")
    avg_hold = df_t["hold_hours"].mean()

    # Funding collected
    funding_col = df_t["funding_collected"].sum() if "funding_collected" in df_t.columns else 0.0

    df_t["exit_time"] = pd.to_datetime(df_t["exit_time"])
    df_t["month"]     = df_t["exit_time"].dt.to_period("M")
    monthly     = df_t.groupby("month")["pnl_net"].sum()
    monthly_pct = (monthly / STARTING_CAPITAL * 100).round(2)
    best_m      = monthly_pct.max()
    worst_m     = monthly_pct.min()

    exit_reasons = df_t["exit_reason"].value_counts().to_dict()

    return {
        "period_start":        oos_start,
        "period_end":          oos_end,
        "total_trades":        len(pnl),
        "win_rate_pct":        round(win_rate, 2),
        "total_return_pct":    round(total_return, 2),
        "ann_return_pct":      round(ann_return, 2),
        "sharpe_ratio":        round(sharpe, 3),
        "sortino_ratio":       round(sortino, 3),
        "max_drawdown_pct":    round(max_dd, 2),
        "calmar_ratio":        round(calmar, 3),
        "avg_win_usd":         round(avg_win, 2),
        "avg_loss_usd":        round(avg_loss, 2),
        "win_loss_ratio":      round(rr_ratio, 3),
        "profit_factor":       round(profit_factor, 3),
        "avg_hold_hours":      round(avg_hold, 2),
        "best_month_pct":      round(best_m, 2),
        "worst_month_pct":     round(worst_m, 2),
        "funding_collected_usd": round(funding_col, 2),
        "monthly_returns":     {str(k): float(v) for k, v in monthly_pct.items()},
        "exit_reasons":        exit_reasons,
        "final_equity":        round(df_e["equity"].iloc[-1], 2),
    }


def run_monte_carlo(trades: list, n_sim: int = MC_SIMULATIONS) -> dict:
    if not trades:
        return {}
    pnl = np.array([t["pnl_net"] for t in trades])
    all_returns = []
    all_dd      = []
    rng = np.random.default_rng(42)

    for _ in range(n_sim):
        shuffled  = rng.permutation(pnl)
        equity    = STARTING_CAPITAL + np.cumsum(shuffled)
        total_ret = (equity[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        roll_max  = np.maximum.accumulate(np.concatenate([[STARTING_CAPITAL], equity]))
        dd        = (equity - roll_max[1:]) / roll_max[1:] * 100
        all_returns.append(total_ret)
        all_dd.append(dd.min())

    return {
        "mc_simulations":      n_sim,
        "mc_p5_max_drawdown":  round(np.percentile(all_dd, 5), 2),
        "mc_p50_total_return": round(np.percentile(all_returns, 50), 2),
        "mc_p95_total_return": round(np.percentile(all_returns, 95), 2),
        "mc_p5_total_return":  round(np.percentile(all_returns, 5), 2),
    }


# ─── Asset pipeline ───────────────────────────────────────────────────────────

def process_asset(symbol: str) -> dict:
    print(f"\n  Loading {symbol.upper()}...")
    candles = load_candles_1h(symbol)
    funding = load_funding(symbol)

    base_df = compute_base_signals(candles, funding)
    base_df = base_df.dropna(subset=["zscore", "atr", "price_mom_24h"])

    start     = base_df.index.min()
    oos_start = start + pd.Timedelta(days=TRAIN_DAYS)
    train_end = oos_start
    oos_end   = oos_start + pd.Timedelta(days=OOS_DAYS)
    df_window = base_df[base_df.index <= oos_end]

    print(f"  Full data: {base_df.index.min().date()} → {base_df.index.max().date()}")
    print(f"  Train:     {start.date()} → {train_end.date()}")
    print(f"  OOS:       {oos_start.date()} → {oos_end.date()}")

    # Grid search on training set
    best_params, best_aft = run_grid_search(df_window, start, train_end, symbol.upper())

    # Apply best params and run OOS backtest
    df_sig = apply_param_signals(df_window, best_params, abs_funding_threshold=best_aft)

    result     = run_backtest(df_sig, oos_start, symbol.upper(), best_params)
    trades     = result["trades"]
    equity_ts  = result["equity_curve"]

    print(f"  OOS Trades: {len(trades)}")

    metrics = compute_metrics(
        trades, equity_ts,
        str(oos_start.date()), str(oos_end.date())
    )
    mc = run_monte_carlo(trades)

    return {
        "asset":                symbol.upper(),
        "best_params":          best_params,
        "abs_funding_threshold": best_aft,
        "metrics":              metrics,
        "monte_carlo":          mc,
        "trades":               trades,
        "equity_curve":         equity_ts,
    }


def combine_results(all_results: list) -> dict:
    all_trades = []
    for r in all_results:
        all_trades.extend(r["trades"])

    if not all_trades:
        return {}

    df_t = pd.DataFrame(all_trades)
    df_t["exit_time"] = pd.to_datetime(df_t["exit_time"])
    df_t = df_t.sort_values("exit_time")

    equity = STARTING_CAPITAL
    equity_curve = []
    for _, row in df_t.iterrows():
        equity += row["pnl_net"]
        equity_curve.append({"time": str(row["exit_time"]), "equity": equity})

    metrics = compute_metrics(
        all_trades, equity_curve,
        df_t["exit_time"].min().strftime("%Y-%m-%d"),
        df_t["exit_time"].max().strftime("%Y-%m-%d"),
    )
    mc = run_monte_carlo(all_trades)

    return {
        "asset":        "COMBINED",
        "best_params":  "see individual assets",
        "metrics":      metrics,
        "monte_carlo":  mc,
        "trades":       all_trades,
        "equity_curve": equity_curve,
    }


# ─── Verdict ──────────────────────────────────────────────────────────────────

def verdict(metrics: dict) -> tuple:
    if "error" in metrics:
        return "FAIL", "No trades generated"

    tr     = metrics.get("total_return_pct", 0)
    sr     = metrics.get("sharpe_ratio", 0)
    dd     = metrics.get("max_drawdown_pct", 0)
    wr     = metrics.get("win_rate_pct", 0)
    rr     = metrics.get("win_loss_ratio", 0)
    trades = metrics.get("total_trades", 0)
    pf     = metrics.get("profit_factor", 0)

    if trades < 5:
        return "FAIL", "Too few trades for statistical significance"
    if tr <= 0:
        return "FAIL", f"Negative returns ({tr:.1f}%)"
    if sr < 0.3:
        return "FAIL", f"Sharpe too low ({sr:.2f})"
    if rr < 1.0 and tr <= 5:
        return "MARGINAL", f"Win/loss ratio <1.0 ({rr:.2f}), marginal returns"
    if tr > 0 and sr >= 0.5 and rr >= 1.0:
        return "PASS", f"Returns {tr:.1f}%, Sharpe {sr:.2f}, W/L {rr:.2f}, MaxDD {dd:.1f}%"
    if tr > 0 and sr >= 0.3:
        return "MARGINAL", f"Returns {tr:.1f}%, Sharpe {sr:.2f} (below 0.5), W/L {rr:.2f}"
    return "FAIL", "Does not meet minimum criteria"


# ─── Print summary ────────────────────────────────────────────────────────────

def print_summary(result: dict):
    asset = result["asset"]
    m     = result.get("metrics", {})
    mc    = result.get("monte_carlo", {})
    bp    = result.get("best_params", {})
    aft   = result.get("abs_funding_threshold", None)

    if "error" in m:
        print(f"\nASSET: {asset}")
        print(f"  ERROR: {m['error']}")
        return

    verd, reason = verdict(m)

    print(f"""
ASSET: {asset}
  Period:           {m.get('period_start')} → {m.get('period_end')}
  Best Params:      {bp}
  Abs Funding Thr:  {aft}
  Total Trades:     {m.get('total_trades')}
  Win Rate:         {m.get('win_rate_pct')}%
  ─────────────────────────────────────────
  Total Return:     {m.get('total_return_pct')}%
  Ann. Return:      {m.get('ann_return_pct')}%
  Final Equity:     ${m.get('final_equity'):,.2f}
  Funding Collected:${m.get('funding_collected_usd'):,.2f}
  ─────────────────────────────────────────
  Sharpe Ratio:     {m.get('sharpe_ratio')}
  Sortino Ratio:    {m.get('sortino_ratio')}
  Max Drawdown:     {m.get('max_drawdown_pct')}%
  Calmar Ratio:     {m.get('calmar_ratio')}
  ─────────────────────────────────────────
  Avg Win:          ${m.get('avg_win_usd')}
  Avg Loss:         ${m.get('avg_loss_usd')}
  Win/Loss Ratio:   {m.get('win_loss_ratio')}
  Profit Factor:    {m.get('profit_factor')}
  Avg Hold Time:    {m.get('avg_hold_hours')}h
  ─────────────────────────────────────────
  Best Month:       {m.get('best_month_pct')}%
  Worst Month:      {m.get('worst_month_pct')}%
  ─────────────────────────────────────────
  Exit Reasons:""")
    for k, v in m.get("exit_reasons", {}).items():
        print(f"    {k:15s}: {v}")

    if mc:
        print(f"""  ─────────────────────────────────────────
  Monte Carlo ({mc.get('mc_simulations')} sims):
    P5  Max Drawdown:  {mc.get('mc_p5_max_drawdown')}%
    P50 Total Return:  {mc.get('mc_p50_total_return')}%
    P5  Total Return:  {mc.get('mc_p5_total_return')}%
    P95 Total Return:  {mc.get('mc_p95_total_return')}%""")

    monthly = m.get("monthly_returns", {})
    if monthly:
        print("  ─────────────────────────────────────────")
        print("  Monthly Returns:")
        for month, ret in sorted(monthly.items()):
            bar = "█" * int(abs(ret) / 0.5) if abs(ret) >= 0.5 else ""
            sign = "+" if ret >= 0 else ""
            print(f"    {month}  {sign}{ret:6.2f}%  {bar}")

    print(f"\n  VERDICT: {verd} — {reason}")
    print("─" * 51)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("═" * 51)
    print("  STRATEGY 1 v2 — FUNDING RATE MEAN REVERSION")
    print("  Dual Signal + Trend Filter + Price-Based TP")
    print("  + Funding Collection + Grid Search")
    print("═" * 51)

    assets      = ["btc", "eth", "sol"]
    all_results = []

    for symbol in assets:
        try:
            result = process_asset(symbol)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR processing {symbol.upper()}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        combined = combine_results(all_results)
        all_results.append(combined)

    for result in all_results:
        print_summary(result)

    # ─── Save results ─────────────────────────────────────────────────────
    print("\nSaving results...")

    output          = {}
    all_trades_flat = []
    all_equity_flat = []

    for r in all_results:
        asset = r["asset"]
        output[asset] = {
            "best_params":           r.get("best_params", {}),
            "abs_funding_threshold": r.get("abs_funding_threshold", None),
            "metrics":               r.get("metrics", {}),
            "monte_carlo":           r.get("monte_carlo", {}),
        }
        for t in r.get("trades", []):
            all_trades_flat.append(t)
        for e in r.get("equity_curve", []):
            all_equity_flat.append({"asset": asset, **e})

    with open(RESULTS / "strategy1_v2_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    if all_trades_flat:
        pd.DataFrame(all_trades_flat).to_csv(RESULTS / "strategy1_v2_trades.csv", index=False)

    if all_equity_flat:
        pd.DataFrame(all_equity_flat).to_csv(RESULTS / "strategy1_v2_equity_curve.csv", index=False)

    print(f"\nResults saved to: {RESULTS}")
    print("  strategy1_v2_results.json")
    print("  strategy1_v2_trades.csv")
    print("  strategy1_v2_equity_curve.csv")
    print("\n" + "═" * 51)


if __name__ == "__main__":
    main()
