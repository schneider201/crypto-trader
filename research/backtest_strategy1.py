#!/usr/bin/env python3
"""
Strategy 1 — Funding Rate Mean Reversion Backtest
Walk-forward, no look-ahead bias, pandas/numpy only.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path("/home/etienne/projects/crypto-trader")
DATA = BASE / "data/historical"
RESULTS = BASE / "research/results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────
STARTING_CAPITAL   = 10_000.0
RISK_PCT           = 0.0075        # 0.75% per trade
ATR_MULT_SL        = 2.0          # stop = 2× ATR
ZSCORE_ENTRY       = 2.0
ZSCORE_TP          = 0.5
MAX_HOLD_HOURS     = 24
TAKER_FEE          = 0.0004       # per side
SLIPPAGE           = 0.0003       # per side
COST_RT            = (TAKER_FEE + SLIPPAGE) * 2   # round-trip = 0.14%
MAX_LEVERAGE       = 4.0
ROLLING_WINDOW     = 720          # 30d × 24h
ATR_PERIOD         = 14
MOMENTUM_PERIODS   = 3
TRAIN_DAYS         = 240
OOS_DAYS           = 125
MC_SIMULATIONS     = 500


def load_candles_1h(symbol: str) -> pd.DataFrame:
    """Load 1m candles and resample to 1h OHLCV."""
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
    """Load hourly funding rates."""
    path = DATA / f"hl_{symbol}_funding_1h.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    # Normalise timestamps to hour-start
    df.index = df.index.floor("h")
    s = df["funding_rate"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — no look-ahead."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_signals(candles: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    """Merge candles + funding, compute all signals. No look-ahead."""
    df = candles.copy()

    # Forward-fill funding onto hourly candle timestamps
    # Align: reindex funding to candle index, then ffill
    funding_aligned = funding.reindex(df.index, method="ffill")
    df["funding_rate"] = funding_aligned

    # ATR
    df["atr"] = compute_atr(df, ATR_PERIOD)

    # Z-score of funding rate (rolling 30d = 720h)
    roll_mean = df["funding_rate"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 4).mean()
    roll_std  = df["funding_rate"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 4).std()
    df["zscore"] = (df["funding_rate"] - roll_mean) / roll_std.replace(0, np.nan)

    # Funding momentum: last 3 epochs all > 0 (positive) or all < 0 (negative)
    # "epoch" here = funding sign
    fsgn = np.sign(df["funding_rate"])
    df["fund_mom_pos"] = (fsgn.rolling(MOMENTUM_PERIODS).sum() ==  MOMENTUM_PERIODS)
    df["fund_mom_neg"] = (fsgn.rolling(MOMENTUM_PERIODS).sum() == -MOMENTUM_PERIODS)

    # OI proxy: price momentum (close / close_N_ago - 1)
    df["price_mom"] = df["close"].pct_change(3)

    # Signal: zscore extreme + momentum confirms
    # SHORT: zscore > 2 + last 3 funding positive (longs overcrowded)
    df["signal_short"] = (
        (df["zscore"] > ZSCORE_ENTRY) &
        df["fund_mom_pos"]
    )
    # LONG: zscore < -2 + last 3 funding negative (shorts overcrowded)
    df["signal_long"] = (
        (df["zscore"] < -ZSCORE_ENTRY) &
        df["fund_mom_neg"]
    )

    return df


def run_backtest(df: pd.DataFrame, oos_start: pd.Timestamp, asset: str) -> dict:
    """
    Walk-forward backtest on the OOS period.
    Returns dict with trades list + equity curve.
    """
    # Only trade OOS period
    oos = df[df.index >= oos_start].copy()

    equity     = STARTING_CAPITAL
    equity_ts  = []
    trades     = []
    in_trade   = False
    trade_info = {}

    # We iterate bar-by-bar; signals are computed on the *current* bar
    # and the trade enters on the *next* bar's open → no look-ahead
    rows = oos.reset_index()

    i = 0
    while i < len(rows) - 1:
        bar     = rows.iloc[i]
        next_bar = rows.iloc[i + 1]

        # ─── Manage open trade ───────────────────────────────────────────
        if in_trade:
            t = trade_info

            # Check exit conditions on CURRENT bar
            cur_close  = bar["close"]
            cur_high   = bar["high"]
            cur_low    = bar["low"]
            cur_zscore = bar["zscore"]
            hours_held = (bar["time"] - t["entry_time"]).total_seconds() / 3600

            exit_price = None
            exit_reason = None

            # Determine worst-case fill for stop (bar high/low)
            if t["direction"] == "SHORT":
                # SL hit if bar high ≥ stop
                if cur_high >= t["stop_price"]:
                    exit_price  = t["stop_price"]
                    exit_reason = "STOP"
                # TP first partial at 2R
                elif not t["partial_done"] and cur_low <= t["tp2r"]:
                    # Take 50% at 2R
                    t["partial_done"] = True
                    t["partial_price"] = t["tp2r"]
                    # Don't close yet, move stop to breakeven
                    t["stop_price"] = t["entry_price"]
                    # Check full TP (zscore reverted)
                    if cur_zscore <= ZSCORE_TP or cur_low <= t["tp_full"]:
                        exit_price  = min(cur_close, t["tp_full"])
                        exit_reason = "TP_FULL"
                elif cur_zscore <= ZSCORE_TP:
                    exit_price  = cur_close
                    exit_reason = "TP_ZSCORE"
                elif hours_held >= MAX_HOLD_HOURS:
                    exit_price  = bar["open"]  # exit at open of expiry bar
                    exit_reason = "TIMEOUT"

            else:  # LONG
                if cur_low <= t["stop_price"]:
                    exit_price  = t["stop_price"]
                    exit_reason = "STOP"
                elif not t["partial_done"] and cur_high >= t["tp2r"]:
                    t["partial_done"] = True
                    t["partial_price"] = t["tp2r"]
                    t["stop_price"] = t["entry_price"]
                    if cur_zscore >= -ZSCORE_TP or cur_high >= t["tp_full"]:
                        exit_price  = max(cur_close, t["tp_full"])
                        exit_reason = "TP_FULL"
                elif cur_zscore >= -ZSCORE_TP:
                    exit_price  = cur_close
                    exit_reason = "TP_ZSCORE"
                elif hours_held >= MAX_HOLD_HOURS:
                    exit_price  = bar["open"]
                    exit_reason = "TIMEOUT"

            if exit_price is not None:
                # Calculate PnL
                entry_p = t["entry_price"]
                pos_size = t["position_usd"]   # USD notional
                direction = t["direction"]

                if direction == "SHORT":
                    raw_pct_chg = (entry_p - exit_price) / entry_p
                else:
                    raw_pct_chg = (exit_price - entry_p) / entry_p

                # Partial exit (50% at 2R)
                if t["partial_done"]:
                    partial_pct = (t["partial_price"] - entry_p) / entry_p * (1 if direction == "LONG" else -1)
                    pnl_partial = pos_size * 0.5 * partial_pct
                    pnl_rest    = pos_size * 0.5 * raw_pct_chg
                    pnl_gross   = pnl_partial + pnl_rest
                else:
                    pnl_gross   = pos_size * raw_pct_chg

                # Costs
                cost = pos_size * COST_RT
                pnl_net = pnl_gross - cost

                equity += pnl_net
                equity = max(equity, 0.01)  # don't go negative

                hold_h = hours_held
                trades.append({
                    "asset":       asset,
                    "entry_time":  str(t["entry_time"]),
                    "exit_time":   str(bar["time"]),
                    "direction":   direction,
                    "entry_price": entry_p,
                    "exit_price":  exit_price,
                    "stop_price":  t["stop_price_orig"],
                    "position_usd": pos_size,
                    "pnl_net":     round(pnl_net, 4),
                    "pnl_pct":     round(pnl_net / t["equity_at_entry"] * 100, 4),
                    "hold_hours":  round(hold_h, 2),
                    "exit_reason": exit_reason,
                    "entry_zscore": t["entry_zscore"],
                })
                in_trade = False
                trade_info = {}

        # ─── Record equity ───────────────────────────────────────────────
        equity_ts.append({"time": str(bar["time"]), "equity": round(equity, 4)})

        # ─── Check for new signal (only if not in trade) ─────────────────
        if not in_trade:
            signal_short = bar["signal_short"]
            signal_long  = bar["signal_long"]

            if signal_short or signal_long:
                direction = "SHORT" if signal_short else "LONG"

                # Entry at next bar open
                entry_price = next_bar["open"]
                atr = bar["atr"]
                if pd.isna(atr) or atr <= 0:
                    i += 1
                    continue

                # Apply slippage to entry
                if direction == "SHORT":
                    entry_price *= (1 - SLIPPAGE)  # better fill for short
                    stop_price   = entry_price + ATR_MULT_SL * atr
                    # TP at 4× ATR (full), 2R partial
                    tp_full  = entry_price - 4 * atr
                    tp2r     = entry_price - ATR_MULT_SL * atr  # 2R = where stop was
                else:
                    entry_price *= (1 + SLIPPAGE)
                    stop_price   = entry_price - ATR_MULT_SL * atr
                    tp_full  = entry_price + 4 * atr
                    tp2r     = entry_price + ATR_MULT_SL * atr

                stop_distance = abs(entry_price - stop_price)
                if stop_distance <= 0:
                    i += 1
                    continue

                risk_amount = equity * RISK_PCT
                position_usd = risk_amount / (stop_distance / entry_price)

                # Cap by leverage
                max_position = equity * MAX_LEVERAGE
                position_usd = min(position_usd, max_position)

                # Skip if position too tiny
                if position_usd < 10:
                    i += 1
                    continue

                in_trade = True
                trade_info = {
                    "entry_time":       next_bar["time"],
                    "entry_price":      entry_price,
                    "stop_price":       stop_price,
                    "stop_price_orig":  stop_price,
                    "tp_full":          tp_full,
                    "tp2r":             tp2r,
                    "direction":        direction,
                    "position_usd":     position_usd,
                    "equity_at_entry":  equity,
                    "partial_done":     False,
                    "partial_price":    None,
                    "entry_zscore":     bar["zscore"],
                }

        i += 1

    return {"trades": trades, "equity_curve": equity_ts}


def compute_metrics(trades: list, equity_curve: list, oos_start: str, oos_end: str) -> dict:
    """Compute full performance metrics."""
    if not trades:
        return {"error": "No trades"}

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity_curve)
    df_e["time"] = pd.to_datetime(df_e["time"])
    df_e = df_e.set_index("time").sort_index()

    pnl   = df_t["pnl_net"].values
    wins  = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_return = (df_e["equity"].iloc[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    n_days = (df_e.index[-1] - df_e.index[0]).total_seconds() / 86400
    ann_return = ((1 + total_return / 100) ** (365 / max(n_days, 1)) - 1) * 100

    # Daily returns
    daily_eq = df_e["equity"].resample("D").last().ffill()
    daily_ret = daily_eq.pct_change().dropna()

    sharpe  = (daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() > 0 else 0
    neg_ret = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / neg_ret.std() * np.sqrt(365)) if len(neg_ret) > 1 and neg_ret.std() > 0 else 0

    # Max drawdown
    roll_max = df_e["equity"].cummax()
    drawdown = (df_e["equity"] - roll_max) / roll_max * 100
    max_dd   = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    win_rate  = len(wins) / len(pnl) * 100 if len(pnl) > 0 else 0
    avg_win   = wins.mean() if len(wins) > 0 else 0
    avg_loss  = abs(losses.mean()) if len(losses) > 0 else 0
    rr_ratio  = avg_win / avg_loss if avg_loss > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf
    avg_hold  = df_t["hold_hours"].mean()

    # Monthly returns
    df_t["exit_time"] = pd.to_datetime(df_t["exit_time"])
    df_t["month"] = df_t["exit_time"].dt.to_period("M")
    monthly = df_t.groupby("month")["pnl_net"].sum()
    monthly_pct = (monthly / STARTING_CAPITAL * 100).round(2)
    best_m  = monthly_pct.max()
    worst_m = monthly_pct.min()

    # Exit reason breakdown
    exit_reasons = df_t["exit_reason"].value_counts().to_dict()

    return {
        "period_start":    oos_start,
        "period_end":      oos_end,
        "total_trades":    len(pnl),
        "win_rate_pct":    round(win_rate, 2),
        "total_return_pct": round(total_return, 2),
        "ann_return_pct":  round(ann_return, 2),
        "sharpe_ratio":    round(sharpe, 3),
        "sortino_ratio":   round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio":    round(calmar, 3),
        "avg_win_usd":     round(avg_win, 2),
        "avg_loss_usd":    round(avg_loss, 2),
        "win_loss_ratio":  round(rr_ratio, 3),
        "profit_factor":   round(profit_factor, 3),
        "avg_hold_hours":  round(avg_hold, 2),
        "best_month_pct":  round(best_m, 2),
        "worst_month_pct": round(worst_m, 2),
        "monthly_returns": {str(k): float(v) for k, v in monthly_pct.items()},
        "exit_reasons":    exit_reasons,
        "final_equity":    round(df_e["equity"].iloc[-1], 2),
    }


def run_monte_carlo(trades: list, n_sim: int = MC_SIMULATIONS) -> dict:
    """Shuffle trade order N times, compute percentile stats."""
    if not trades:
        return {}
    pnl = np.array([t["pnl_net"] for t in trades])
    all_returns = []
    all_dd      = []
    rng = np.random.default_rng(42)

    for _ in range(n_sim):
        shuffled = rng.permutation(pnl)
        equity   = STARTING_CAPITAL + np.cumsum(shuffled)
        total_ret = (equity[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        roll_max  = np.maximum.accumulate(np.concatenate([[STARTING_CAPITAL], equity]))
        dd = (equity - roll_max[1:]) / roll_max[1:] * 100
        all_returns.append(total_ret)
        all_dd.append(dd.min())

    return {
        "mc_simulations":       n_sim,
        "mc_p5_max_drawdown":   round(np.percentile(all_dd, 5), 2),
        "mc_p50_total_return":  round(np.percentile(all_returns, 50), 2),
        "mc_p95_total_return":  round(np.percentile(all_returns, 95), 2),
        "mc_p5_total_return":   round(np.percentile(all_returns, 5), 2),
    }


def process_asset(symbol: str) -> dict:
    """Full pipeline for one asset."""
    print(f"\n  Loading {symbol.upper()}...")
    candles = load_candles_1h(symbol)
    funding = load_funding(symbol)

    # Compute signals
    df = compute_signals(candles, funding)
    df = df.dropna(subset=["zscore", "atr"])

    # Walk-forward split
    start = df.index.min()
    oos_start = start + pd.Timedelta(days=TRAIN_DAYS)
    oos_end   = oos_start + pd.Timedelta(days=OOS_DAYS)
    df_window = df[df.index <= oos_end]

    print(f"  Full data: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  OOS:       {oos_start.date()} → {oos_end.date()}")

    # Run backtest
    result = run_backtest(df_window, oos_start, symbol.upper())
    trades     = result["trades"]
    equity_ts  = result["equity_curve"]

    print(f"  Trades: {len(trades)}")

    metrics = compute_metrics(
        trades, equity_ts,
        str(oos_start.date()), str(oos_end.date())
    )
    mc = run_monte_carlo(trades)

    return {
        "asset":    symbol.upper(),
        "metrics":  metrics,
        "monte_carlo": mc,
        "trades":   trades,
        "equity_curve": equity_ts,
    }


def combine_results(all_results: list) -> dict:
    """Merge all-asset trades into combined metrics."""
    all_trades = []
    for r in all_results:
        all_trades.extend(r["trades"])

    # Sort combined equity by time using merged equity curves
    # Rebuild equity from combined trades
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
        "asset": "COMBINED",
        "metrics": metrics,
        "monte_carlo": mc,
        "trades": all_trades,
        "equity_curve": equity_curve,
    }


def verdict(metrics: dict) -> tuple:
    """Simple PASS/FAIL verdict."""
    if "error" in metrics:
        return "FAIL", "No trades generated — insufficient signal occurrences"

    tr    = metrics.get("total_return_pct", 0)
    sr    = metrics.get("sharpe_ratio", 0)
    dd    = metrics.get("max_drawdown_pct", 0)
    wr    = metrics.get("win_rate_pct", 0)
    trades = metrics.get("total_trades", 0)

    if trades < 5:
        return "FAIL", "Too few trades for statistical significance"
    if tr <= 0:
        return "FAIL", f"Negative returns ({tr:.1f}%) — strategy loses money out-of-sample"
    if sr < 0.5:
        return "FAIL", f"Low Sharpe ratio ({sr:.2f}) — insufficient risk-adjusted returns"
    if dd < -30:
        return "WARN", f"Returns positive but drawdown too large ({dd:.1f}%)"
    if tr > 0 and sr >= 0.5:
        return "PASS", f"Positive returns ({tr:.1f}%), Sharpe {sr:.2f}, max DD {dd:.1f}%"
    return "FAIL", "Strategy does not meet minimum performance criteria"


def print_summary(result: dict):
    asset   = result["asset"]
    m       = result.get("metrics", {})
    mc      = result.get("monte_carlo", {})

    if "error" in m:
        print(f"\nASSET: {asset}")
        print(f"  ERROR: {m['error']}")
        return

    verd, reason = verdict(m)

    print(f"""
ASSET: {asset}
  Period:           {m.get('period_start')} → {m.get('period_end')}
  Total Trades:     {m.get('total_trades')}
  Win Rate:         {m.get('win_rate_pct')}%
  ─────────────────────────────────────────
  Total Return:     {m.get('total_return_pct')}%
  Ann. Return:      {m.get('ann_return_pct')}%
  Final Equity:     ${m.get('final_equity'):,.2f}
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

    # Monthly returns table
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


def main():
    print("═" * 51)
    print("  STRATEGY 1 — FUNDING RATE MEAN REVERSION")
    print("  Backtest Results (Out-of-Sample)")
    print("═" * 51)

    assets = ["btc", "eth", "sol"]
    all_results = []
    all_trades_combined  = []
    all_equity_combined  = []

    for symbol in assets:
        try:
            result = process_asset(symbol)
            all_results.append(result)
            all_trades_combined.extend(result["trades"])
        except Exception as e:
            print(f"  ERROR processing {symbol.upper()}: {e}")
            import traceback
            traceback.print_exc()

    # Combined
    if all_results:
        combined = combine_results(all_results)
        all_results.append(combined)

    # Print summaries
    for result in all_results:
        print_summary(result)

    # ─── Save results ─────────────────────────────────────────────────────
    print("\nSaving results...")

    # Full JSON results
    output = {}
    all_trades_flat = []
    all_equity_flat = []

    for r in all_results:
        asset = r["asset"]
        output[asset] = {
            "metrics":     r.get("metrics", {}),
            "monte_carlo": r.get("monte_carlo", {}),
        }
        for t in r.get("trades", []):
            all_trades_flat.append(t)
        for e in r.get("equity_curve", []):
            all_equity_flat.append({"asset": asset, **e})

    with open(RESULTS / "strategy1_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    if all_trades_flat:
        pd.DataFrame(all_trades_flat).to_csv(RESULTS / "strategy1_trades.csv", index=False)

    if all_equity_flat:
        pd.DataFrame(all_equity_flat).to_csv(RESULTS / "strategy1_equity_curve.csv", index=False)

    print(f"\nResults saved to: {RESULTS}")
    print("  strategy1_results.json")
    print("  strategy1_trades.csv")
    print("  strategy1_equity_curve.csv")
    print("\n" + "═" * 51)


if __name__ == "__main__":
    main()
