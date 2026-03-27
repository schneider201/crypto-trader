#!/usr/bin/env python3
"""
Strategy 1 v3 — Funding Rate Mean Reversion (Fixed)

Key fixes over v1:
  1. Trend filter: skip trades when 24h price momentum > threshold
  2. Better TP: 3R price target (not zscore-based) + partial exit at 2R
  3. Funding income: collect funding while holding position
  4. ATR-based stop on 4h candles (more stable than 1h)
  5. Only trade when zscore extreme AND rate sustained for 2+ epochs

Key fix over v2:
  - Uses zscore (not absolute rate) — HL rates are too small for absolute thresholds
  - Reasonable trade frequency restored
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("research/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/historical")

# ── Constants ──────────────────────────────────────────────────────────────
CAPITAL = 10_000.0
RISK_PCT = 0.0075          # 0.75% risk per trade
MAX_LEVERAGE = 4.0
FEE_PER_SIDE = 0.0007      # 0.04% taker + 0.03% slippage
ROUND_TRIP_COST = FEE_PER_SIDE * 2

TRAIN_DAYS = 240
OOS_DAYS = 125

ASSETS = {
    "BTC": ("binance_btcusdt_candles_1m.parquet", "hl_btc_funding_1h.parquet"),
    "ETH": ("binance_ethusdt_candles_1m.parquet", "hl_eth_funding_1h.parquet"),
    "SOL": ("binance_solusdt_candles_1m.parquet", "hl_sol_funding_1h.parquet"),
}

# ── Parameters to grid-search on TRAIN set ─────────────────────────────────
PARAM_GRID = {
    "zscore_threshold": [1.5, 2.0, 2.5],
    "trend_filter_pct": [0.02, 0.04, 0.06],   # 24h price move limit
    "max_hold_hours": [16, 24, 48],
    "atr_multiplier": [1.5, 2.0, 2.5],
    "tp_r_multiple": [2.5, 3.0, 4.0],
}


# ── Data loading ────────────────────────────────────────────────────────────
def load_data(candle_file: str, funding_file: str) -> pd.DataFrame:
    candles = pd.read_parquet(DATA_DIR / candle_file)
    candles.index = pd.to_datetime(candles.index, utc=True)
    candles = candles.sort_index()

    # Resample to 1h
    ohlcv_1h = candles.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    funding = pd.read_parquet(DATA_DIR / funding_file)
    funding.index = pd.to_datetime(funding.index, utc=True)
    funding = funding.sort_index()

    # Merge on hourly index
    df = ohlcv_1h.copy()
    df["funding_rate"] = funding["funding_rate"].reindex(df.index, method="ffill")
    df = df.dropna(subset=["funding_rate"])
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ATR(14) on hourly candles
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Funding zscore (720h = 30 days rolling)
    roll = df["funding_rate"].rolling(720, min_periods=168)
    df["f_mean"] = roll.mean()
    df["f_std"] = roll.std()
    df["f_zscore"] = (df["funding_rate"] - df["f_mean"]) / df["f_std"]

    # Funding momentum: rate consistently above mean for 3+ consecutive hours
    df["f_above_mean"] = df["funding_rate"] > df["f_mean"]
    df["f_sustained_high"] = df["f_above_mean"].rolling(3).sum() == 3    # 3 consecutive high
    df["f_sustained_low"] = (~df["f_above_mean"]).rolling(3).sum() == 3  # 3 consecutive low

    # 24h price momentum
    df["price_mom_24h"] = (df["close"] - df["close"].shift(24)) / df["close"].shift(24)

    # 4h ATR (for more stable stop)
    tr4 = tr.rolling(4).sum()
    df["atr_4h"] = tr4.rolling(14).mean()

    return df.dropna()


# ── Backtester ───────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    z_thr = params["zscore_threshold"]
    trend_lim = params["trend_filter_pct"]
    max_hold = params["max_hold_hours"]
    atr_mult = params["atr_multiplier"]
    tp_r = params["tp_r_multiple"]

    equity = CAPITAL
    trades = []
    equity_curve = []

    in_trade = False
    entry_price = stop_price = tp_price = partial_tp_price = 0.0
    direction = 0
    hours_held = 0
    partial_exited = False
    position_size = 0.0
    entry_funding_rate = 0.0
    entry_idx = 0

    for i, (ts, row) in enumerate(df.iterrows()):
        equity_curve.append({"time": ts, "equity": equity})

        if in_trade:
            hours_held += 1
            price = row["close"]

            # Collect funding income every 8 hours
            if hours_held % 8 == 0:
                funding_income = abs(entry_funding_rate) * position_size * equity * 8 / 100
                equity += funding_income

            # Check exits
            exit_reason = None
            exit_price = price

            if direction == -1:  # SHORT
                if price >= stop_price:
                    exit_reason = "STOP"
                    exit_price = stop_price
                elif not partial_exited and price <= partial_tp_price:
                    # Partial exit at 2R
                    pnl = (entry_price - exit_price) / entry_price * position_size * equity * 0.5
                    pnl -= ROUND_TRIP_COST * equity * 0.5
                    equity += pnl
                    position_size *= 0.5
                    partial_exited = True
                    # Trail stop to breakeven
                    stop_price = entry_price * 1.001
                elif price <= tp_price:
                    exit_reason = "TP"
                elif hours_held >= max_hold:
                    exit_reason = "TIMEOUT"
            else:  # LONG
                if price <= stop_price:
                    exit_reason = "STOP"
                    exit_price = stop_price
                elif not partial_exited and price >= partial_tp_price:
                    pnl = (exit_price - entry_price) / entry_price * position_size * equity * 0.5
                    pnl -= ROUND_TRIP_COST * equity * 0.5
                    equity += pnl
                    position_size *= 0.5
                    partial_exited = True
                    stop_price = entry_price * 0.999
                elif price >= tp_price:
                    exit_reason = "TP"
                elif hours_held >= max_hold:
                    exit_reason = "TIMEOUT"

            if exit_reason:
                if direction == -1:
                    pnl_pct = (entry_price - exit_price) / entry_price
                else:
                    pnl_pct = (exit_price - entry_price) / entry_price

                pnl = pnl_pct * position_size * equity - ROUND_TRIP_COST * equity
                equity += pnl

                trades.append({
                    "entry_time": df.index[entry_idx],
                    "exit_time": ts,
                    "direction": "SHORT" if direction == -1 else "LONG",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_usd": pnl,
                    "hold_hours": hours_held,
                    "exit_reason": exit_reason,
                    "equity_after": equity,
                })
                in_trade = False

        # Entry logic (only when not in trade)
        if not in_trade and i > 720:
            zscore = row["f_zscore"]
            mom = abs(row["price_mom_24h"])
            atr = row["atr_4h"] if row["atr_4h"] > 0 else row["atr"]

            # Trend filter
            if mom > trend_lim:
                continue

            price = row["close"]
            stop_dist = atr * atr_mult

            # Check leverage
            risk_usd = equity * RISK_PCT
            pos_size_usd = risk_usd / (stop_dist / price)
            if pos_size_usd > equity * MAX_LEVERAGE:
                pos_size_usd = equity * MAX_LEVERAGE

            position_size = pos_size_usd / equity

            if zscore > z_thr and row["f_sustained_high"]:
                # SHORT — funding too positive, overcrowded longs
                direction = -1
                entry_price = price
                stop_price = price + stop_dist
                partial_tp_price = price - stop_dist * 2.0
                tp_price = price - stop_dist * tp_r
                in_trade = True
                partial_exited = False
                hours_held = 0
                entry_funding_rate = row["funding_rate"]
                entry_idx = i

            elif zscore < -z_thr and row["f_sustained_low"]:
                # LONG — funding too negative, overcrowded shorts
                direction = 1
                entry_price = price
                stop_price = price - stop_dist
                partial_tp_price = price + stop_dist * 2.0
                tp_price = price + stop_dist * tp_r
                in_trade = True
                partial_exited = False
                hours_held = 0
                entry_funding_rate = row["funding_rate"]
                entry_idx = i

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_equity": equity,
    }


def compute_metrics(trades: list, equity_curve: list, initial: float = CAPITAL) -> dict:
    if len(trades) < 3:
        return {"total_trades": len(trades), "insufficient_data": True}

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity_curve).set_index("time")

    pnls = df_t["pnl_usd"].values
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    equity_vals = df_e["equity"].values
    returns = np.diff(equity_vals) / equity_vals[:-1]

    sharpe = (returns.mean() / returns.std() * np.sqrt(8760)) if returns.std() > 0 else 0
    neg_ret = returns[returns < 0]
    sortino = (returns.mean() / neg_ret.std() * np.sqrt(8760)) if len(neg_ret) > 0 and neg_ret.std() > 0 else 0

    peak = np.maximum.accumulate(equity_vals)
    dd = (equity_vals - peak) / peak
    max_dd = dd.min() * 100

    total_return = (equity_vals[-1] - initial) / initial * 100
    days = (df_e.index[-1] - df_e.index[0]).days or 1
    ann_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    win_rate = len(winners) / len(pnls) * 100
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    hold_times = df_t["hold_hours"].values
    exit_counts = df_t["exit_reason"].value_counts().to_dict()

    # Monthly returns
    df_e["month"] = df_e.index.to_period("M")
    monthly = {}
    for month, grp in df_e.groupby("month"):
        start_eq = grp["equity"].iloc[0]
        end_eq = grp["equity"].iloc[-1]
        monthly[str(month)] = round((end_eq - start_eq) / start_eq * 100, 2)

    return {
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 2),
        "total_return_pct": round(total_return, 2),
        "ann_return_pct": round(ann_return, 2),
        "final_equity": round(equity_vals[-1], 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "win_loss_ratio": round(wl_ratio, 3),
        "profit_factor": round(profit_factor, 3),
        "avg_hold_hours": round(hold_times.mean(), 2),
        "exit_reasons": exit_counts,
        "monthly_returns": monthly,
    }


def monte_carlo(trades: list, n_sims: int = 500) -> dict:
    if len(trades) < 5:
        return {}
    pnls = [t["pnl_usd"] for t in trades]
    results = []
    for _ in range(n_sims):
        shuffled = np.random.choice(pnls, size=len(pnls), replace=False)
        equity = CAPITAL
        peak = CAPITAL
        max_dd = 0.0
        for p in shuffled:
            equity += p
            peak = max(peak, equity)
            dd = (equity - peak) / peak
            max_dd = min(max_dd, dd)
        results.append({"total_return": (equity - CAPITAL) / CAPITAL * 100, "max_dd": max_dd * 100})
    df_mc = pd.DataFrame(results)
    return {
        "p5_max_drawdown": round(df_mc["max_dd"].quantile(0.05), 2),
        "p50_max_drawdown": round(df_mc["max_dd"].quantile(0.5), 2),
        "p50_total_return": round(df_mc["total_return"].quantile(0.5), 2),
        "p95_total_return": round(df_mc["total_return"].quantile(0.95), 2),
    }


# ── Grid search on train set ─────────────────────────────────────────────────
def grid_search(df_train: pd.DataFrame) -> dict:
    best_sharpe = -999
    best_params = None
    best_result = None

    keys = list(PARAM_GRID.keys())
    vals = list(PARAM_GRID.values())

    from itertools import product
    combos = list(product(*vals))

    for combo in combos:
        params = dict(zip(keys, combo))
        result = run_backtest(df_train, params)
        if len(result["trades"]) < 5:
            continue
        m = compute_metrics(result["trades"], result["equity_curve"])
        if m.get("insufficient_data"):
            continue
        sharpe = m.get("sharpe", -999)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            best_result = result

    return best_params or {
        "zscore_threshold": 2.0,
        "trend_filter_pct": 0.04,
        "max_hold_hours": 24,
        "atr_multiplier": 2.0,
        "tp_r_multiple": 3.0,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 55)
    print("  STRATEGY 1 v3 — FUNDING RATE MEAN REVERSION")
    print("  Backtest Results (Out-of-Sample)")
    print("═" * 55)

    all_results = {}
    all_trades = []
    all_equity = []

    for asset, (candle_file, funding_file) in ASSETS.items():
        print(f"\nLoading {asset}...")
        df = load_data(candle_file, funding_file)
        df = compute_features(df)
        print(f"  Data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} hours)")

        # Split train/OOS
        train_end = df.index[0] + pd.Timedelta(days=TRAIN_DAYS)
        df_train = df[df.index < train_end]
        df_oos = df[df.index >= train_end]

        print(f"  Train: {len(df_train)}h | OOS: {len(df_oos)}h")
        print(f"  Grid searching best params on train set...")

        best_params = grid_search(df_train)
        print(f"  Best params: {best_params}")

        print(f"  Running OOS backtest...")
        oos_result = run_backtest(df_oos, best_params)
        trades = oos_result["trades"]
        equity_curve = oos_result["equity_curve"]

        metrics = compute_metrics(trades, equity_curve)
        mc = monte_carlo(trades)

        all_results[asset] = {
            "best_params": best_params,
            "metrics": metrics,
            "monte_carlo": mc,
        }

        # Collect for combined
        for t in trades:
            t["asset"] = asset
            all_trades.append(t)

        # Print results
        print(f"\n  ─── {asset} OOS Results ───")
        if metrics.get("insufficient_data"):
            print(f"  ⚠ Only {metrics['total_trades']} trades — insufficient for analysis")
            verdict = "INSUFFICIENT DATA"
        else:
            print(f"  Period:       {df_oos.index[0].date()} → {df_oos.index[-1].date()}")
            print(f"  Trades:       {metrics['total_trades']}")
            print(f"  Win Rate:     {metrics['win_rate_pct']}%")
            print(f"  Total Return: {metrics['total_return_pct']}%")
            print(f"  Ann. Return:  {metrics['ann_return_pct']}%")
            print(f"  Sharpe:       {metrics['sharpe']}")
            print(f"  Max DD:       {metrics['max_drawdown_pct']}%")
            print(f"  Win/Loss:     {metrics['win_loss_ratio']}")
            print(f"  Profit Factor:{metrics['profit_factor']}")
            print(f"  Exit reasons: {metrics['exit_reasons']}")
            if mc:
                print(f"  MC P5 DD:     {mc.get('p5_max_drawdown')}%")
                print(f"  MC P50 Ret:   {mc.get('p50_total_return')}%")
            print("  Monthly P&L:")
            for month, ret in metrics.get("monthly_returns", {}).items():
                bar = "█" * int(abs(ret))
                sign = "+" if ret > 0 else ""
                print(f"    {month}  {sign}{ret}%  {bar}")

            sharpe = metrics["sharpe"]
            ret = metrics["total_return_pct"]
            dd = metrics["max_drawdown_pct"]
            trades_n = metrics["total_trades"]

            if trades_n < 10:
                verdict = "INSUFFICIENT DATA"
            elif sharpe > 1.0 and ret > 0 and dd > -15:
                verdict = "PASS"
            elif sharpe > 0.5 and ret > 0:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"

        print(f"\n  VERDICT: {verdict}")
        all_results[asset]["verdict"] = verdict

    # Combined backtest
    print(f"\n{'─'*55}")
    print("  COMBINED (all assets, equal weight)")
    print(f"{'─'*55}")

    if len(all_trades) >= 10:
        # Build combined equity curve from trade P&Ls
        all_trades_sorted = sorted(all_trades, key=lambda x: x["entry_time"])
        equity = CAPITAL
        combined_equity = [{"time": all_trades_sorted[0]["entry_time"], "equity": equity}]
        for t in all_trades_sorted:
            equity += t["pnl_usd"]
            combined_equity.append({"time": t["exit_time"], "equity": equity})

        combined_metrics = compute_metrics(all_trades, combined_equity)
        combined_mc = monte_carlo(all_trades)
        all_results["COMBINED"] = {"metrics": combined_metrics, "monte_carlo": combined_mc}

        print(f"  Total Trades:  {combined_metrics['total_trades']}")
        print(f"  Win Rate:      {combined_metrics['win_rate_pct']}%")
        print(f"  Total Return:  {combined_metrics['total_return_pct']}%")
        print(f"  Sharpe:        {combined_metrics['sharpe']}")
        print(f"  Max DD:        {combined_metrics['max_drawdown_pct']}%")
        print(f"  Profit Factor: {combined_metrics['profit_factor']}")
        if combined_mc:
            print(f"  MC P5 DD:      {combined_mc.get('p5_max_drawdown')}%")
        print("  Monthly P&L:")
        for month, ret in combined_metrics.get("monthly_returns", {}).items():
            bar = "█" * int(abs(ret))
            sign = "+" if ret > 0 else ""
            print(f"    {month}  {sign}{ret}%  {bar}")

        sharpe = combined_metrics["sharpe"]
        ret = combined_metrics["total_return_pct"]
        dd = combined_metrics["max_drawdown_pct"]
        if combined_metrics["total_trades"] < 10:
            combined_verdict = "INSUFFICIENT DATA"
        elif sharpe > 1.0 and ret > 0 and dd > -15:
            combined_verdict = "PASS"
        elif sharpe > 0.5 and ret > 0:
            combined_verdict = "MARGINAL"
        else:
            combined_verdict = "FAIL"
        print(f"\n  VERDICT: {combined_verdict}")
        all_results["COMBINED"]["verdict"] = combined_verdict
    else:
        print(f"  Insufficient combined trades ({len(all_trades)})")

    print("\n" + "═" * 55)

    # Save results
    with open(RESULTS_DIR / "strategy1_v3_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if all_trades:
        pd.DataFrame(all_trades).to_csv(RESULTS_DIR / "strategy1_v3_trades.csv", index=False)

    print(f"\n✅ Results saved to research/results/")


if __name__ == "__main__":
    main()
