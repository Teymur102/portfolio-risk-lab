"""Reproduce every table and figure in the README.

    python run.py

Writes CSV/JSON to results/ and PNGs to figures/. Uses cached data in data/
(downloads it on first run).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prl import backtest as bt
from prl import data, metrics, plots
from prl import var as vr

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"

WINDOW = 756          # 3-year estimation window (trading days)
COST_BPS = 10.0       # one-way cost per unit traded notional
VAR_WINDOW = 500      # ~2-year look-back for HS / Normal / FHS
LAMBDA = 0.94         # RiskMetrics decay
N_BOOT = 5000
SEED = 42
CRISES = {
    "2008-09 (Lehman to Mar 2009 low)": ("2008-09-01", "2009-03-31"),
    "2020 (COVID crash)": ("2020-02-01", "2020-05-29"),
}


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def write_tables_md(perf, tests, var_df, yr, crisis, yearly):
    """Markdown versions of the headline tables, formatted from the same
    numbers written to the CSVs (the README tables are copied from here)."""
    out = ["# Generated tables (python run.py)\n"]
    p = pd.DataFrame({
        "CAGR": perf["CAGR"].map(pct), "Vol": perf["Volatility"].map(pct),
        "Sharpe": perf["Sharpe"].map(lambda x: f"{x:.2f}"),
        "Sortino": perf["Sortino"].map(lambda x: f"{x:.2f}"),
        "Max DD": perf["Max drawdown"].map(lambda x: pct(x, 1)),
        "Calmar": perf["Calmar"].map(lambda x: f"{x:.2f}"),
        "Turnover/yr": perf["Turnover (one-way, ann.)"].map(lambda x: f"{x * 100:.0f}%"),
        "$1 grows to": perf["Growth of $1"].map(lambda x: f"${x:.2f}"),
    })
    out += ["## Strategy performance, net of costs\n", _md(p), ""]
    t = tests.set_index(tests["strategy"] + " vs " + tests["benchmark"])
    t = pd.DataFrame({
        "Sharpe diff": t["diff"].map(lambda x: f"{x:+.2f}"),
        "95% bootstrap CI": [f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(t["boot_ci_low"], t["boot_ci_high"])],
        "bootstrap p": t["boot_p"].map(lambda x: f"{x:.2f}"),
        "JK-Memmel p": t["jk_memmel_p"].map(lambda x: f"{x:.2f}"),
    })
    out += ["## Sharpe difference tests\n", _md(t), ""]
    for conf in ["99%", "95%"]:
        v = var_df[var_df["confidence"] == conf].set_index("model")
        d = pd.DataFrame({
            "Exceptions": v["exceptions"].astype(int).astype(str) + " / " + v["expected_exceptions"].map(lambda x: f"{x:.0f}"),
            "Rate": v["exception_rate"].map(lambda x: pct(x)),
            "Kupiec p": v["kupiec_p"].map(lambda x: f"{x:.3f}"),
            "Christoffersen ind. p": v["christoffersen_ind_p"].map(lambda x: f"{x:.3f}"),
            "Cond. coverage p": v["cond_coverage_p"].map(lambda x: f"{x:.3f}"),
            "Avg VaR": v["avg_VaR"].map(pct), "Avg ES": v["avg_ES"].map(pct),
            "Loss / ES on exception days": v["loss_to_ES_ratio"].map(lambda x: f"{x:.2f}"),
        })
        if conf == "99%":
            d["Basel: share of 250d windows green/yellow/red"] = [
                f"{g * 100:.0f}% / {y * 100:.0f}% / {r * 100:.0f}%" for g, y, r in
                zip(v["share_windows_green"], v["share_windows_yellow"], v["share_windows_red"])]
            d["Worst 250d count"] = v["max_exceptions_any_250d"].astype(int)
            d["Last 250d"] = v["basel_last250_exceptions"].astype(int).astype(str) + " (" + v["basel_last250_zone"] + ")"
        out += [f"## VaR backtest, {conf}\n", _md(d), ""]
    y = yr[yr["year"].isin([2008, 2020, 2022])]
    y = y.assign(cell=y["exceptions"].astype(str) + " (" + y["zone_if_250d"] + ")").pivot(index="model", columns="year", values="cell")
    out += ["## 99% VaR exceptions in selected calendar years (Basel zone)\n", _md(y), ""]
    c = crisis.assign(cell=crisis["exceptions"].astype(str) + " (exp. " + crisis["expected"].map(lambda x: f"{x:.1f}") + ")")
    c = c.pivot(index="model", columns="window", values="cell")
    out += ["## 99% VaR exceptions in crisis windows\n", _md(c), ""]
    yy = yearly.loc[[2008, 2020, 2022]].T.map(pct)
    out += ["## Calendar-year net returns\n", _md(yy), ""]
    (RES / "tables.md").write_text("\n".join(out))


def rp_check(rets: pd.DataFrame, w: pd.DataFrame) -> float:
    """Largest ratio of max to min risk contribution across all risk-parity rebalances."""
    from prl import optimize as opt
    worst = 1.0
    for d, row in w.iterrows():
        i = rets.index.get_loc(d)
        cov, _ = opt.ledoit_wolf(rets.iloc[i - WINDOW + 1: i + 1].to_numpy())
        rc = opt.risk_contributions(row.to_numpy(), cov)
        worst = max(worst, float(rc.max() / rc.min()))
    return worst


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    summary = {"settings": {
        "estimation_window_days": WINDOW, "cost_bps_one_way": COST_BPS,
        "var_window_days": VAR_WINDOW, "ewma_lambda": LAMBDA,
        "bootstrap_reps": N_BOOT, "bootstrap_block_days": 21, "seed": SEED,
        "universe": data.UNIVERSE, "end_date": data.END_DATE}}

    # ------------------------------------------------------------------ data
    px_all = data.load_prices()
    rf_all = data.load_rf()
    px = px_all[data.UNIVERSE].dropna()
    rets = px.pct_change().dropna()
    rf = data.rf_on(rets.index, rf_all)
    summary["data"] = {
        "first_price_by_ticker": {k: str(v.date()) for k, v in px_all.apply(lambda s: s.first_valid_index()).items()},
        "common_sample_start": str(rets.index[0].date()),
        "common_sample_end": str(rets.index[-1].date()),
        "n_daily_returns": int(len(rets)),
    }

    # ------------------------------------------------------- strategies
    res = bt.run_backtest(rets, rf, WINDOW, COST_BPS)
    net, gross = res.net, res.gross
    rf_oos = rf.reindex(net.index)
    perf = metrics.perf_table(net, rf_oos, res.turnover)
    perf_gross = metrics.perf_table(gross, rf_oos)
    perf.to_csv(RES / "strategy_performance_net.csv", float_format="%.6f")
    perf_gross.to_csv(RES / "strategy_performance_gross.csv", float_format="%.6f")
    net.to_csv(RES / "strategy_daily_net_returns.csv", float_format="%.8f")
    res.turnover.to_csv(RES / "rebalance_turnover.csv", float_format="%.6f")
    for name, w in res.weights.items():
        w.to_csv(RES / f"weights_{name.replace('/', '-').replace(' ', '_').lower()}.csv", float_format="%.6f")
    avg_w = pd.DataFrame({k: v.mean() for k, v in res.weights.items()})
    avg_w.to_csv(RES / "average_weights.csv", float_format="%.4f")
    summary["backtest"] = {
        "oos_start": str(net.index[0].date()), "oos_end": str(net.index[-1].date()),
        "n_oos_days": int(len(net)), "n_rebalances": int(res.turnover.shape[0]),
        "ledoit_wolf_shrinkage_mean": float(res.shrinkage.mean()),
        "ledoit_wolf_shrinkage_min": float(res.shrinkage.min()),
        "ledoit_wolf_shrinkage_max": float(res.shrinkage.max()),
        "max_sharpe_fallbacks_to_min_var": res.max_sharpe_fallbacks,
        "risk_parity_worst_max_over_min_risk_contribution": rp_check(rets, res.weights["Risk parity"]),
    }

    # calendar-year returns (net)
    yearly = (1 + net).groupby(net.index.year).prod() - 1
    yearly.to_csv(RES / "strategy_calendar_year_returns.csv", float_format="%.6f")

    # robustness: split the out-of-sample period into two halves
    half = len(net) // 2
    halves = {}
    for lbl, sl in [("first half", slice(0, half)), ("second half", slice(half, None))]:
        sub = net.iloc[sl]
        p = metrics.perf_table(sub, rf_oos.loc[sub.index])
        key = f"{lbl} {sub.index[0]:%Y-%m}..{sub.index[-1]:%Y-%m}"
        halves[f"Sharpe {key}"] = p["Sharpe"]
        halves[f"CAGR {key}"] = p["CAGR"]
    pd.DataFrame(halves).to_csv(RES / "strategy_subperiods.csv", float_format="%.4f")

    # cost sensitivity
    sens = {}
    for c in [0.0, 10.0, 25.0, 50.0]:
        r_c = bt.run_backtest(rets, rf, WINDOW, c).net
        sens[f"{c:g} bps"] = metrics.perf_table(r_c, rf_oos)["Sharpe"]
    pd.DataFrame(sens).to_csv(RES / "sharpe_cost_sensitivity.csv", float_format="%.4f")

    # Sharpe difference tests
    ex = net.sub(rf_oos, axis=0)
    tests = []
    for bench in ["Equal weight", "60/40"]:
        for s in bt.STRATEGIES:
            if s == bench:
                continue
            z, pjk = metrics.jobson_korkie_memmel(ex[s].to_numpy(), ex[bench].to_numpy())
            bb = metrics.block_bootstrap_sharpe_diff(ex[s].to_numpy(), ex[bench].to_numpy(),
                                                     N_BOOT, 21, SEED)
            tests.append({"strategy": s, "benchmark": bench,
                          "sharpe_strategy": perf.loc[s, "Sharpe"], "sharpe_benchmark": perf.loc[bench, "Sharpe"],
                          "diff": bb["diff"], "boot_ci_low": bb["ci_low"], "boot_ci_high": bb["ci_high"],
                          "boot_p": bb["p_value"], "jk_memmel_z": z, "jk_memmel_p": pjk})
    tests = pd.DataFrame(tests)
    tests.to_csv(RES / "sharpe_difference_tests.csv", index=False, float_format="%.4f")

    plots.equity_and_drawdown(net, FIG / "equity_drawdown.png")

    # ------------------------------------------------------------- VaR / ES
    p6040 = px_all[["SPY", "AGG"]].dropna()
    r6040 = (p6040.pct_change().dropna() * np.array([0.6, 0.4])).sum(axis=1)
    r6040.name = "60/40"
    var_rows, year_rows, crisis_rows, zone_rows = [], [], [], []
    forecasts = {}
    for alpha in [0.01, 0.05]:
        conf = f"{int((1 - alpha) * 100)}%"
        fc = vr.rolling_forecasts(r6040, alpha, VAR_WINDOW, LAMBDA)
        forecasts[conf] = fc
        fc.to_csv(RES / f"var_forecasts_{conf.rstrip('%')}.csv", float_format="%.8f")
        T = len(fc)
        for m in vr.MODELS:
            hits = (fc["return"] < -fc[f"{m} VaR"]).to_numpy()
            x = int(hits.sum())
            lr_pof, p_pof = vr.kupiec_pof(x, T, alpha)
            lr_ind, p_ind = vr.christoffersen_ind(hits)
            lr_cc = lr_pof + lr_ind
            from scipy import stats
            p_cc = float(stats.chi2.sf(lr_cc, 2))
            loss_on_exc = -fc["return"][hits]
            es_on_exc = fc[f"{m} ES"][hits]
            row = {"confidence": conf, "model": m, "days": T, "expected_exceptions": alpha * T,
                   "exceptions": x, "exception_rate": x / T,
                   "kupiec_LR": lr_pof, "kupiec_p": p_pof,
                   "christoffersen_ind_LR": lr_ind, "christoffersen_ind_p": p_ind,
                   "cond_coverage_LR": lr_cc, "cond_coverage_p": p_cc,
                   "avg_VaR": fc[f"{m} VaR"].mean(), "avg_ES": fc[f"{m} ES"].mean(),
                   "avg_loss_on_exception": float(loss_on_exc.mean()),
                   "avg_ES_on_exception": float(es_on_exc.mean()),
                   "loss_to_ES_ratio": float(loss_on_exc.mean() / es_on_exc.mean())}
            if alpha == 0.01:
                h = pd.Series(hits.astype(int), index=fc.index)
                roll = h.rolling(250).sum().dropna()
                zones = roll.map(lambda n: vr.basel_zone(int(n)))
                last = int(h.iloc[-250:].sum())
                row.update({"basel_last250_exceptions": last, "basel_last250_zone": vr.basel_zone(last),
                            "share_windows_green": float((zones == "green").mean()),
                            "share_windows_yellow": float((zones == "yellow").mean()),
                            "share_windows_red": float((zones == "red").mean()),
                            "max_exceptions_any_250d": int(roll.max())})
                for yr, g in h.groupby(h.index.year):
                    year_rows.append({"model": m, "year": yr, "days": len(g), "exceptions": int(g.sum()),
                                      "zone_if_250d": vr.basel_zone(int(g.sum())) if len(g) >= 240 else "partial year"})
                for lbl, (a, b) in CRISES.items():
                    g = h.loc[a:b]
                    crisis_rows.append({"window": lbl, "model": m, "days": len(g), "exceptions": int(g.sum()),
                                        "expected": 0.01 * len(g),
                                        "worst_day_loss": float(-fc["return"].loc[a:b].min()),
                                        "VaR_on_worst_day": float(fc[f"{m} VaR"].loc[fc["return"].loc[a:b].idxmin()])})
            var_rows.append(row)
    pd.DataFrame(var_rows).to_csv(RES / "var_backtest.csv", index=False, float_format="%.6f")
    yr = pd.DataFrame(year_rows)
    yr.to_csv(RES / "var99_exceptions_by_year.csv", index=False)
    pd.DataFrame(crisis_rows).to_csv(RES / "var99_crisis_windows.csv", index=False, float_format="%.6f")
    summary["var"] = {"portfolio": "60% SPY / 40% AGG, rebalanced daily (constant weights)",
                      "returns_start": str(r6040.index[0].date()),
                      "forecast_start": str(forecasts["99%"].index[0].date()),
                      "forecast_end": str(forecasts["99%"].index[-1].date()),
                      "n_forecast_days": int(len(forecasts["99%"]))}

    plots.var_full(forecasts["99%"], ["Historical sim", "Filtered HS"], FIG / "var99_vs_returns.png")
    plots.var_crises(forecasts["99%"], vr.MODELS, CRISES, FIG / "var99_crises.png")

    write_tables_md(perf, tests, pd.DataFrame(var_rows), yr, pd.DataFrame(crisis_rows), yearly)

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------- console
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(summary, indent=2))
    print("\nStrategy performance (net of costs):\n", perf.round(4))
    print("\nSharpe tests:\n", tests.round(4))
    print("\nVaR backtest:\n", pd.DataFrame(var_rows).round(4))
    print(f"\nDone in {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
