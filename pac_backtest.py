# Tutto il TUO codice di prima, esclusa la parte che inizia con
# if __name__ == "__main__": e finisce in fondo.
# Incolla qui ↓↓↓   (Ctrl + V)
# -*- coding: utf-8 -*-
"""
pac_backtest.py
Funzioni per simulare un PAC (Piano di Accumulo Capitale) con:
• download dati via yfinance
• costi di commissione, TER
• calcolo XIRR e metriche
• grafici equity, drawdown, rolling XIRR
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import traceback
import warnings
from scipy.optimize import newton

warnings.filterwarnings("ignore")  # ignora warning non critici

# ============================================================
# 1. DATA ACQUISITION
# ============================================================

def download_pac_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scarica prezzi da yfinance (colonne Open/Close) e restituisce
    un DataFrame pulito, pronto per il back‑test.
    """
    print(f"Download dati per {ticker} da {start_date} a {end_date}…")
    try:
        df_raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        if df_raw is None or df_raw.empty:
            raise ValueError(f"Nessun dato scaricato per {ticker}.")

        # Gestione MultiIndex o colonne singole
        open_series, close_series = None, None
        if isinstance(df_raw.columns, pd.MultiIndex):
            if "Open" in df_raw.columns.get_level_values(0):
                open_data = df_raw["Open"]
                open_series = (
                    open_data.iloc[:, 0].copy()
                    if isinstance(open_data, pd.DataFrame) and not open_data.empty
                    else open_data.copy()
                )
            if "Close" in df_raw.columns.get_level_values(0):
                close_data = df_raw["Close"]
                close_series = (
                    close_data.iloc[:, 0].copy()
                    if isinstance(close_data, pd.DataFrame) and not close_data.empty
                    else close_data.copy()
                )
        else:
            if "Open" in df_raw.columns:
                open_series = df_raw["Open"].copy()
            if "Close" in df_raw.columns:
                close_series = df_raw["Close"].copy()

        if open_series is None or close_series is None:
            raise ValueError(
                f"Impossibile estrarre Open/Close. Colonne trovate: {df_raw.columns}"
            )

        df = pd.DataFrame({"Open": open_series, "Close": close_series})
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        rows_before = len(df)
        df.dropna(inplace=True)
        if rows_before > len(df):
            print(f"Rimosse {rows_before - len(df)} righe con NaN.")

        if df.empty:
            raise ValueError("Nessun dato valido dopo la pulizia.")

        print(f"Dati validi: {len(df)} righe.")
        return df

    except Exception as e:
        print(f"Errore download/elaborazione {ticker}: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================
# 2. BACKTEST ENGINE PAC (Costi + Flussi)
# ============================================================

def run_pac_backtest(
    price_df: pd.DataFrame,
    monthly_investment: float,
    max_total_investment: float,
    commission_per_trade: float = 0.0,
    ter_annual_percent: float = 0.0,
):
    """
    Esegue il back‑test PAC:
    • acquisti mensili finché non si raggiunge max_total_investment
    • commissioni e TER inclusi
    Ritorna:
        portfolio_df, total_invested, total_commissions, total_ter_fees,
        cashflow_dates, cashflow_values
    """
    print("\n--- Back‑test PAC ---")

    if price_df.empty:
        print("Dati di prezzo mancanti.")
        return None, 0, 0, 0, [], []

    total_shares = 0.0
    total_invested = 0.0
    total_commissions = 0.0
    total_ter_fees = 0.0
    equity_curve, dates = [], []
    cashflow_dates, cashflow_values = [], []

    ter_daily_rate = (ter_annual_percent / 100.0) / 252.0 if ter_annual_percent else 0
    investing_phase = True
    last_month = None
    start_year = price_df.index[0].year

    for date, row in price_df.iterrows():
        month, year = date.month, date.year
        open_price, close_price = row["Open"], row["Close"]

        # acquisto mensile
        if investing_phase and month != last_month:
            if total_invested < max_total_investment:
                available = monthly_investment - commission_per_trade
                if available > 0 and pd.notna(open_price) and open_price > 0:
                    shares = available / open_price
                    total_shares += shares
                    total_invested += monthly_investment
                    total_commissions += commission_per_trade
                    cashflow_dates.append(date)
                    cashflow_values.append(-monthly_investment)
                    if total_invested >= max_total_investment:
                        print(
                            f"Target {max_total_investment:,.0f}$ raggiunto a {date:%Y-%m}"
                        )
                        investing_phase = False
                else:
                    print(f"Skip acquisto {date.date()} (Open non valido)")
                last_month = month
            else:
                investing_phase = False

        # valore portafoglio
        gross_value = total_shares * close_price if pd.notna(close_price) else (
            equity_curve[-1] if equity_curve else 0
        )

        # TER (dopo il primo anno)
        daily_fee = (
            gross_value * ter_daily_rate if year > start_year and gross_value > 0 else 0
        )
        total_ter_fees += daily_fee
        portfolio_value = gross_value - daily_fee

        equity_curve.append(portfolio_value)
        dates.append(date)

    if not dates:
        print("Nessuna data valida.")
        return None, total_invested, total_commissions, total_ter_fees, [], []

    cashflow_dates.append(dates[-1])
    cashflow_values.append(equity_curve[-1])

    portfolio_df = pd.DataFrame({"Equity": equity_curve}, index=pd.Index(dates, name="Date"))
    print(
        f"Back‑test concluso. Valore finale: {equity_curve[-1]:,.2f}$ "
        f"(investito {total_invested:,.2f}$)"
    )

    return (
        portfolio_df,
        total_invested,
        total_commissions,
        total_ter_fees,
        cashflow_dates,
        cashflow_values,
    )


# ============================================================
# 3. METRICHE + XIRR
# ============================================================

def xnpv(rate, dates, cashflows):
    """NPV “esatto” con date reali per XIRR."""
    if rate <= -1:
        return np.inf
    dates = pd.to_datetime(dates)
    t0 = dates[0]
    years = (dates - t0).days / 365.0
    return np.sum(cashflows / (1 + rate) ** years)


def calculate_xirr(dates, cashflows, guess=0.1):
    """Calcola XIRR con Newton; restituisce np.nan se non converge."""
    try:
        mask = np.isfinite(cashflows) & pd.notna(dates)
        d, c = np.array(dates)[mask], np.array(cashflows)[mask]
        if len(d) < 2 or not (any(c > 0) and any(c < 0)):
            return np.nan
        return newton(lambda r: xnpv(r, d, c), x0=guess, maxiter=100)
    except Exception:
        try:
            return newton(lambda r: xnpv(r, d, c), x0=0, maxiter=100)
        except Exception:
            return np.nan


def calculate_metrics(
    portfolio_df: pd.DataFrame,
    total_invested: float,
    total_commissions: float,
    total_ter_fees: float,
    cf_dates: list,
    cf_values: list,
) -> dict:
    """Ritorna dizionario con metriche chiave."""
    if portfolio_df is None or portfolio_df.empty:
        return {}

    metrics = {
        "Final Portfolio Value ($)": portfolio_df["Equity"].iloc[-1],
        "Total Invested ($)": total_invested,
        "Total Commissions ($)": total_commissions,
        "Total TER Fees ($)": total_ter_fees,
    }
    metrics["Total Net P/L ($)"] = metrics["Final Portfolio Value ($)"] - total_invested
    metrics["Total Net P/L (%)"] = (
        metrics["Total Net P/L ($)"] / total_invested * 100 if total_invested else 0
    )

    peak = portfolio_df["Equity"].cummax()
    dd_monetary = peak - portfolio_df["Equity"]
    metrics["Max Monetary Drawdown ($)"] = dd_monetary.max()
    dd_pct = (dd_monetary / peak).replace([np.inf, -np.inf], 0).fillna(0)
    metrics["Max Drawdown (%)"] = dd_pct.max() * 100
    metrics["Drawdown_Monetary_Series"] = dd_monetary

    xirr_val = calculate_xirr(cf_dates, cf_values)
    metrics["XIRR (%)"] = xirr_val * 100 if pd.notna(xirr_val) else np.nan

    return metrics


# ============================================================
# 4. PLOT EQUITY & DRAWDOWN
# ============================================================

def plot_pac_results(
    portfolio_df: pd.DataFrame,
    metrics: dict,
    ticker: str,
    start: str,
    end: str,
):
    """Plotta equity e drawdown monetario."""
    if (
        portfolio_df is None
        or portfolio_df.empty
        or not metrics
    ):
        print("Dati insuff. per plot.")
        return

    dd_series = metrics.get("Drawdown_Monetary_Series", pd.Series(dtype=float))
    if dd_series.empty:
        dd_series = pd.Series(0, index=portfolio_df.index)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(f"Back‑test PAC – {ticker} ({start} → {end})", fontsize=16)

    # equity
    axes[0].plot(
        portfolio_df.index,
        portfolio_df["Equity"],
        label="Valore Portafoglio",
        color="blue",
    )
    axes[0].set_ylabel("Valore ($)", color="blue")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # drawdown monetario
    axes[1].fill_between(
        dd_series.index, -dd_series, 0, color="red", alpha=0.4, label="Drawdown"
    )
    axes[1].set_ylabel("Drawdown ($)", color="red")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${-x:,.0f}"))
    axes[1].set_xlabel("Data")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ============================================================
# 5. ROLLING XIRR
# ============================================================

def calculate_and_plot_rolling_xirr(
    portfolio_df: pd.DataFrame,
    cf_dates: list,
    cf_values: list,
    ticker: str,
    freq: str = "QE",  # 'QE' trimestre, 'YE' anno, 'ME' mese
):
    """Calcola e plotta l'XIRR rolling."""
    if (
        portfolio_df is None
        or portfolio_df.empty
        or not cf_dates
        or not cf_values
    ):
        print("Dati insuff. per Rolling XIRR.")
        return

    print(f"Rolling XIRR freq={freq}")

    invest_df = pd.DataFrame({"Value": cf_values[:-1]}, index=pd.to_datetime(cf_dates[:-1]))
    resample_dates = portfolio_df.resample(freq).last().index
    resample_dates = resample_dates[
        (resample_dates >= portfolio_df.index.min())
        & (resample_dates <= portfolio_df.index.max())
    ]
    if portfolio_df.index[-1] not in resample_dates:
        resample_dates = resample_dates.append(pd.Index([portfolio_df.index[-1]]))

    min_flows = 3
    rolling_vals, rolling_dates = [], []

    for end_date in resample_dates:
        current_cf = invest_df[invest_df.index <= end_date]

        try:
            port_val = portfolio_df.loc[end_date, "Equity"]
        except KeyError:
            port_val = portfolio_df.loc[:end_date, "Equity"].iloc[-1]
            end_date = portfolio_df.loc[:end_date].index[-1]

        dates_list = current_cf.index.tolist() + [end_date]
        values_list = current_cf["Value"].tolist() + [port_val]

        if len(dates_list) >= min_flows:
            xirr = calculate_xirr(dates_list, values_list)
            if pd.notna(xirr):
                rolling_vals.append(xirr * 100)
                rolling_dates.append(end_date)

    if not rolling_dates:
        print("Nessun XIRR rolling valido.")
        return

    roll_series = pd.Series(rolling_vals, index=pd.Index(rolling_dates, name="Date"))

    plt.figure(figsize=(12, 6))
    roll_series.plot(marker=".", linestyle="-", label="XIRR Rolling")

    final_xirr = calculate_xirr(cf_dates, cf_values)
    if pd.notna(final_xirr):
        plt.axhline(
            final_xirr * 100,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"XIRR Finale ({final_xirr:.2%})",
        )

    plt.title(f"Rolling XIRR – PAC {ticker}")
    plt.ylabel("XIRR (%)")
    plt.xlabel("Data fine periodo")
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
