import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pac_backtest import (
    download_pac_data, run_pac_backtest,
    calculate_metrics, plot_pac_results,
    calculate_and_plot_rolling_xirr
)

st.set_page_config(page_title="Simulatore PAC", layout="wide")
st.title("📈 Simulatore PAC con costi, TER e XIRR")

with st.sidebar:
    st.header("Impostazioni")
    ticker      = st.text_input("Ticker", "CSPX.L")
    anni_stor   = st.slider("Anni di storico", 3, 30, 10)
    inv_mens    = st.number_input("Invest. mensile ($)", 1000.0, step=100.0)
    max_inv     = st.number_input("Invest. massimo ($)", 50000.0, step=1000.0)
    commissione = st.number_input("Commissione ($)", 2.0, step=0.5)
    ter_annuale = st.number_input("TER (%)", 0.07, step=0.01, format="%.2f")
    data_fine   = st.date_input("Data fine", datetime(2024, 12, 31))
    freq_xirr   = st.selectbox("Freq. rolling XIRR",
                 {"QE":"Trimestrale","YE":"Annuale","ME":"Mensile"})

if st.button("Esegui back‑test"):
    data_fine_str = data_fine.strftime("%Y-%m-%d")
    data_inizio   = (data_fine - pd.DateOffset(years=anni_stor)).strftime("%Y-01-01")
    st.info("Scarico dati…")
    prezzi = download_pac_data(ticker, data_inizio, data_fine_str)
    if prezzi.empty:
        st.error("Download dati fallito."); st.stop()
    prezzi = prezzi[prezzi.index >= (data_fine - pd.DateOffset(years=anni_stor)).strftime("%Y-%m-%d")]
    st.info("Back‑test in corso…")
    port, tot_inv, tot_comm, tot_ter, cf_d, cf_v = run_pac_backtest(
        prezzi, inv_mens, max_inv, commissione, ter_annuale
    )
    if port is None or port.empty:
        st.error("Nessun risultato."); st.stop()
    metrics = calculate_metrics(port, tot_inv, tot_comm, tot_ter, cf_d, cf_v)
    st.subheader("Risultati")
    st.write(pd.DataFrame(metrics, index=["Valore"]).T)
    tab1, tab2 = st.tabs(["Equity/Drawdown","Rolling XIRR"])
    with tab1:
        plot_pac_results(port, metrics, ticker,
                         port.index[0].date(), port.index[-1].date())
        st.pyplot(plt)
    with tab2:
        calculate_and_plot_rolling_xirr(port, cf_d, cf_v, ticker, freq=freq_xirr)
        st.pyplot(plt)
    st.success("✅ Finito!")
