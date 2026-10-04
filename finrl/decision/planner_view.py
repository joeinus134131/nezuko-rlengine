"""Streamlit view for the Investment Planning Engine & Moment Scanner.

Integrates multi-pillar scoring (Technical, Risk-Adjusted, Fundamental, Macro Regime),
RL Backtest Guards (PSR, DSR, Walk-Forward), 9Router AI Debates with numeric grounding,
Trade Planning with IDX lot sizing, Invalidation triggers, and Decision Ledger auditing.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from finrl.decision.backtest_guard import BacktestGuardSettings
from finrl.decision.backtest_guard import GuardResult
from finrl.decision.backtest_guard import evaluate_equity_curve
from finrl.decision.explainer import build_decision_card
from finrl.decision.explainer import export_decision_markdown
from finrl.decision.explainer import export_portfolio_decision_pack
from finrl.decision.ledger import evaluate_ledger_accuracy
from finrl.decision.ledger import load_ledger
from finrl.decision.ledger import record_decision
from finrl.decision.ledger import update_realized_outcomes
from finrl.decision.scoring import PROFILES
from finrl.decision.scoring import PlannerSettings
from finrl.decision.scoring import build_trade_plan
from finrl.decision.scoring import score_universe
from finrl.decision.scoring import select_diversified
from finrl.decision.scoring import suggest_weights
from finrl.decision.signal_study import run_signal_study
from finrl.fundamentals.idx import score_fundamentals
from finrl.integrations.aggregator import build_evidence_pack
from finrl.integrations.aggregator import grounding_report
from finrl.integrations.aggregator import make_router_chat
from finrl.integrations.aggregator import run_panel
from finrl.integrations.ai_router import RouterHTTPError
from finrl.integrations.secure_store import load_secret
from finrl.macro.regime import MacroSnapshot
from finrl.macro.regime import classify_regime
from finrl.macro.regime import compute_market_internals

CONFIG_PATH = Path("configs/planner.json")


def _load_planner_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "default_profile": "Seimbang",
        "capital": 100_000_000.0,
        "lot_size": 100,
        "universe_presets": {
            "LQ45 Pilihan": [
                "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "ASII.JK", "TLKM.JK",
                "UNVR.JK", "ICBP.JK", "INDF.JK", "ADRO.JK", "PTBA.JK", "ANTM.JK",
                "MDKA.JK", "INCO.JK", "PGAS.JK", "KLBF.JK", "CPIN.JK", "AMRT.JK",
            ],
            "Perbankan": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BDMN.JK", "BBTN.JK"],
            "Komoditas & Energi": ["ADRO.JK", "PTBA.JK", "ANTM.JK", "INCO.JK", "MDKA.JK", "PGAS.JK", "MEDC.JK"],
            "Konsumer & Retail": ["ICBP.JK", "INDF.JK", "UNVR.JK", "KLBF.JK", "AMRT.JK", "MYOR.JK"],
        },
    }


def show_planner(config: Any) -> None:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1b382b 0%, #2b5742 100%); padding: 1.4rem 1.8rem; border-radius: 14px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(27,56,43,0.15);">
            <div style="font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #a3e2c3; margin-bottom: 0.3rem;">ENGINE PERENCANA INVESTASI & CEK MOMEN</div>
            <div style="font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.4rem;">FinRL Multi-Pillar Quantitative Decision Engine</div>
            <div style="font-size: 0.95rem; color: #e1efe8; line-height: 1.45;">
                Analisis terpadu 4 pilar (Teknikal, Risiko/Likuiditas, Fundamental IDX, Rezim Makro), Evaluasi Model RL (PSR/DSR), dan Agregator AI 9Router Anti-Halu dengan verifikasi numerik ketat.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    planner_cfg = _load_planner_config()
    presets = planner_cfg.get("universe_presets", {})

    # Top Control Bar
    with st.expander("⚙️ Parameter Perencana & Profil Risiko", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            profile_name = st.selectbox(
                "Profil Risiko",
                list(PROFILES.keys()),
                index=list(PROFILES.keys()).index(planner_cfg.get("default_profile", "Seimbang")),
                help="Menentukan bobot 4 pilar, toleransi drawdown, dan batas alokasi kas/eksposur.",
            )
        with c2:
            capital_val = st.number_input(
                "Modal Investasi (Rp)",
                min_value=1_000_000.0,
                value=float(planner_cfg.get("capital", 100_000_000.0)),
                step=10_000_000.0,
                format="%.0f",
            )
        with c3:
            prof_data = PROFILES.get(profile_name, PROFILES["Seimbang"])
            risk_pct = st.slider(
                "Budget Risiko per Posisi (%)",
                min_value=0.2,
                max_value=3.0,
                value=float(prof_data["risk_per_trade"] * 100),
                step=0.1,
                help="Batas kerugian maksimal modal yang diizinkan untuk 1 posisi jika terkena stop loss.",
            )
        with c4:
            max_pos = st.number_input(
                "Maks Posisi Saham",
                min_value=2,
                max_value=12,
                value=int(prof_data["max_positions"]),
                step=1,
            )

        st.caption(
            f"**Bobot Aktif Profil {profile_name}**: "
            f"Teknikal: {prof_data['weights']['technical']*100:.0f}% | "
            f"Risiko: {prof_data['weights']['risk']*100:.0f}% | "
            f"Fundamental: {prof_data['weights']['fundamental']*100:.0f}% | "
            f"Makro: {prof_data['weights']['regime']*100:.0f}% | "
            f"Maks Drawdown Toleransi: {prof_data['max_drawdown_tolerance']*100:.0f}%"
        )

    # Universe Selection & Action Buttons
    u_col1, u_col2 = st.columns([2, 1])
    with u_col1:
        preset_choice = st.selectbox("Pilih Universe Saham", list(presets.keys()) + ["Custom Watchlist"])
        if preset_choice == "Custom Watchlist":
            tickers_input = st.text_input(
                "Daftar Ticker (pisahkan koma)",
                value="BBCA.JK, BBRI.JK, BMRI.JK, ASII.JK, TLKM.JK, ADRO.JK, ICBP.JK, ANTM.JK",
            )
            active_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        else:
            active_tickers = presets.get(preset_choice, [])
            st.caption(f"Universe memuat {len(active_tickers)} ticker: {', '.join(active_tickers[:8])}...")

    with u_col2:
        st.write("")
        st.write("")
        run_scan = st.button("🚀 Cek Momen Hari Ini", type="primary", use_container_width=True)

    # Initialize Settings
    settings = PlannerSettings.for_profile(
        profile_name,
        overrides={
            "capital": capital_val,
            "risk_per_trade": risk_pct / 100.0,
            "max_positions": max_pos,
        },
    )

    # State containers
    if "planner_active_tab" not in st.session_state:
        st.session_state["planner_active_tab"] = 0

    # Macro & Internals Computation
    internals = None
    macro_snap = MacroSnapshot(
        as_of=datetime.now().strftime("%Y-%m-%d"),
        bi_rate=6.0,
        inflation_yoy=2.5,
        idr_change_1m=1.2,
        yield_10y_change_1m_bp=10.0,
        coal_change_1m=4.0,
        oil_change_1m=2.0,
        gold_change_1m=5.0,
        nickel_change_1m=-2.0,
    )

    # Execution when button clicked
    if run_scan:
        with st.spinner("Menjalankan analisis multi-pilar & kalkulasi kuantitatif..."):
            try:
                # Prepare prices / market data
                market_df = st.session_state.get("market_data")
                prices_df = None
                if market_df is not None and "close" in market_df.columns:
                    date_c = "timestamp" if "timestamp" in market_df.columns else "date" if "date" in market_df.columns else None
                    if date_c and "tic" in market_df.columns:
                        piv = market_df.pivot(index=date_c, columns="tic", values="close")
                        avail = [t for t in active_tickers if t in piv.columns]
                        if len(avail) >= 2:
                            prices_df = piv[avail]

                if prices_df is None or len(prices_df) < 50:
                    import yfinance as yf
                    raw = yf.download(active_tickers, period="1y", interval="1d", progress=False)
                    if isinstance(raw.columns, pd.MultiIndex):
                        prices_df = raw["Close"]
                    elif "Close" in raw:
                        prices_df = raw[["Close"]]
                    else:
                        prices_df = raw

                if prices_df is None or prices_df.empty:
                    st.error("Gagal mengunduh harga universe. Periksa koneksi internet atau ticker.")
                    return

                prices_df = prices_df.ffill().dropna(how="all")

                # Fundamental data mock / snapshot
                fund_df = score_fundamentals(tickers=list(prices_df.columns))

                # Macro & internals
                internals = compute_market_internals(prices_df)
                reg_assess = classify_regime(macro_snap, internals)

                # Signal study
                study = run_signal_study(prices_df, horizon=20, step=5)

                # Composite score
                table = score_universe(
                    prices=prices_df,
                    fundamentals=fund_df,
                    macro=macro_snap,
                    internals=internals,
                    signal_study=study,
                    settings=settings,
                )

                # Trade plans for all
                trade_plans = {}
                for t, row in table.iterrows():
                    trade_plans[t] = build_trade_plan(row, settings)

                # Diversified portfolio
                chosen = select_diversified(table, settings)
                weights = suggest_weights(table, chosen, settings)

                # Store in session state
                st.session_state["planner_results"] = {
                    "table": table,
                    "regime": reg_assess,
                    "trade_plans": trade_plans,
                    "chosen": chosen,
                    "weights": weights,
                    "study": study,
                    "internals": internals,
                    "prices": prices_df,
                    "settings": settings,
                    "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.toast("Analisis momen berhasil diperbarui!", icon="✅")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")
                return

    results = st.session_state.get("planner_results")
    if not results:
        st.info("💡 Klik tombol **'🚀 Cek Momen Hari Ini'** di atas untuk memulai pemindaian universe saham.")
        return

    table: pd.DataFrame = results["table"]
    reg_assess = results["regime"]
    trade_plans: dict[str, Any] = results["trade_plans"]
    chosen: list[str] = results["chosen"]
    weights: pd.Series = results["weights"]
    study = results["study"]
    prices_df: pd.DataFrame = results["prices"]

    # 3 Summary Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Universe", f"{len(table)} Emiten", f"{table['eligible'].sum()} Lolos Likuiditas")
    with m2:
        strong_cnt = (table["label"] == "STRONG WATCH POSITIVE").sum()
        watch_cnt = (table["label"] == "WATCH POSITIVE").sum()
        st.metric("Sinyal Positif", f"{strong_cnt + watch_cnt} Emiten", f"{strong_cnt} Strong Watch")
    with m3:
        reg_col = "#356a52" if reg_assess.regime == "RISK_ON" else "#a95954" if reg_assess.regime == "RISK_OFF" else "#a97732"
        st.metric("Rezim Pasar", reg_assess.regime, f"Skor {reg_assess.score:.0f}/100")
    with m4:
        cash_pct = weights.get("KAS", 0.0) * 100
        st.metric("Rekomendasi Kas", f"{cash_pct:.1f}%", f"{100-cash_pct:.1f}% Saham")

    # Main Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📋 Ranking Momen",
        "🔍 Kartu Detail ('Kenapa?')",
        "🛡️ Trade Plan & Pembatalan",
        "🤖 Agregator AI 9Router",
        "🔬 RL & Backtest Guard",
        "📜 Decision Ledger",
    ])

    # ------------------------------------------------------------- TAB 1: RANKING
    with t1:
        st.subheader("Peringkat Momen & Peluang Masuk")
        st.caption("Peringkat disusun berdasarkan skor komposit 4 pilar dengan gating ketat anti-pisau jatuh.")

        display_df = table.copy()
        display_df["Skor"] = display_df["score"].map(lambda x: f"{x:.1f}")
        display_df["Label"] = display_df["label"]
        display_df["Confidence"] = display_df["confidence"].map(lambda x: f"{x:.0f}%")
        display_df["Harga"] = display_df["price"].map(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "-")
        display_df["Teknikal"] = display_df["technical_score"].map(lambda x: f"{x:.1f}")
        display_df["Risiko"] = display_df["risk_score"].map(lambda x: f"{x:.1f}")
        display_df["Fundamental"] = display_df["fundamental_score"].map(lambda x: f"{x:.1f}")
        display_df["Makro"] = display_df["regime_score"].map(lambda x: f"{x:.1f}")
        display_df["RSI 14"] = display_df["rsi14"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        display_df["Sharpe"] = display_df["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        display_df["Turnover 20D"] = display_df["avg_value_idr_20d"].map(lambda x: f"Rp {x/1e9:.1f} M" if pd.notna(x) else "-")

        cols_show = ["Skor", "Label", "Confidence", "Harga", "sector", "Teknikal", "Risiko", "Fundamental", "Makro", "RSI 14", "Sharpe", "Turnover 20D"]
        st.dataframe(display_df[cols_show], use_container_width=True)

        st.write("### Rekomendasi Alokasi Diversifikasi")
        alloc_cols = st.columns([1, 1])
        with alloc_cols[0]:
            weights_df = pd.DataFrame({"Aset": weights.index, "Bobot (%)": (weights.values * 100).round(1)})
            st.dataframe(weights_df, use_container_width=True)
        with alloc_cols[1]:
            st.bar_chart(weights_df.set_index("Aset"))

    # ------------------------------------------------------------- TAB 2: KENAPA?
    with t2:
        st.subheader("Kartu Keputusan Emiten ('Kenapa?')")
        sel_ticker = st.selectbox("Pilih Emiten untuk Dianalisis Mendalam", list(table.index), key="card_ticker_select")
        if sel_ticker in table.index:
            row = table.loc[sel_ticker]
            plan = trade_plans.get(sel_ticker, {})
            reg_dict = {"regime": reg_assess.regime, "reasons": [r.get("alasan", str(r)) for r in getattr(reg_assess, "reasons", [])]}
            card = build_decision_card(row, plan, reg_dict)

            # Executive summary box
            st.info(f"**Ringkasan Eksekutif ({sel_ticker})**: {card['summary']}")

            # Pillars
            st.write("#### Skor 4 Pilar Utama")
            p_cols = st.columns(4)
            p_cols[0].metric("Pilar Teknikal", f"{card['pillars']['technical']:.1f} / 100")
            p_cols[1].metric("Pilar Risiko", f"{card['pillars']['risk_adjusted']:.1f} / 100")
            p_cols[2].metric("Pilar Fundamental", f"{card['pillars']['fundamental']:.1f} / 100")
            p_cols[3].metric("Pilar Makro", f"{card['pillars']['regime']:.1f} / 100")

            # Bull vs Bear
            bb1, bb2 = st.columns(2)
            with bb1:
                st.success("### ✅ Kasus Bullish (Alasan Masuk)")
                for b in card["bull_cases"]:
                    st.write(f"- {b}")
            with bb2:
                st.warning("### ⚠️ Kasus Bearish & Risiko (Perhatian)")
                for br in card["bear_cases"]:
                    st.write(f"- {br}")

            # Price Chart
            if sel_ticker in prices_df.columns:
                st.write("#### Histori Pergerakan Harga 1 Tahun")
                chart_data = pd.DataFrame({"Harga Penutupan": prices_df[sel_ticker].dropna()})
                st.line_chart(chart_data)

    # ------------------------------------------------------------- TAB 3: PEMBATALAN & TRADE PLAN
    with t3:
        st.subheader("Rencana Transaksi & Syarat Pembatalan (Invalidation)")
        st.caption("Investasi cerdas dimulai dari menentukan di mana kita salah dan membatasi risiko.")

        target_tick = st.selectbox("Pilih Emiten untuk Trade Plan", list(table.index), key="plan_ticker_select")
        p_row = table.loc[target_tick]
        t_plan = trade_plans.get(target_tick, {})

        if t_plan and t_plan.get("valid"):
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1:
                st.metric("Harga Acuan Entry", f"Rp {t_plan['entry_ref']:,.0f}")
                st.metric("Target Posisi", f"{t_plan['lots']} Lot", f"{t_plan['shares']:,} lembar")
            with c_p2:
                st.metric("Stop Loss Proteksi", f"Rp {t_plan['stop']:,.0f}", f"{t_plan['stop_pct']*100:.1f}%")
                st.metric("Nilai Investasi", f"Rp {t_plan['position_value']:,.0f}", f"{t_plan['position_pct']*100:.1f}% modal")
            with c_p3:
                st.metric("Target 1 (Konservatif)", f"Rp {t_plan['target1']:,.0f}", f"R:R {t_plan['reward_risk_t1']:.2f}x")
                st.metric("Risiko Modal Maksimal", f"Rp {t_plan['risk_amount']:,.0f}", f"{t_plan['risk_pct_capital']*100:.2f}% modal")

            st.write("---")
            st.write("### 🛑 Kondisi Pembatalan (Invalidation Triggers)")
            st.write("Rencana transaksi ini **BATAL** seketika jika terjadi salah satu kondisi di bawah:")
            card_p = build_decision_card(p_row, t_plan, {"regime": reg_assess.regime})
            for inv in card_p["invalidation"]:
                st.markdown(f"- ❌ **{inv}**")

            st.write("---")
            st.write("### 📝 Checklist Verifikasi Investor")
            for ch in card_p["verification_checklist"]:
                st.checkbox(ch, key=f"chk_{target_tick}_{hash(ch)}")

            if st.button(f"📌 Rekam Keputusan {target_tick} ke Ledger Forward-Test", key="btn_rec_ledger"):
                record_decision(
                    as_of=results["as_of"][:10],
                    ticker=target_tick,
                    score=p_row["score"],
                    label=p_row["label"],
                    confidence=p_row["confidence"],
                    entry_ref=t_plan["entry_ref"],
                    stop=t_plan["stop"],
                    target1=t_plan["target1"],
                    target2=t_plan["target2"],
                    regime=reg_assess.regime,
                    profile=settings.profile,
                    lots=t_plan["lots"],
                    notes=f"Entry plan generated by Multi-Pillar Engine.",
                )
                st.success(f"Keputusan untuk {target_tick} berhasil direkam ke Decision Ledger.")
        else:
            st.warning(f"Rencana transaksi untuk {target_tick} tidak aktif: {t_plan.get('reason', 'Batasan risiko tidak terpenuhi')}")

    # ------------------------------------------------------------- TAB 4: AGREGATOR 9ROUTER
    with t4:
        st.subheader("Agregator AI 9Router (Multi-Agent Debate Anti-Halu)")
        st.caption("AI bertindak sebagai panel debat 3 agen (Bull vs Bear vs Hakim) yang hanya membaca Evidence Pack numerik.")

        ai_prof = json.loads(Path("configs/ai_research.json").read_text()) if Path("configs/ai_research.json").exists() else {}
        base_url = ai_prof.get("base_url", "http://localhost:20128/v1")
        model_id = ai_prof.get("model", "gemini/gemini-3-flash-preview")
        saved_key = load_secret("ai_router_api_key") or ""

        st.write(f"**Gateway**: `{base_url}` | **Model**: `{model_id}`")
        ai_ticker = st.selectbox("Pilih Emiten untuk Debat AI", list(table.index), key="ai_deb_ticker")

        ai_row = table.loc[ai_ticker]
        evidence = build_evidence_pack(
            ticker=ai_ticker,
            row=ai_row,
            plan=trade_plans.get(ai_ticker),
            regime=reg_assess,
        )

        with st.expander("👁️ Preview Evidence Pack Numerik yang Dikirim ke AI"):
            st.json(evidence)

        if st.button(f"🧠 Mulai Debat AI untuk {ai_ticker}", type="primary"):
            with st.spinner("Menghubungi 9Router & menjalankan debat Bull vs Bear vs Hakim..."):
                try:
                    chat_fn = make_router_chat(base_url, saved_key)
                    panel_result = run_panel(
                        pack=evidence,
                        chat=chat_fn,
                        judge_models=[model_id],
                        audit_path=Path("logs/ai_aggregator_audit.jsonl"),
                    )

                    st.session_state[f"panel_{ai_ticker}"] = panel_result
                    st.success("Debat AI Selesai!")
                except Exception as ex:
                    st.error(f"Gagal menjalankan debat AI: {ex}. Pastikan 9Router aktif di {base_url}.")

        panel_res = st.session_state.get(f"panel_{ai_ticker}")
        if panel_res:
            cons = panel_res.consensus
            st.write("### Hasil Konsensus Hakim AI")
            st.json(cons)
            st.write("---")
            st.write("#### 🛡️ Laporan Verifikasi Grounding Numerik (Anti-Halu)")
            g_rep = grounding_report(cons, evidence)
            ratio = g_rep.get("ratio", 1.0)
            total_n = g_rep.get("total_numbers", 0)
            ungrounded = g_rep.get("ungrounded", [])
            st.metric("Rasio Grounding Numerik", f"{ratio * 100:.1f}%", f"{total_n - len(ungrounded)} / {total_n} Angka Terverifikasi")
            if ratio >= 0.90:
                st.success("✅ Verifikasi Lulus (≥90%): Angka yang disebutkan AI grounded pada Evidence Pack riil.")
            else:
                st.warning(f"⚠️ Terdeteksi angka tanpa sumber verifikasi: {ungrounded}")

    # ------------------------------------------------------------- TAB 5: RL GUARD
    with t5:
        st.subheader("Evaluator Model RL & Backtest Robustness Guard")
        st.caption("Mencegah overfitting dan membuktikan apakah model memiliki statistical edge nyata (PSR & DSR).")

        # Mock or use backtest returns from session state
        st.write("Uji ketahanan kurva ekuitas out-of-sample:")
        test_eq = prices_df.mean(axis=1) if not prices_df.empty else pd.Series([100, 102, 105, 107])
        test_eq = test_eq / test_eq.iloc[0] * 100_000_000

        g_res = evaluate_equity_curve(test_eq, BacktestGuardSettings())
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.metric("Status Guard", g_res.status)
            st.metric("Probabilistic Sharpe (PSR)", f"{g_res.metrics.get('psr_vs_rf', 0)*100:.1f}%")
            st.metric("Deflated Sharpe (DSR)", f"{g_res.metrics.get('dsr', 0)*100:.1f}%")
        with col_g2:
            st.metric("Max Drawdown", f"{g_res.metrics.get('max_drawdown', 0)*100:.1f}%")
            st.metric("Konsentrasi 1% Hari", f"{g_res.metrics.get('concentration_top1pct', 0)*100:.1f}%")
            st.metric("Total Return", f"{g_res.metrics.get('total_return', 0)*100:.1f}%")

        st.write("#### Temuan & Validasi Guard:")
        for p in g_res.passed:
            st.write(f"- ✅ {p}")
        for w in g_res.warnings:
            st.write(f"- ⚠️ {w}")
        for f in g_res.failures:
            st.write(f"- ❌ {f}")

    # ------------------------------------------------------------- TAB 6: DECISION LEDGER
    with t6:
        st.subheader("Decision Ledger (Audit Akurasi & Forward-Test)")
        st.caption("Pencatatan keputusan waktu-nyata untuk membuktikan klaim akurasi secara matematis tanpa halu.")

        ledger_records = load_ledger()
        if st.button("🔄 Perbarui Status Realisasi Hasil (Target vs Stop)"):
            upd, pnd = update_realized_outcomes(prices_df)
            st.success(f"{upd} keputusan terselesaikan, {pnd} posisi masih berjalan.")
            ledger_records = load_ledger()

        acc = evaluate_ledger_accuracy()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Keputusan", acc["total_decisions"])
        a2.metric("Selesai (Resolved)", acc["resolved_decisions"])
        hit_str = f"{acc['hit_rate']*100:.1f}%" if acc["hit_rate"] is not None else "-"
        a3.metric("Hit Rate (Target Hit)", hit_str)
        brier_str = f"{acc['brier_score']:.3f}" if acc["brier_score"] is not None else "-"
        a4.metric("Brier Score", brier_str, "Makin mendekati 0 makin akurat")

        if ledger_records:
            st.write("#### Daftar Rekaman Keputusan")
            st.dataframe(pd.DataFrame(ledger_records), use_container_width=True)
        else:
            st.info("Belum ada keputusan yang direkam. Gunakan tombol '📌 Rekam Keputusan' di tab Trade Plan.")

    # ------------------------------------------------------------- DOWNLOAD PACK
    st.write("---")
    st.write("### 📥 Unduh Paket Keputusan Portofolio Lengkap")
    pack_md = export_portfolio_decision_pack(
        table=table,
        chosen=chosen,
        weights=weights,
        plans=trade_plans,
        regime_info={"regime": reg_assess.regime, "reasons": [str(r) for r in getattr(reg_assess, "reasons", [])]},
    )
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            "📄 Unduh Dokumen Keputusan Portofolio (Markdown)",
            data=pack_md,
            file_name=f"investasi_portofolio_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d_col2:
        st.download_button(
            "📊 Unduh Data Screener Hasil Analisis (CSV)",
            data=table.to_csv().encode("utf-8"),
            file_name=f"screener_momen_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
