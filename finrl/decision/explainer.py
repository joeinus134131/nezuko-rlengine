"""Decision explainer: structured decision cards, bull/bear cases, and reports.

This module bridges quantitative math and human decision-making for an investor
who understands technology and has taken advanced stock classes.

Every card includes:
1. Executive summary in plain Indonesian.
2. 4-Pillar score breakdown (Technical, Risk-Adjusted, Fundamental, Macro Regime).
3. Evidence-grounded Bull vs Bear arguments.
4. Risk-first Trade Plan (Entry, Stop ATR, Target 1, Target 2, IDX Lot sizing).
5. Invalidation conditions (When does the hypothesis break down?).
6. Actionable checklist for human verification (financial reports, corporate actions, macro dates).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def build_bull_bear_cases(
    row: pd.Series | dict[str, Any],
    ai_consensus: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Synthesize evidence-backed Bull and Bear arguments from data + AI consensus."""
    bull_cases: list[str] = []
    bear_cases: list[str] = []

    # If AI consensus provided, prioritize its grounded cases
    if ai_consensus:
        bull_from_ai = ai_consensus.get("bullish_case", [])
        bear_from_ai = ai_consensus.get("bearish_case", [])
        risks_from_ai = ai_consensus.get("risiko", [])
        if bull_from_ai:
            bull_cases.extend([str(item) for item in bull_from_ai])
        if bear_from_ai:
            bear_cases.extend([str(item) for item in bear_from_ai])
        if risks_from_ai and not bear_from_ai:
            bear_cases.extend([str(item) for item in risks_from_ai])

    # Supplementary or fallback quantitative rules
    price = float(row.get("price", np.nan))
    sma20 = float(row.get("sma20", np.nan))
    sma50 = float(row.get("sma50", np.nan))
    sma200 = float(row.get("sma200", np.nan))
    rsi = float(row.get("rsi14", np.nan))
    sharpe = float(row.get("sharpe", np.nan))
    max_dd = float(row.get("max_drawdown", np.nan))
    per = float(row.get("per", np.nan))
    pbv = float(row.get("pbv", np.nan))
    roe = float(row.get("roe", np.nan))
    der = float(row.get("der", np.nan))
    div_yield = float(row.get("dividend_yield", np.nan))
    avg_value = float(row.get("avg_value_idr_20d", np.nan))

    # Bullish quant evidence
    if np.isfinite(price) and np.isfinite(sma50) and price > sma50:
        if np.isfinite(sma200) and price > sma200:
            bull_cases.append(f"Tren kuat di atas SMA50 (Rp {sma50:,.0f}) dan SMA200 (Rp {sma200:,.0f})")
        else:
            bull_cases.append(f"Harga berada di atas SMA50 (Rp {sma50:,.0f})")
    if np.isfinite(rsi) and 45 <= rsi <= 65:
        bull_cases.append(f"RSI 14 di level {rsi:.1f} (momentum stabil, belum jenuh beli/overbought)")
    if np.isfinite(sharpe) and sharpe >= 1.0:
        bull_cases.append(f"Risk-adjusted return historis tinggi (Sharpe ratio {sharpe:.2f})")
    if np.isfinite(roe) and roe >= 15.0:
        bull_cases.append(f"Profitabilitas modal sangat sehat (ROE {roe:.1f}%)")
    if np.isfinite(div_yield) and div_yield >= 4.0:
        bull_cases.append(f"Dividend yield atraktif sebesar {div_yield:.1f}%")
    if np.isfinite(avg_value) and avg_value >= 10e9:
        bull_cases.append(f"Likuiditas sangat tinggi (rata-rata transaksi Rp {avg_value / 1e9:.1f} Miliar/hari)")

    # Bearish / Risk quant evidence
    if np.isfinite(price) and np.isfinite(sma200) and price < sma200:
        bear_cases.append(f"Harga masih di bawah SMA200 (Rp {sma200:,.0f}), waspada tren jangka panjang masih bearish")
    if np.isfinite(rsi) and rsi > 75:
        bear_cases.append(f"RSI 14 mencapai {rsi:.1f}, rawan aksi ambil untung (profit taking) jangka pendek")
    if np.isfinite(max_dd) and max_dd < -0.30:
        bear_cases.append(f"Riwayat koreksi tajam (Historical Max Drawdown {max_dd * 100:.1f}%)")
    if np.isfinite(der) and der > 2.0:
        bear_cases.append(f"Rasio utang terhadap ekuitas relatif tinggi (DER {der:.2f}x)")
    if np.isfinite(pbv) and pbv > 5.0:
        bear_cases.append(f"Valuasi premi tinggi (PBV {pbv:.2f}x), sensitif terhadap penurunan ekspektasi laba")
    if np.isfinite(avg_value) and avg_value < 2e9:
        bear_cases.append(f"Likuiditas moderat/tipis (Rp {avg_value / 1e9:.2f} Miliar/hari), waspada slippage saat keluar")

    if not bull_cases:
        bull_cases.append("Tidak ada konfirmasi momentum teknikal atau valuasi kuat saat ini.")
    if not bear_cases:
        bear_cases.append("Risiko sistemik pasar modal dan perubahan makroekonomi tetap perlu diantisipasi.")

    def _dedup(items: list[str]) -> list[str]:
        seen = set()
        res = []
        for x in items:
            s = x.strip()
            if s and s not in seen:
                seen.add(s)
                res.append(s)
        return res

    return _dedup(bull_cases), _dedup(bear_cases)


def build_invalidation_rules(
    row: pd.Series | dict[str, Any],
    trade_plan: dict[str, Any] | None = None,
    regime_info: dict[str, Any] | None = None,
    ai_consensus: dict[str, Any] | None = None,
) -> list[str]:
    """Define objective, verifiable triggers that invalidate this trading/investment idea."""
    rules: list[str] = []

    # 1. Trade plan stop loss
    if trade_plan and trade_plan.get("stop"):
        stop_price = trade_plan["stop"]
        rules.append(f"Harga penutupan (Daily Close) menembus ke bawah Stop Loss Rp {stop_price:,.0f}.")

    # 2. Moving average breakdown
    sma50 = float(row.get("sma50", np.nan))
    if np.isfinite(sma50):
        rules.append(f"Penutupan 3 hari berturut-turut di bawah SMA50 (Rp {sma50:,.0f}) dengan volume meningkat.")

    # 3. Macro regime invalidation
    if regime_info:
        current_regime = regime_info.get("regime", "NEUTRAL")
        if current_regime != "RISK_OFF":
            rules.append("Kondisi makroekonomi memburuk menjadi 'RISK_OFF' (IHSG breakdown atau tekanan moneter tajam).")

    # 4. AI-suggested invalidations if grounded
    if ai_consensus:
        ai_inval = ai_consensus.get("invalidation", [])
        for item in ai_inval:
            rules.append(str(item))

    # 5. Fundamental invalidation
    rules.append("Laporan keuangan kuartal terbaru menunjukkan penurunan laba bersih > 20% YoY atau rasio utang melonjak drastis.")

    seen = set()
    cleaned = []
    for r in rules:
        r_str = r.strip()
        if r_str and r_str not in seen:
            seen.add(r_str)
            cleaned.append(r_str)
    return cleaned


def build_verification_checklist(row: pd.Series | dict[str, Any]) -> list[str]:
    """Actionable checklist for the human investor before executing orders."""
    ticker = str(row.get("ticker", "Emiten"))
    return [
        f"Verifikasi pengumuman aksi korporasi {ticker} di keterbukaan informasi IDX (RUPS, Dividen, Rights Issue).",
        f"Periksa tanggal rilis laporan keuangan kuartalan terbaru {ticker}.",
        "Cek kalender kebijakan suku bunga (Rapat Dewan Gubernur BI / FOMC) dalam 1-2 minggu ke depan.",
        "Pastikan order buy dipasang dengan limit order pada fraksi harga (tick size) resmi BEI.",
        "Konfirmasi kesesuaian alokasi modal dengan profil risiko dan dana darurat pribadi.",
    ]


def build_decision_card(
    row: pd.Series | dict[str, Any],
    trade_plan: dict[str, Any] | None = None,
    regime_info: dict[str, Any] | None = None,
    ai_consensus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive decision card for a single ticker."""
    data = dict(row)
    ticker = str(data.get("ticker") or getattr(row, "name", "-") or "-")
    score = float(data.get("score", 0.0))
    label = str(data.get("label", "NEUTRAL"))
    confidence = float(data.get("confidence", 50.0))
    timing = str(data.get("timing", "SWING / TREND"))
    sector = str(data.get("sector", "Umum"))
    price = float(data.get("price", np.nan))

    # Pillars
    tech_score = float(data.get("technical_score", 50.0))
    risk_score = float(data.get("risk_score", 50.0))
    fund_score = float(data.get("fundamental_score", 50.0))
    regime_score = float(data.get("regime_score", 50.0))

    bull_cases, bear_cases = build_bull_bear_cases(data, ai_consensus)
    invalidation = build_invalidation_rules(data, trade_plan, regime_info, ai_consensus)
    checklist = build_verification_checklist(data)

    # Executive summary
    if label == "STRONG WATCH POSITIVE":
        summary = (
            f"{ticker} menunjukkan sinyal momentum kuat di sektor {sector} didukung oleh "
            f"skor teknikal {tech_score:.1f} dan ketahanan risiko yang solid ({risk_score:.1f}). "
            f"Peluang entry memiliki reward:risk memadai jika disiplin dengan level proteksi."
        )
    elif label == "WATCH POSITIVE":
        summary = (
            f"{ticker} berada dalam radar positif dengan konfirmasi tren yang konstruktif. "
            f"Fundamental dan rezim pasar cukup mendukung, namun pantau konfirmasi volume transaksi."
        )
    elif label == "NEUTRAL":
        summary = (
            f"{ticker} berada di fase konsolidasi atau transisi. Sebagian indikator belum memberikan konfirmasi "
            f"yang tegas untuk entry saat ini. Disarankan menunggu konfirmasi arah harga."
        )
    elif label == "CAUTION":
        summary = (
            f"{ticker} saat ini menunjukkan sinyal kehati-hatian karena tekanan tren menurun, "
            f"volatilitas tinggi, atau ketidaksesuaian dengan rezim makroekonomi saat ini."
        )
    else:
        summary = f"{ticker} memiliki likuiditas transaksi di bawah batas minimum yang disyaratkan."

    return {
        "ticker": ticker,
        "sector": sector,
        "price": price,
        "score": score,
        "label": label,
        "confidence": confidence,
        "timing": timing,
        "summary": summary,
        "pillars": {
            "technical": tech_score,
            "risk_adjusted": risk_score,
            "fundamental": fund_score,
            "regime": regime_score,
        },
        "metrics": {
            "rsi14": data.get("rsi14"),
            "sma20": data.get("sma20"),
            "sma50": data.get("sma50"),
            "sma200": data.get("sma200"),
            "sharpe": data.get("sharpe"),
            "max_drawdown": data.get("max_drawdown"),
            "avg_value_idr_20d": data.get("avg_value_idr_20d"),
            "per": data.get("per"),
            "pbv": data.get("pbv"),
            "roe": data.get("roe"),
            "der": data.get("der"),
            "dividend_yield": data.get("dividend_yield"),
            "fund_completeness": data.get("fund_completeness"),
        },
        "gates": data.get("gates", []),
        "trade_plan": trade_plan or {},
        "bull_cases": bull_cases,
        "bear_cases": bear_cases,
        "invalidation": invalidation,
        "verification_checklist": checklist,
        "ai_consensus": ai_consensus,
    }


def export_decision_markdown(card: dict[str, Any]) -> str:
    """Format single ticker decision card as clean Markdown."""
    ticker = card["ticker"]
    score = card["score"]
    label = card["label"]
    conf = card["confidence"]
    sector = card["sector"]
    price = card.get("price", 0)
    p = card["pillars"]
    plan = card.get("trade_plan", {})

    lines = [
        f"# Kartu Keputusan Investasi: {ticker}",
        f"**Tanggal Analisis**: {datetime.now().strftime('%d %B %Y %H:%M WIB')}  ",
        f"**Sektor**: {sector} | **Harga Terakhir**: Rp {price:,.0f}  ",
        "",
        "## Ringkasan Eksekutif",
        f"> **Skor Komposit**: **{score:.1f} / 100** | **Label**: `{label}` | **Confidence**: `{conf:.0f}%`  ",
        f"> {card['summary']}",
        "",
        "## 1. Analisis 4 Pilar Utama",
        "| Pilar | Skor (0-100) | Keterangan |",
        "|---|---|---|",
        f"| **Teknikal & Momentum** | {p['technical']:.1f} | Tren vs SMA20/50/200, momentum 12-1, RSI |",
        f"| **Risk-Adjusted & Likuiditas** | {p['risk_adjusted']:.1f} | Sharpe, Max Drawdown, CVaR95, Turnover 20D |",
        f"| **Fundamental & Kualitas** | {p['fundamental']:.1f} | Valuasi (PER/PBV), ROE, DER, Div Yield |",
        f"| **Rezim Makro & Mikro** | {p['regime']:.1f} | Suku Bunga, Inflasi, Kurs IDR, IHSG Breadth |",
        "",
    ]

    # Trade Plan
    if plan and plan.get("valid"):
        lines.extend([
            "## 2. Rencana Transaksi (Risk-First Trade Plan)",
            "Rencana ini dirancang dengan prinsip manajemen risiko terukur (stop loss berbasis ATR):",
            "",
            f"- **Harga Acuan Entry**: Rp {plan.get('entry_ref', price):,.0f}",
            f"- **Stop Loss (Proteksi)**: Rp {plan.get('stop', 0):,.0f} ({plan.get('stop_pct', 0) * 100:.1f}%)",
            f"- **Target 1 (Konservatif)**: Rp {plan.get('target1', 0):,.0f} (Reward/Risk: {plan.get('reward_risk_t1', 0):.2f}x)",
            f"- **Target 2 (Ekspansif)**: Rp {plan.get('target2', 0):,.0f}",
            f"- **Ukuran Posisi**: **{plan.get('lots', 0)} Lot** ({plan.get('shares', 0):,} lembar)",
            f"- **Nilai Posisi Estimasi**: Rp {plan.get('position_value', 0):,.0f} ({plan.get('position_pct', 0) * 100:.1f}% modal)",
            f"- **Risiko Modal**: Rp {plan.get('risk_amount', 0):,.0f} ({plan.get('risk_pct_capital', 0) * 100:.2f}% modal)",
            f"- **Faktor Pembatas**: `{plan.get('limited_by', 'budget risiko')}`",
            "",
        ])
    elif plan and not plan.get("valid"):
        lines.extend([
            "## 2. Rencana Transaksi",
            f"*Rencana tidak aktif*: {plan.get('reason', 'Tidak memenuhi batas risiko')}",
            "",
        ])

    # Bull vs Bear
    lines.extend([
        "## 3. Argumen Pendukung vs Risiko",
        "### Kasus Bullish (Alasan Masuk):",
    ])
    for b in card["bull_cases"]:
        lines.append(f"- ✅ {b}")

    lines.extend(["", "### Kasus Bearish & Risiko (Perhatian):"])
    for br in card["bear_cases"]:
        lines.append(f"- ⚠️ {br}")

    # Invalidation
    lines.extend([
        "",
        "## 4. Kondisi Pembatalan Analisis (Invalidation)",
        "Analisis dan hipotesis ini **batal** jika terjadi salah satu kondisi berikut:",
    ])
    for inv in card["invalidation"]:
        lines.append(f"- 🛑 {inv}")

    # Checklist
    lines.extend([
        "",
        "## 5. Checklist Verifikasi Mandiri",
    ])
    for ch in card["verification_checklist"]:
        lines.append(f"- [ ] {ch}")

    lines.extend([
        "",
        "---",
        "*Catatan: Seluruh analisis ini adalah hasil pemrosesan kuantitatif dan riset bantuan AI, "
        "bukan merupakan ajakan atau rekomendasi resmi transaksi efek. Keputusan investasi "
        "sepenuhnya menjadi tanggung jawab masing-masing investor.*",
    ])
    return "\n".join(lines)


def export_portfolio_decision_pack(
    table: pd.DataFrame,
    chosen: list[str],
    weights: pd.Series,
    plans: dict[str, dict[str, Any]],
    regime_info: dict[str, Any] | None = None,
    ai_consensus_map: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format full portfolio investment package in comprehensive Markdown."""
    now_str = datetime.now().strftime("%d %B %Y %H:%M WIB")
    regime_label = regime_info.get("regime", "NEUTRAL") if regime_info else "NEUTRAL"
    regime_reasons = regime_info.get("reasons", []) if regime_info else []

    sections = [
        "# Paket Perencanaan Investasi & Keputusan Portofolio",
        f"**Waktu Pembuatan**: {now_str}  ",
        f"**Rezim Makroekonomi**: `{regime_label}`  ",
        "",
        "## 1. Analisis Rezim Makroekonomi & Pasar",
        f"Kondisi pasar saat ini diklasifikasikan sebagai **{regime_label}**.",
    ]
    if regime_reasons:
        sections.append("\nFaktor-faktor penentu:")
        for r in regime_reasons:
            sections.append(f"- {r}")

    # Allocation Table
    sections.extend([
        "",
        "## 2. Alokasi Portofolio yang Disarankan",
        "Alokasi diversifikasi berdasarkan bobot inverse-volatility dan batas eksposur rezim:",
        "",
        "| Aset / Ticker | Bobot (%) | Nilai Posisi (Estimasi) | Target Lot | Label Momen |",
        "|---|---|---|---|---|",
    ])
    for ticker, weight in weights.items():
        w_pct = weight * 100
        if ticker == "KAS":
            sections.append(f"| **KAS / DEPOSITO** | **{w_pct:.1f}%** | Siaga cadangan likuiditas | - | Proteksi Rezim |")
        else:
            plan = plans.get(ticker, {})
            val = plan.get("position_value", 0)
            lots = plan.get("lots", 0)
            lbl = table.loc[ticker, "label"] if ticker in table.index else "-"
            sections.append(f"| **{ticker}** | {w_pct:.1f}% | Rp {val:,.0f} | {lots} Lot | `{lbl}` |")

    # Detailed Cards for Chosen
    sections.extend(["", "## 3. Kartu Keputusan Emiten Terpilih", ""])
    for ticker in chosen:
        if ticker in table.index:
            row = table.loc[ticker]
            ai_c = (ai_consensus_map or {}).get(ticker)
            card = build_decision_card(row, plans.get(ticker), regime_info, ai_c)
            sections.append(export_decision_markdown(card))
            sections.append("\n---\n")

    # Full Table Ranking
    sections.extend([
        "## 4. Tabel Ranking Lengkap Universe",
        "| Peringkat | Ticker | Sektor | Skor | Label | Confidence | RSI | Sharpe | Max DD | Likuiditas/hari |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    rank = 1
    for ticker, row in table.iterrows():
        val = row.get("avg_value_idr_20d", 0)
        val_str = f"Rp {val / 1e9:.1f} M" if pd.notna(val) else "-"
        rsi_val = f"{row.get('rsi14', 0):.1f}" if pd.notna(row.get('rsi14')) else "-"
        shp_val = f"{row.get('sharpe', 0):.2f}" if pd.notna(row.get('sharpe')) else "-"
        dd_val = f"{row.get('max_drawdown', 0) * 100:.1f}%" if pd.notna(row.get('max_drawdown')) else "-"
        sections.append(
            f"| {rank} | **{ticker}** | {row.get('sector', '-')} | {row.get('score', 0):.1f} | `{row.get('label', '-')}` | "
            f"{row.get('confidence', 0):.0f}% | {rsi_val} | {shp_val} | {dd_val} | {val_str} |"
        )
        rank += 1

    sections.extend([
        "",
        "## 5. Batasan & Tata Kelola Risiko",
        "1. **Bukan Ramalan**: Skor dan ranking adalah pengukuran kondisi saat ini terhadap histori, bukan jaminan harga di masa depan.",
        "2. **Satuan Lot IDX**: Semua perhitungan menggunakan kelipatan 100 lembar saham sesuai regulasi Bursa Efek Indonesia.",
        "3. **Biaya Transaksi**: Asumsi biaya beli 0.15% dan biaya jual 0.25% (termasuk pajak final PPh penjualan saham 0.1%).",
        "4. **Kedisiplinan Stop Loss**: Jika kondisi pembatalan tercapai, investor wajib mengevaluasi kembali posisi tanpa emosional.",
    ])
    return "\n".join(sections)
