"""Interactive E2E workbench for configuring and evaluating FinRL experiments.

Run with: ``streamlit run finrl/dashboard.py``
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import MISSING
from html import escape
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit executes this file as a script and only adds ``finrl/`` to
# ``sys.path``. Add the repository root so absolute ``finrl.*`` imports work
# regardless of the directory from which Streamlit is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
LOG_DIR = PROJECT_ROOT / "logs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from finrl.config_tickers import LQ45_TICKER
from finrl.config_tickers import SRI_KEHATI_TICKER
from finrl.integrations import auth
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


SUPPORTED_LIBRARIES = ("stable_baselines3", "elegantrl", "rllib")
SUPPORTED_MODELS = {
    "stable_baselines3": ("a2c", "ddpg", "ppo", "sac", "td3"),
    "elegantrl": ("ddpg", "ppo", "sac", "td3"),
    "rllib": ("a2c", "ddpg", "ppo", "td3"),
}
DEFAULT_AGENT_PARAMS = {
    "stable_baselines3": {"learning_rate": 0.0003},
    "elegantrl": {"learning_rate": 0.0001, "eval_times": 32},
    "rllib": {"lr": 0.0001, "train_batch_size": 64, "gamma": 0.99},
}
UNIVERSES = (
    "LQ45 Indonesia",
    "SRI-KEHATI Indonesia",
    "Custom Indonesia",
    "Custom global",
)
NAVIGATION = (
    "Beranda",
    "Data Market",
    "Data Sources & Broker",
    "Analisa IDX",
    "Training",
    "Model & Evaluasi",
    "AI Research Copilot",
    "Monitoring & Notifikasi",
    "Paper Trading",
    "Dokumentasi & Output",
)
APP_NAME = "NEZU"
NAVIGATION_LABELS = {
    "Beranda": "Today",
    "Data Market": "Data & Quality",
    "Data Sources & Broker": "Connections",
    "Analisa IDX": "Stock Intelligence",
    "Training": "Experiment Lab",
    "Model & Evaluasi": "Models & Evaluation",
    "AI Research Copilot": "AI Research",
    "Monitoring & Notifikasi": "Monitor",
    "Paper Trading": "Paper Trading",
    "Dokumentasi & Output": "Documentation",
}
NAVIGATION_ICONS = {
    "Beranda": ":material/home:",
    "Data Market": ":material/database:",
    "Data Sources & Broker": ":material/hub:",
    "Analisa IDX": ":material/query_stats:",
    "Training": ":material/science:",
    "Model & Evaluasi": ":material/model_training:",
    "AI Research Copilot": ":material/auto_awesome:",
    "Monitoring & Notifikasi": ":material/monitoring:",
    "Paper Trading": ":material/contract:",
    "Dokumentasi & Output": ":material/menu_book:",
}
NAVIGATION_GROUPS = (
    ("Intelligence", ("Analisa IDX", "AI Research Copilot")),
    ("Experiments", ("Data Market", "Training", "Model & Evaluasi")),
    ("Operations", ("Monitoring & Notifikasi", "Paper Trading")),
    ("System", ("Data Sources & Broker", "Dokumentasi & Output")),
)
PAGE_META = {
    "Beranda": ("TODAY", "Decision workspace", "Ringkasan pasar, kesiapan eksperimen, dan langkah aman berikutnya."),
    "Data Market": ("EXPERIMENTS / DATA", "Data & quality", "Muat, validasi, dan audit histori pasar sebelum digunakan oleh model."),
    "Data Sources & Broker": ("SYSTEM / CONNECTIONS", "Connections", "Kelola provenance data, broker paper, dan readiness setiap provider."),
    "Analisa IDX": ("STOCKS / MARKET SCANNER", "Stock intelligence", "Tinjau kondisi teknikal, relasi, risiko, dan bukti historis ticker."),
    "Training": ("EXPERIMENTS / TRAINING", "Experiment lab", "Latih agen reinforcement learning dengan konfigurasi yang dapat direproduksi."),
    "Model & Evaluasi": ("EXPERIMENTS / MODELS", "Models & evaluation", "Load artefak model dan ukur performa out-of-sample secara objektif."),
    "AI Research Copilot": ("STOCKS / RESEARCH", "AI research", "Sintesis fundamental, berita, sentimen, dan konteks model sebagai bukti tambahan."),
    "Monitoring & Notifikasi": ("MONITOR", "Monitoring & notifications", "Pantau kondisi harian IDX dan audit pengiriman notifikasi."),
    "Paper Trading": ("CONNECTIONS / PAPER", "Paper trading", "Uji alur eksekusi tanpa modal riil pada instrumen yang didukung broker."),
    "Dokumentasi & Output": ("SYSTEM / DOCUMENTATION", "Documentation", "Panduan operasi, mekanisme engine, katalog output, dan batas penggunaan."),
}


NEZU_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
:root {
  --nezu-canvas: #f5f6f2;
  --nezu-surface: #ffffff;
  --nezu-muted: #eef1ec;
  --nezu-ink: #202923;
  --nezu-ink-muted: #68716b;
  --nezu-border: #dde2dc;
  --nezu-moss: #356a52;
  --nezu-moss-dark: #28523f;
  --nezu-moss-soft: #ddebe3;
  --nezu-amber: #a97732;
  --nezu-amber-soft: #f5e9d7;
  --nezu-clay: #a95954;
  --nezu-info: #566a79;
}
html, body, [class*="css"], .stApp {font-family: "DM Sans", -apple-system, BlinkMacSystemFont, sans-serif;}
.stApp {background: var(--nezu-canvas); color: var(--nezu-ink);}
[data-testid="stHeader"] {background: rgba(245,246,242,.88); backdrop-filter: blur(12px);}
[data-testid="stToolbar"] {right: 1.25rem;}
.block-container {max-width: 1480px; padding: 1rem 2rem 4rem;}
[data-testid="stSidebar"] {
  background: var(--nezu-surface);
  border-right: 1px solid var(--nezu-border);
}
[data-testid="stSidebarContent"] {padding: .7rem .85rem 2rem;}
.nezu-brand {display:flex; align-items:center; gap:.75rem; padding:.55rem .35rem 1.1rem;}
.nezu-mark {
  width:40px; height:40px; border-radius:12px; display:grid; place-items:center;
  color:white; background:var(--nezu-moss); font-weight:700; letter-spacing:-.04em;
  box-shadow:0 8px 20px rgba(53,106,82,.18);
}
.nezu-brand-name {font-size:1.12rem; line-height:1.15; font-weight:700; letter-spacing:.08em; color:var(--nezu-ink);}
.nezu-brand-sub {font-size:.72rem; color:var(--nezu-ink-muted); margin-top:.18rem;}
.nezu-side-label {font-size:.67rem; font-weight:700; letter-spacing:.12em; color:#8b948e; padding:.4rem .55rem .35rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.22rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  min-height:42px; border-radius:10px; padding:.46rem .58rem; transition:all .16s ease; color:var(--nezu-ink);
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label p {color:inherit !important;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:var(--nezu-muted);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:var(--nezu-moss-soft); color:var(--nezu-moss-dark); font-weight:600;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {display:none;}
.st-key-nezu_navigation {margin:.1rem 0 .7rem;}
.st-key-nezu_navigation .stButton {margin-bottom:.2rem;}
.st-key-nezu_navigation .stButton > button {
  width:100%; justify-content:flex-start; padding-left:.75rem; min-height:40px;
  border-color:transparent; background:transparent; color:var(--nezu-ink); font-weight:500;
}
.st-key-nezu_navigation .stButton > button > div,
.st-key-nezu_navigation .stButton > button > div > span {width:100%; justify-content:flex-start;}
.st-key-nezu_navigation .stButton > button:hover {background:var(--nezu-muted); border-color:transparent;}
.st-key-nezu_navigation .stButton > button[kind="primary"] {
  background:var(--nezu-moss-soft); border-color:transparent; color:var(--nezu-moss-dark); font-weight:650;
}
.st-key-nezu_navigation [data-testid="stExpander"] {
  border:0; border-radius:10px; background:transparent; margin:.15rem 0;
}
.st-key-nezu_navigation [data-testid="stExpander"] details > summary {
  min-height:40px; border-radius:10px; padding:.2rem .55rem; font-size:.8rem; font-weight:650;
}
.st-key-nezu_navigation [data-testid="stExpander"] details > summary:hover {background:var(--nezu-muted);}
.st-key-nezu_navigation [data-testid="stExpanderDetails"] {padding:.15rem 0 .25rem .55rem;}
.nezu-side-health {
  border:1px solid var(--nezu-border); border-radius:12px; padding:.75rem .8rem; margin:.8rem .2rem .7rem;
  background:#fbfcfa; color:var(--nezu-ink-muted); font-size:.72rem; line-height:1.7;
}
.nezu-side-health strong {color:var(--nezu-ink); font-weight:600;}
.nezu-dot {display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--nezu-moss); margin-right:.38rem;}
.nezu-dot.neutral {background:var(--nezu-info);}
.nezu-topbar {
  min-height:44px; display:flex; align-items:center; justify-content:space-between; gap:1rem;
  border-bottom:1px solid var(--nezu-border); color:var(--nezu-ink-muted); font-size:.76rem; margin-bottom:1.75rem;
}
.nezu-topbar strong {color:var(--nezu-ink); font-weight:600;}
.nezu-page-head {margin-bottom:1.5rem; max-width:850px;}
.nezu-eyebrow {font-size:.7rem; letter-spacing:.14em; font-weight:700; color:var(--nezu-moss); margin-bottom:.38rem;}
.nezu-page-head h1 {font-size:2rem; line-height:1.2; letter-spacing:-.035em; color:var(--nezu-ink); margin:0 0 .45rem; font-weight:650;}
.nezu-page-head p {font-size:.94rem; line-height:1.6; color:var(--nezu-ink-muted); margin:0;}
.nezu-hero {
  padding:1.35rem 1.45rem; border:1px solid var(--nezu-border); border-radius:16px;
  background:linear-gradient(135deg, #ffffff 0%, #edf5ef 100%); margin-bottom:1rem;
}
.nezu-hero-kicker {font-size:.7rem; letter-spacing:.12em; color:var(--nezu-moss); font-weight:700;}
.nezu-hero h2 {font-size:1.42rem; line-height:1.35; color:var(--nezu-ink); margin:.32rem 0 .4rem;}
.nezu-hero p {color:var(--nezu-ink-muted); margin:0; max-width:760px; line-height:1.55;}
.nezu-status-card {
  min-height:128px; padding:1rem; border:1px solid var(--nezu-border); border-radius:14px;
  background:var(--nezu-surface); box-shadow:0 8px 24px rgba(32,41,35,.035);
}
.nezu-status-label {font-size:.66rem; letter-spacing:.09em; color:var(--nezu-ink-muted); font-weight:700;}
.nezu-status-value {font-size:1.05rem; line-height:1.3; color:var(--nezu-ink); font-weight:650; margin:.58rem 0 .32rem;}
.nezu-status-note {font-size:.75rem; line-height:1.45; color:var(--nezu-ink-muted);}
.nezu-pill {display:inline-flex; align-items:center; gap:.35rem; border-radius:999px; padding:.2rem .52rem; background:var(--nezu-moss-soft); color:var(--nezu-moss-dark); font-size:.68rem; font-weight:600;}
.nezu-workflow {
  display:grid; grid-template-columns:repeat(7, minmax(90px, 1fr)); gap:8px; margin:.5rem 0 1.1rem;
}
.nezu-step {background:var(--nezu-surface); border:1px solid var(--nezu-border); border-radius:12px; padding:.8rem; min-height:92px;}
.nezu-step b {display:block; width:24px; height:24px; border-radius:8px; background:var(--nezu-moss-soft); color:var(--nezu-moss-dark); text-align:center; line-height:24px; font-size:.72rem; margin-bottom:.55rem;}
.nezu-step span {font-size:.75rem; line-height:1.35; font-weight:600; color:var(--nezu-ink);}
h1, h2, h3 {color:var(--nezu-ink); letter-spacing:-.02em;}
h3 {font-size:1.08rem !important; margin-top:1.6rem !important;}
[data-testid="stMetric"] {
  background:var(--nezu-surface); padding:.92rem 1rem; border:1px solid var(--nezu-border);
  border-radius:14px; box-shadow:0 8px 24px rgba(32,41,35,.03);
}
[data-testid="stMetricLabel"] {color:var(--nezu-ink-muted);}
[data-testid="stMetricValue"] {color:var(--nezu-ink); font-variant-numeric:tabular-nums;}
.stButton > button, .stDownloadButton > button {
  border-radius:10px; min-height:40px; border-color:var(--nezu-border); font-weight:600;
  box-shadow:none; transition:all .16s ease; background:var(--nezu-surface); color:var(--nezu-ink);
}
.stButton > button:hover, .stDownloadButton > button:hover {border-color:var(--nezu-moss); color:var(--nezu-moss-dark);}
.stButton > button[kind="primary"] {background:var(--nezu-moss); border-color:var(--nezu-moss); color:white;}
.stButton > button[kind="primary"]:hover {background:var(--nezu-moss-dark); color:white;}
[data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  border-color:var(--nezu-border) !important; border-radius:10px !important; background:var(--nezu-surface) !important;
}
[data-baseweb="tab-list"] {gap:.35rem; border-bottom:1px solid var(--nezu-border);}
[data-baseweb="tab"] {border-radius:9px 9px 0 0; color:var(--nezu-ink-muted);}
[aria-selected="true"][data-baseweb="tab"] {color:var(--nezu-moss-dark); font-weight:600; background:var(--nezu-moss-soft);}
[data-testid="stExpander"] {background:var(--nezu-surface); border:1px solid var(--nezu-border); border-radius:12px; overflow:hidden;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {color:var(--nezu-ink) !important;}
[data-testid="stDataFrame"] {border:1px solid var(--nezu-border); border-radius:12px; overflow:hidden; background:var(--nezu-surface);}
[data-testid="stAlert"] {border-radius:12px; border-width:1px;}
hr {border-color:var(--nezu-border) !important;}
code {color:var(--nezu-moss-dark); background:var(--nezu-muted); border-radius:5px; padding:.08rem .28rem;}
@media (max-width: 1100px) {
  .nezu-workflow {grid-template-columns:repeat(4, 1fr);}
  .block-container {padding-left:1.1rem; padding-right:1.1rem;}
}
@media (max-width: 700px) {
  .nezu-topbar {align-items:flex-start; flex-direction:column; padding-bottom:.7rem;}
  .nezu-workflow {grid-template-columns:repeat(2, 1fr);}
  .nezu-page-head h1 {font-size:1.65rem;}
}
</style>
"""


@dataclass
class ExperimentConfig:
    tickers: list[str]
    data_source: str
    interval: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    indicators: list[str]
    use_vix: bool
    drl_lib: str
    model_name: str
    model_path: str
    timesteps: int
    agent_params: dict[str, Any]
    universe: str
    risk_free_rate: float
    buy_cost_pct: float = 0.0016
    sell_cost_pct: float = 0.0035
    lot_size: int = 100
    stop_loss_pct: float = 0.0


def _go_to(page: str) -> None:
    st.session_state["navigation"] = page


def _render_sidebar_navigation() -> str:
    """Render a compact hierarchy while preserving the existing page routing."""
    page = st.session_state.get("navigation", "Beranda")
    if page not in NAVIGATION:
        page = "Beranda"
        st.session_state["navigation"] = page

    st.button(
        NAVIGATION_LABELS["Beranda"],
        key="nav_beranda",
        icon=NAVIGATION_ICONS["Beranda"],
        type="primary" if page == "Beranda" else "secondary",
        use_container_width=True,
        on_click=_go_to,
        args=("Beranda",),
    )
    for group, pages in NAVIGATION_GROUPS:
        with st.expander(group, expanded=page in pages):
            for target in pages:
                st.button(
                    NAVIGATION_LABELS[target],
                    key=f"nav_{target.lower().replace(' ', '_').replace('&', 'and')}",
                    icon=NAVIGATION_ICONS[target],
                    type="primary" if page == target else "secondary",
                    use_container_width=True,
                    on_click=_go_to,
                    args=(target,),
                )
    return page


def _render_page_header(page: str) -> None:
    eyebrow, title, description = PAGE_META[page]
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    session = "Pra-pasar"
    if now.weekday() < 5:
        minute = now.hour * 60 + now.minute
        if 9 * 60 <= minute < 12 * 60:
            session = "Sesi 1 IDX"
        elif 13 * 60 + 30 <= minute < 16 * 60:
            session = "Sesi 2 IDX"
        elif minute >= 16 * 60:
            session = "Pasar tutup"
    st.markdown(
        f"""
        <div class="nezu-topbar">
          <div><span class="nezu-dot"></span><strong>{escape(session)}</strong> · Research workspace</div>
          <div>Waktu Jakarta · {now.strftime('%d %b %Y, %H:%M WIB')}</div>
        </div>
        <div class="nezu-page-head">
          <div class="nezu-eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notify_success(message: str, banner: bool = True, title: str | None = None) -> None:
    toast_msg = f"{title}: {message}" if title else message
    toast_popup = toast_msg[:117] + "..." if len(toast_msg) > 120 else toast_msg
    try:
        st.toast(toast_popup, icon="✅")
    except Exception:
        pass
    if banner:
        st.success(f"**{title}**: {message}" if title else message)


def notify_warning(message: str, banner: bool = True, title: str | None = None) -> None:
    toast_msg = f"{title}: {message}" if title else message
    toast_popup = toast_msg[:117] + "..." if len(toast_msg) > 120 else toast_msg
    try:
        st.toast(toast_popup, icon="⚠️")
    except Exception:
        pass
    if banner:
        st.warning(f"**{title}**: {message}" if title else message)


def notify_info(message: str, banner: bool = True, title: str | None = None) -> None:
    toast_msg = f"{title}: {message}" if title else message
    toast_popup = toast_msg[:117] + "..." if len(toast_msg) > 120 else toast_msg
    try:
        st.toast(toast_popup, icon="ℹ️")
    except Exception:
        pass
    if banner:
        st.info(f"**{title}**: {message}" if title else message)


def notify_error(
    message: str | Exception,
    banner: bool = True,
    title: str | None = None,
    details: str | None = None,
) -> None:
    if isinstance(message, Exception):
        raw_text = str(message)
        if not details:
            import traceback

            tb = traceback.format_exc()
            if tb and tb.strip() != "NoneType: None":
                details = tb
            else:
                details = f"{type(message).__name__}: {raw_text}"
    else:
        raw_text = str(message)

    if "Connection refused" in raw_text or "Errno 61" in raw_text or "Errno 111" in raw_text:
        summary = "Router tidak dapat dihubungi (Connection refused). Pastikan service router sedang aktif dan Base URL benar."
    elif "401" in raw_text or "Unauthorized" in raw_text:
        summary = "Akses ditolak (401 Unauthorized). Periksa API key atau kredensial Anda."
    elif "403" in raw_text or "Forbidden" in raw_text:
        summary = "Akses dilarang (403 Forbidden). Izin model atau kuota habis."
    elif "404" in raw_text or "Not Found" in raw_text:
        summary = "Endpoint tidak ditemukan (404 Not Found). Periksa kembali Base URL."
    elif "timed out" in raw_text.lower() or "timeout" in raw_text.lower():
        summary = "Koneksi time out. Server tujuan merespons terlalu lambat."
    elif "nodename nor servname provided" in raw_text or "getaddrinfo failed" in raw_text:
        summary = "Gagal menyelesaikan domain/host (DNS Error). Periksa koneksi internet dan Base URL."
    else:
        summary = raw_text

    toast_msg = f"{title}: {summary}" if title else summary
    toast_popup = toast_msg[:117] + "..." if len(toast_msg) > 120 else toast_msg
    try:
        st.toast(toast_popup, icon="🚨")
    except Exception:
        pass

    if banner:
        banner_msg = f"**{title}**: {summary}" if title else summary
        st.error(banner_msg)
        technical_detail = details or (raw_text if raw_text != summary or len(raw_text) > 80 else None)
        if technical_detail:
            try:
                with st.expander("Detail teknis"):
                    st.code(technical_detail, language="text")
            except Exception:
                safe_detail = escape(technical_detail)
                st.markdown(
                    f"<details style='margin-top:8px;'><summary style='cursor:pointer;font-weight:600;font-size:0.85rem;color:var(--nezu-ink-muted);'>Detail teknis</summary><pre style='white-space:pre-wrap;font-size:0.78rem;background:var(--nezu-muted);padding:8px;border-radius:6px;margin-top:4px;'>{safe_detail}</pre></details>",
                    unsafe_allow_html=True,
                )


def _status_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="nezu-status-card">'
        f'<div class="nezu-status-label">{escape(label.upper())}</div>'
        f'<div class="nezu-status-value">{escape(value)}</div>'
        f'<div class="nezu-status-note">{escape(note)}</div>'
        '</div>'
    )


def _parse_json(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} harus berupa JSON object yang valid: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} harus berupa JSON object.")
    return parsed


def _tickers(value: str) -> list[str]:
    tickers = [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("Masukkan minimal satu ticker.")
    return tickers


def _data_kwargs(source: str, api_key: str = "", api_secret: str = "", api_url: str = "") -> dict[str, str]:
    if source != "alpaca":
        return {}
    if not all((api_key, api_secret, api_url)):
        raise ValueError("Alpaca membutuhkan API key, API secret, dan base URL.")
    return {"API_KEY": api_key, "API_SECRET": api_secret, "API_BASE_URL": api_url}


def _read_json_config(name: str) -> dict[str, Any] | None:
    try:
        from finrl.db.bridge import read_app_config

        value = read_app_config(name, CONFIG_DIR)
        if value is not None:
            return value
    except Exception:
        pass
    path = CONFIG_DIR / name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_config(name: str, payload: dict[str, Any]) -> Path:
    try:
        from finrl.db.bridge import write_app_config

        result = write_app_config(name, payload, CONFIG_DIR)
        # write_app_config returns display string when DB active; resolve real path.
        if "PostgreSQL" not in result:
            return Path(result)
        return CONFIG_DIR / name
    except Exception:
        pass
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _apply_config_payload(payload: dict[str, Any]) -> None:
    """Apply a downloaded experiment config before widgets are instantiated."""
    payload = dict(payload)
    for field_name, field in ExperimentConfig.__dataclass_fields__.items():
        if field_name not in payload and field.default is not MISSING:
            payload[field_name] = field.default

    required = {
        field_name for field_name, field in ExperimentConfig.__dataclass_fields__.items()
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Field konfigurasi belum lengkap: {', '.join(sorted(missing))}")

    universe = str(payload["universe"])
    library = str(payload["drl_lib"])
    model = str(payload["model_name"])
    if universe not in UNIVERSES:
        raise ValueError(f"Universe tidak didukung: {universe}")
    if library not in SUPPORTED_LIBRARIES or model not in SUPPORTED_MODELS[library]:
        raise ValueError(f"Kombinasi library/model tidak didukung: {library}/{model}")

    st.session_state["cfg_universe"] = universe
    st.session_state[f"tickers_{universe}"] = ", ".join(payload["tickers"])
    st.session_state["cfg_source"] = str(payload["data_source"])
    st.session_state["cfg_interval"] = str(payload["interval"])
    for name in ("train_start", "train_end", "test_start", "test_end"):
        st.session_state[f"cfg_{name}"] = pd.Timestamp(payload[name]).date()
    st.session_state["cfg_indicators"] = list(payload["indicators"])
    st.session_state[f"use_vix_{universe}"] = bool(payload["use_vix"])
    st.session_state["cfg_risk_free"] = float(payload["risk_free_rate"]) * 100
    st.session_state["cfg_library"] = library
    st.session_state[f"model_{library}"] = model
    st.session_state[f"path_{library}_{model}"] = str(payload["model_path"])
    st.session_state["cfg_timesteps"] = int(payload["timesteps"])
    st.session_state[f"params_{library}"] = json.dumps(
        payload["agent_params"], indent=2, ensure_ascii=False
    )
    st.session_state["cfg_buy_cost"] = float(payload.get("buy_cost_pct", 0.0016)) * 100
    st.session_state["cfg_sell_cost"] = float(payload.get("sell_cost_pct", 0.0035)) * 100
    st.session_state["cfg_lot_size"] = int(payload.get("lot_size", 100))
    st.session_state["cfg_stop_loss"] = float(payload.get("stop_loss_pct", 0.0)) * 100
    st.session_state["loaded_config_name"] = payload.get("model_path", "konfigurasi")


@st.cache_data(show_spinner=False)
def load_market_data(
    tickers: tuple[str, ...], source: str, start: str, end: str, interval: str,
    indicators: tuple[str, ...], use_vix: bool, source_kwargs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Download and transform data once per configuration."""
    processor = DataProcessor(source, **dict(source_kwargs))
    data = processor.download_data(list(tickers), start, end, interval)
    data = processor.clean_data(data)
    data = processor.add_technical_indicator(data, list(indicators))
    if use_vix:
        data = processor.add_vix(data)
    return data


@st.cache_data(ttl=900, show_spinner=False)
def load_company_research_context(ticker: str) -> dict[str, Any]:
    """Load a compact, source-attributed company snapshot from Yahoo Finance."""
    import yfinance as yf

    company = yf.Ticker(ticker)
    info = company.info or {}
    fundamental_fields = (
        "shortName", "sector", "industry", "currency", "currentPrice",
        "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook",
        "dividendYield", "beta", "revenueGrowth", "earningsGrowth", "profitMargins",
        "returnOnEquity", "debtToEquity", "targetMeanPrice",
    )
    fundamentals = {
        field: info[field]
        for field in fundamental_fields
        if field in info and info[field] is not None
    }

    try:
        raw_news = company.news or []
    except Exception:
        raw_news = []
    news = []
    for item in raw_news[:10]:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        canonical = content.get("canonicalUrl", {}) if isinstance(content, dict) else {}
        provider = content.get("provider", {}) if isinstance(content, dict) else {}
        title = content.get("title") or item.get("title")
        url = canonical.get("url") or item.get("link")
        if not title:
            continue
        news.append(
            {
                "title": title,
                "summary": content.get("summary") or item.get("summary"),
                "publisher": provider.get("displayName") or item.get("publisher"),
                "published": content.get("pubDate") or item.get("providerPublishTime"),
                "url": url,
            }
        )
    return {
        "ticker": ticker,
        "retrieved_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
        "source": "Yahoo Finance via yfinance",
        "fundamentals": fundamentals,
        "news": news,
    }


def build_config() -> ExperimentConfig:
    with st.sidebar.expander("Konfigurasi eksperimen", expanded=False):
        st.caption("Data, agent, periode, dan artefak model aktif.")
        uploaded_config = st.file_uploader(
            "Load konfigurasi / manifest",
            type=("json",),
            help="Gunakan file konfigurasi JSON atau sidecar .config.json dari model.",
        )
        if st.button(
            "Terapkan konfigurasi",
            disabled=uploaded_config is None,
            use_container_width=True,
        ):
            try:
                payload = json.loads(uploaded_config.getvalue().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Isi file harus berupa JSON object.")
                _apply_config_payload(payload)
                st.rerun()
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                notify_error(error, title="Gagal Memuat Konfigurasi")
        if st.session_state.get("loaded_config_name"):
            st.success(f"Config aktif: {st.session_state['loaded_config_name']}")

        universe = st.selectbox(
            "Universe saham",
            UNIVERSES,
            key="cfg_universe",
        )
        preset_tickers = {
            "LQ45 Indonesia": LQ45_TICKER,
            "SRI-KEHATI Indonesia": SRI_KEHATI_TICKER,
            "Custom Indonesia": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"],
            "Custom global": DOW_30_TICKER[:5],
        }[universe]
        ticker_key = f"tickers_{universe}"
        if ticker_key not in st.session_state:
            st.session_state[ticker_key] = ", ".join(preset_tickers)
        ticker_text = st.text_area("Ticker (pisahkan koma)", key=ticker_key)
        source = st.selectbox(
            "Sumber data",
            ("yahoofinance", "licensed_provider", "alpaca", "wrds"),
            key="cfg_source",
            help="licensed_provider menggunakan profil aktif dari menu Data Sources & Broker.",
        )
        if source == "alpaca":
            st.caption("Kredensial hanya disimpan di memori sesi browser.")
            source_api_key = st.text_input("Alpaca data API key", type="password")
            source_api_secret = st.text_input("Alpaca data API secret", type="password")
            source_api_url = st.text_input(
                "Alpaca data base URL", value="https://paper-api.alpaca.markets"
            )
            st.session_state["source_kwargs"] = {
                "API_KEY": source_api_key,
                "API_SECRET": source_api_secret,
                "API_BASE_URL": source_api_url,
            }
        elif source == "licensed_provider":
            try:
                from finrl.db.bridge import load_provider_store as load_provider_bridge

                provider_store = load_provider_bridge(CONFIG_DIR)
            except Exception:
                from finrl.integrations.provider_adapter import load_provider_store

                provider_path = CONFIG_DIR / "providers.json"
                try:
                    provider_store = load_provider_store(provider_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    provider_store = {"active_market_data": ""}
            provider_path = CONFIG_DIR / "providers.json"
            try:
                active_provider = provider_store.get("active_market_data", "")
            except (OSError, ValueError, json.JSONDecodeError):
                active_provider = ""
            st.caption(
                f"Provider aktif: {active_provider or 'belum dipilih'}. "
                "Atur endpoint dan mapping di menu Data Sources & Broker."
            )
            st.session_state["source_kwargs"] = {
                "PROFILE_ID": active_provider,
                "CONFIG_PATH": str(provider_path),
            }
        else:
            st.session_state["source_kwargs"] = {}
        interval = st.selectbox(
            "Interval", ("1D", "1Min", "5Min", "15Min", "1H"), key="cfg_interval"
        )
        date_defaults = {
            "cfg_train_start": pd.Timestamp("2014-01-01").date(),
            "cfg_train_end": pd.Timestamp("2020-12-31").date(),
            "cfg_test_start": pd.Timestamp("2021-01-01").date(),
            "cfg_test_end": pd.Timestamp("2021-12-31").date(),
        }
        for key, default in date_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
        col1, col2 = st.columns(2)
        train_start = col1.date_input(
            "Train mulai", key="cfg_train_start"
        ).isoformat()
        train_end = col2.date_input(
            "Train selesai", key="cfg_train_end"
        ).isoformat()
        col1, col2 = st.columns(2)
        test_start = col1.date_input(
            "Test mulai", key="cfg_test_start"
        ).isoformat()
        test_end = col2.date_input(
            "Test selesai", key="cfg_test_end"
        ).isoformat()
        if "cfg_indicators" not in st.session_state:
            st.session_state["cfg_indicators"] = list(INDICATORS)
        indicators = st.multiselect(
            "Indikator teknikal", INDICATORS, key="cfg_indicators"
        )
        vix_key = f"use_vix_{universe}"
        if vix_key not in st.session_state:
            st.session_state[vix_key] = "Indonesia" not in universe
        use_vix = st.toggle(
            "Gunakan VIX sebagai sinyal risiko",
            key=vix_key,
            help="Untuk universe IDX, default dimatikan karena VIX merepresentasikan pasar AS.",
        )
        if "cfg_risk_free" not in st.session_state:
            st.session_state["cfg_risk_free"] = 6.0
        risk_free_rate = st.number_input(
            "Risk-free tahunan (%)", min_value=0.0, max_value=30.0,
            step=0.25, key="cfg_risk_free",
        )
        library = st.selectbox("Library DRL", SUPPORTED_LIBRARIES, key="cfg_library")
        model = st.selectbox(
            "Model", SUPPORTED_MODELS[library], key=f"model_{library}"
        )
        path_key = f"path_{library}_{model}"
        if path_key not in st.session_state:
            st.session_state[path_key] = f"trained_models/{library}_{model}"
        model_path = st.text_input("Path model", key=path_key)
        if "cfg_timesteps" not in st.session_state:
            st.session_state["cfg_timesteps"] = 50_000
        timesteps = st.number_input(
            "Training timesteps", min_value=1_000, step=1_000,
            key="cfg_timesteps",
        )
        default_params = json.dumps(DEFAULT_AGENT_PARAMS[library], indent=2)
        params_key = f"params_{library}"
        if params_key not in st.session_state:
            st.session_state[params_key] = default_params
        params = st.text_area("Parameter agent (JSON)", key=params_key)

        is_idx = any(str(t).endswith(".JK") for t in _tickers(ticker_text)) or "Indonesia" in universe
        col_c1, col_c2 = st.columns(2)
        buy_cost = col_c1.number_input(
            "Fee Beli (%)", min_value=0.0, max_value=5.0,
            value=0.16 if is_idx else 0.10, step=0.01, key="cfg_buy_cost",
        )
        sell_cost = col_c2.number_input(
            "Fee Jual + Pajak (%)", min_value=0.0, max_value=5.0,
            value=0.35 if is_idx else 0.10, step=0.01, key="cfg_sell_cost",
            help="Di IDX mencakup komisi broker (~0.25%) + PPh Final pasal 4(2) 0.1% + levy.",
        )
        col_l1, col_l2 = st.columns(2)
        lot_sz = col_l1.number_input(
            "Satuan Lot (lembar)", min_value=1,
            value=100 if is_idx else 1, step=1, key="cfg_lot_size",
            help="Di BEI 1 lot = 100 lembar saham.",
        )
        stop_loss = col_l2.number_input(
            "Hard Stop-Loss per Posisi (%)", min_value=0.0, max_value=50.0,
            value=0.0, step=0.5, key="cfg_stop_loss",
            help="Likuidasi otomatis jika posisi rugi melebihi ambang batas. 0 = nonaktif.",
        )

    return ExperimentConfig(
        tickers=_tickers(ticker_text), data_source=source, interval=interval,
        train_start=train_start, train_end=train_end, test_start=test_start,
        test_end=test_end, indicators=indicators, use_vix=use_vix, drl_lib=library,
        model_name=model, model_path=model_path, timesteps=int(timesteps),
        agent_params=_parse_json(params, "Parameter agent"), universe=universe,
        risk_free_rate=float(risk_free_rate) / 100,
        buy_cost_pct=float(buy_cost) / 100,
        sell_cost_pct=float(sell_cost) / 100,
        lot_size=int(lot_sz),
        stop_loss_pct=float(stop_loss) / 100,
    )


def run_training(config: ExperimentConfig, source_kwargs: dict[str, str]) -> Any:
    from finrl.train import train

    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    shared = dict(
        source_kwargs,
        cwd=config.model_path,
        buy_cost_pct=config.buy_cost_pct,
        sell_cost_pct=config.sell_cost_pct,
        lot_size=config.lot_size,
        stop_loss_pct=config.stop_loss_pct,
    )
    if config.drl_lib == "stable_baselines3":
        shared.update(agent_params=config.agent_params, total_timesteps=config.timesteps)
    elif config.drl_lib == "elegantrl":
        shared.update(erl_params=config.agent_params, break_step=config.timesteps)
    else:
        shared.update(rllib_params=config.agent_params, total_episodes=max(1, config.timesteps // 1_000))
    trained_model = train(
        config.train_start, config.train_end, config.tickers, config.data_source,
        config.interval, config.indicators, config.drl_lib, StockTradingEnv,
        config.model_name, config.use_vix, **shared,
    )
    # Save a secret-free sidecar manifest so a model can later be reconstructed
    # with the same ticker ordering, indicators, risk feature, and dimensions.
    manifest_path = Path(f"{config.model_path}.config.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    st.session_state["model_manifest_path"] = str(manifest_path.resolve())
    return trained_model


def run_backtest(config: ExperimentConfig, source_kwargs: dict[str, str]) -> list[float]:
    from finrl.test import test

    values = test(
        config.test_start, config.test_end, config.tickers, config.data_source,
        config.interval, config.indicators, config.drl_lib, StockTradingEnv,
        config.model_name, config.use_vix, cwd=config.model_path,
        buy_cost_pct=config.buy_cost_pct,
        sell_cost_pct=config.sell_cost_pct,
        lot_size=config.lot_size,
        stop_loss_pct=config.stop_loss_pct,
        **source_kwargs,
    )
    return list(values)


def _resolve_model_path(model_path: str) -> Path | None:
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        return path
    zipped = path.with_suffix(".zip")
    return zipped if zipped.exists() else None


def _load_manifest(model_path: str) -> dict[str, Any] | None:
    path = Path(f"{model_path}.config.json")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_mismatches(config: ExperimentConfig, manifest: dict[str, Any]) -> list[str]:
    current = asdict(config)
    fields = (
        "tickers", "data_source", "interval", "indicators", "use_vix",
        "drl_lib", "model_name",
    )
    return [field for field in fields if manifest.get(field) != current.get(field)]


def _verify_model_load(config: ExperimentConfig, path: Path) -> None:
    if config.drl_lib != "stable_baselines3":
        # ElegantRL and RLlib checkpoint loading needs a fully constructed
        # environment. The backtest performs that final deserialization check.
        return
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    models = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "sac": SAC, "td3": TD3}
    models[config.model_name].load(str(path))


def _build_evaluation_report(
    values: list[float], config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = pd.Series(values, dtype=float)
    strategy = strategy / strategy.iloc[0] * 100
    comparison = pd.DataFrame({"AI strategy": strategy})

    data = st.session_state.get("market_data")
    if data is not None and not data.empty:
        date_col = "timestamp" if "timestamp" in data.columns else "date"
        if {date_col, "tic", "close"}.issubset(data.columns):
            frame = data.copy()
            frame[date_col] = pd.to_datetime(frame[date_col])
            mask = frame[date_col].between(config.test_start, config.test_end)
            prices = frame.loc[mask].pivot_table(
                index=date_col, columns="tic", values="close", aggfunc="last"
            ).sort_index().ffill()
            if not prices.empty:
                normalized = prices.div(prices.iloc[0]).mean(axis=1) * 100
                length = min(len(comparison), len(normalized))
                comparison = comparison.iloc[:length].copy()
                comparison.index = normalized.index[:length]
                comparison["Benchmark equal-weight"] = normalized.iloc[:length].values

    drawdown = comparison.div(comparison.cummax()).sub(1).mul(100)
    factor = {
        "1D": 252,
        "1H": 252 * 6,
        "15Min": 252 * 24,
        "5Min": 252 * 72,
        "1Min": 252 * 360,
    }.get(config.interval, 252)
    metrics: dict[str, dict[str, float]] = {}
    for column in comparison:
        returns = comparison[column].pct_change().dropna()
        total_return = comparison[column].iloc[-1] / comparison[column].iloc[0] - 1
        volatility = returns.std() * np.sqrt(factor) if len(returns) > 1 else np.nan
        annual_return = returns.mean() * factor if len(returns) else np.nan
        sharpe = (
            (annual_return - config.risk_free_rate) / volatility
            if volatility and np.isfinite(volatility)
            else np.nan
        )
        metrics[column] = {
            "Total return (%)": total_return * 100,
            "Annual volatility (%)": volatility * 100,
            "Max drawdown (%)": drawdown[column].min(),
            "Sharpe": sharpe,
            "Observations": float(len(comparison)),
        }
    return comparison, pd.DataFrame(metrics).T


def show_home(config: ExperimentConfig) -> None:
    artifact = _resolve_model_path(config.model_path)
    data = st.session_state.get("market_data")
    equity = st.session_state.get("equity_curve")
    st.markdown(
        """
        <div class="nezu-hero">
          <div class="nezu-hero-kicker">WORKSPACE AKTIF · HUMAN IN CONTROL</div>
          <h2>Apa yang patut ditinjau hari ini?</h2>
          <p>NEZU menyatukan data, bukti teknikal, reinforcement learning, dan konteks riset.
          Setiap output adalah dukungan keputusan—bukan instruksi beli atau jual otomatis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_a, action_b, action_space = st.columns((1, 1, 3.5))
    action_a.button(
        "Review market", type="primary", use_container_width=True,
        on_click=_go_to, args=("Analisa IDX",),
    )
    action_b.button(
        "Run analysis", use_container_width=True,
        on_click=_go_to, args=("Data Market",),
    )

    a, b, c, d = st.columns(4)
    a.markdown(
        _status_card("Market universe", config.universe, f"{len(config.tickers)} ticker dalam konfigurasi"),
        unsafe_allow_html=True,
    )
    b.markdown(
        _status_card(
            "Data quality", "Siap dianalisis" if data is not None else "Menunggu data",
            "Dataset tersedia di memori sesi" if data is not None else "Muat dan validasi sebelum eksperimen",
        ),
        unsafe_allow_html=True,
    )
    c.markdown(
        _status_card(
            "Active model", config.model_name.upper() if artifact else "Belum tersedia",
            "Artefak ditemukan · research only" if artifact else "Training atau load model diperlukan",
        ),
        unsafe_allow_html=True,
    )
    d.markdown(
        _status_card(
            "Evaluation", "OOS tersedia" if equity is not None else "Belum dijalankan",
            "Equity curve siap direview" if equity is not None else "Jalankan backtest out-of-sample",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Alur keputusan & eksperimen")
    st.markdown(
        """
        <div class="nezu-workflow">
          <div class="nezu-step"><b>1</b><span>Konfigurasi</span></div>
          <div class="nezu-step"><b>2</b><span>Data & QA</span></div>
          <div class="nezu-step"><b>3</b><span>Analisa</span></div>
          <div class="nezu-step"><b>4</b><span>Train RL</span></div>
          <div class="nezu-step"><b>5</b><span>Model registry</span></div>
          <div class="nezu-step"><b>6</b><span>Evaluasi OOS</span></div>
          <div class="nezu-step"><b>7</b><span>Paper test</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Status kesiapan")
    status = pd.DataFrame(
        {
            "Tahap": ["Konfigurasi", "Data", "Model", "Evaluasi"],
            "Status": [
                "Siap",
                "Siap" if data is not None else "Perlu dijalankan",
                "Siap" if artifact else "Perlu training/load",
                "Siap" if equity else "Perlu backtest",
            ],
            "Langkah berikut": [
                "Review config aktif",
                "Buka Data & Quality",
                "Buka Experiment Lab atau Models & Evaluation",
                "Jalankan backtest out-of-sample",
            ],
        }
    )
    st.dataframe(status, hide_index=True, use_container_width=True)


def show_data(config: ExperimentConfig, source_kwargs: dict[str, str]) -> None:
    st.subheader("Data dan sinyal")
    if st.button("Muat & validasi data", type="primary", use_container_width=True):
        try:
            with st.spinner("Mengunduh dan memproses data..."):
                data = load_market_data(
                    tuple(config.tickers), config.data_source, config.train_start,
                    config.test_end, config.interval, tuple(config.indicators),
                    config.use_vix, tuple(sorted(source_kwargs.items())),
                )
            st.session_state["market_data"] = data
            notify_success(f"{len(data):,} baris siap digunakan untuk eksperimen.")
        except Exception as error:
            notify_error(error, title="Gagal Memuat Data")
    data = st.session_state.get("market_data")
    if data is not None:
        date_col = "timestamp" if "timestamp" in data.columns else "date"
        if {date_col, "tic"}.issubset(data.columns):
            coverage = data.groupby("tic").agg(
                start=(date_col, "min"),
                end=(date_col, "max"),
                observations=(date_col, "count"),
                missing_close=("close", lambda values: int(values.isna().sum())),
            )
            a, b, c = st.columns(3)
            a.metric("Baris", f"{len(data):,}")
            b.metric("Ticker tersedia", data["tic"].nunique())
            c.metric("Missing close", int(data["close"].isna().sum()))
            with st.expander("Data quality & coverage", expanded=True):
                st.dataframe(coverage, use_container_width=True)

        table_tab, history_tab = st.tabs(("Preview data", "Histori per ticker"))
        with table_tab:
            st.dataframe(data.tail(100), use_container_width=True)
        with history_tab:
            if "tic" in data.columns:
                selected_ticker = st.selectbox(
                    "Pilih ticker", sorted(data["tic"].dropna().unique()), key="history_ticker"
                )
                ticker_data = data[data["tic"] == selected_ticker].copy()
                if date_col in ticker_data.columns:
                    ticker_data = ticker_data.set_index(date_col)
                chart_columns = [
                    column for column in ("close", "rsi_14", "rsi_30", "macd")
                    if column in ticker_data.columns
                ]
                if chart_columns:
                    st.line_chart(ticker_data[chart_columns])
                st.dataframe(ticker_data.tail(250), use_container_width=True)
        st.download_button(
            "Unduh seluruh data pasar (CSV)",
            data.to_csv(index=False).encode("utf-8"),
            "nezu_market_data.csv",
            "text/csv",
        )
        if {date_col, "tic", "close"}.issubset(data.columns):
            chart = data.pivot_table(index=date_col, columns="tic", values="close", aggfunc="last")
            st.line_chart(chart)


def show_provider_hub(config: ExperimentConfig) -> None:
    """Configure and compare research, licensed, and broker quote sources."""
    from finrl.db.bridge import load_provider_store as load_provider_bridge
    from finrl.db.bridge import save_provider_store as save_provider_bridge
    from finrl.integrations.provider_adapter import GenericRESTProvider
    from finrl.integrations.provider_adapter import ProviderProfile
    from finrl.integrations.provider_adapter import yahoo_research_quote
    from finrl.integrations.secure_store import save_secret

    path = CONFIG_DIR / "providers.json"
    try:
        store = load_provider_bridge(CONFIG_DIR)
        profiles = [
            ProviderProfile.from_dict(item) for item in store.get("providers", [])
        ]
    except Exception as error:
        notify_error(error, title="Konfigurasi Provider Error")
        return
    by_id = {profile.id: profile for profile in profiles}

    st.subheader("Licensed market data & broker gateway")
    st.caption(
        "Satu ticker aktif dapat dibandingkan pada seluruh sumber. Harga berlisensi, "
        "quote broker, dan sumber research sengaja diberi tier berbeda agar provenance "
        "dan kelayakan eksekusinya tidak tercampur."
    )
    selected_ticker = st.selectbox(
        "Ticker aktif", config.tickers, key="provider_selected_ticker",
        help="Suffix .JK otomatis dapat dihapus untuk provider yang memakai simbol IDX murni.",
    )

    snapshot_tab, profiles_tab, architecture_tab = st.tabs(
        ("Quote & health", "Provider profiles", "Production gate")
    )
    with snapshot_tab:
        market_ids = [profile.id for profile in profiles if profile.kind == "market_data"]
        broker_ids = [profile.id for profile in profiles if profile.kind == "broker"]
        col1, col2 = st.columns(2)
        market_options = [""] + market_ids
        broker_options = [""] + broker_ids
        active_market = col1.selectbox(
            "Primary market-data source", market_options,
            index=market_options.index(store.get("active_market_data", ""))
            if store.get("active_market_data", "") in market_ids else 0,
            format_func=lambda value: by_id[value].name if value in by_id else "Belum dipilih",
        )
        active_broker = col2.selectbox(
            "Primary broker", broker_options,
            index=broker_options.index(store.get("active_broker", ""))
            if store.get("active_broker", "") in broker_ids else 0,
            format_func=lambda value: by_id[value].name if value in by_id else "Belum dipilih",
        )
        if st.button("Simpan source priority", use_container_width=True):
            store["active_market_data"] = active_market
            store["active_broker"] = active_broker
            save_provider_bridge(CONFIG_DIR, store)
            notify_success("Source priority tersimpan (PostgreSQL + file mirror).")

        include_yahoo = st.toggle(
            "Sertakan Yahoo sebagai pembanding research", value=True,
            help="Tidak pernah dipromosikan menjadi licensed/executable source.",
        )
        selected_sources = st.multiselect(
            "Provider yang diuji", [profile.id for profile in profiles],
            default=[profile.id for profile in profiles],
            format_func=lambda value: f"{by_id[value].name} · {by_id[value].tier}",
        )
        if st.button("Ambil snapshot ticker", type="primary", use_container_width=True):
            rows: list[dict[str, Any]] = []
            errors: list[str] = []
            if include_yahoo:
                try:
                    rows.append(yahoo_research_quote(selected_ticker).as_dict())
                except Exception as error:
                    errors.append(f"Yahoo research: {error}")
            for provider_id in selected_sources:
                profile = by_id[provider_id]
                try:
                    rows.append(
                        GenericRESTProvider(profile).get_quote(selected_ticker).as_dict()
                    )
                except Exception as error:
                    errors.append(f"{profile.name}: {error}")
            if rows:
                result = pd.DataFrame(rows)
                trusted = result[
                    result["tier"].isin(["licensed", "executable"])
                    & result["status"].eq("fresh")
                ]
                if not trusted.empty:
                    reference = float(trusted.iloc[0]["last"])
                    result["deviation_vs_primary_pct"] = (
                        (result["last"] / reference - 1) * 100
                    ).round(4)
                else:
                    result["deviation_vs_primary_pct"] = np.nan
                    st.warning(
                        "Belum ada quote fresh dari source licensed/executable. "
                        "Snapshot ini belum lolos production data gate."
                    )
                st.session_state["provider_quote_snapshot"] = result
            st.session_state["provider_quote_errors"] = errors

        result = st.session_state.get("provider_quote_snapshot")
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(
                result[[
                    "provider_name", "tier", "provider_symbol", "timestamp", "last",
                    "bid", "ask", "volume", "age_seconds", "status",
                    "deviation_vs_primary_pct",
                ]], hide_index=True, use_container_width=True,
            )
        for error in st.session_state.get("provider_quote_errors", []):
            st.error(error)

        if active_broker and st.button("Uji account broker (read-only)"):
            try:
                account = GenericRESTProvider(by_id[active_broker]).get_account()
                notify_success("Autentikasi account broker berhasil.")
                st.json(account)
            except Exception as error:
                notify_error(error, title="Autentikasi Broker Gagal")
        st.info(
            "Order placement sengaja belum tersedia. Aktivasi order memerlukan adapter "
            "khusus vendor, sandbox, idempotency key, pre-trade limits, audit trail, "
            "rekonsiliasi, dan kill switch."
        )

    with profiles_tab:
        existing_ids = [profile.id for profile in profiles]
        edit_id = st.selectbox(
            "Edit profile", ["__new__"] + existing_ids,
            format_func=lambda value: "Profil baru" if value == "__new__" else by_id[value].name,
        )
        current = by_id.get(edit_id)
        default_quote_mapping = {
            "symbol": "symbol", "timestamp": "timestamp", "last": "last",
            "bid": "bid", "ask": "ask", "volume": "volume",
            "currency": "currency", "exchange": "exchange",
        }
        default_history_mapping = {
            "timestamp": "timestamp", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        }
        form_key = edit_id.replace("-", "_")
        with st.form(f"provider_profile_{form_key}"):
            left, right = st.columns(2)
            provider_id = left.text_input(
                "Provider ID", value=current.id if current else "", disabled=current is not None,
                placeholder="contoh: idx_licensed_primary",
            )
            provider_name = right.text_input(
                "Nama provider", value=current.name if current else "",
                placeholder="Nama sesuai kontrak",
            )
            kinds = ("market_data", "broker", "research")
            tiers = ("licensed", "executable", "research")
            kind = left.selectbox(
                "Jenis", kinds, index=kinds.index(current.kind) if current else 0,
            )
            tier = right.selectbox(
                "Trust tier", tiers, index=tiers.index(current.tier) if current else 0,
            )
            base_url = st.text_input(
                "Base URL resmi", value=current.base_url if current else "",
                placeholder="https://api.vendor-resmi.example",
            )
            quote_path = left.text_input(
                "Quote path", value=current.quote_path if current else "",
                placeholder="/v1/quotes/{symbol}",
            )
            history_path = right.text_input(
                "History path", value=current.history_path if current else "",
                placeholder="/v1/history/{symbol}",
            )
            account_path = st.text_input(
                "Account path (broker, opsional)", value=current.account_path if current else "",
                placeholder="/v1/account",
            )
            auth_types = ("bearer", "header", "query", "none")
            auth_type = left.selectbox(
                "Auth type", auth_types,
                index=auth_types.index(current.auth_type) if current else 0,
            )
            auth_header = right.text_input(
                "Auth header / query name", value=current.auth_header if current else "Authorization",
            )
            auth_prefix = left.text_input(
                "Auth prefix", value=current.auth_prefix if current else "Bearer ",
            )
            secret_name = right.text_input(
                "Credential vault key", value=current.secret_name if current else "",
                placeholder="provider_vendor_token",
            )
            secret_value = st.text_input(
                "API token baru (opsional; disimpan di Keychain)", type="password",
            )
            strip_suffix = left.text_input(
                "Hapus suffix simbol", value=current.strip_suffix if current else ".JK",
            )
            timeout = right.number_input(
                "Timeout (detik)", min_value=1, max_value=120,
                value=current.timeout_seconds if current else 15,
            )
            stale_after = left.number_input(
                "Stale setelah (detik)", min_value=1, max_value=86400,
                value=current.stale_after_seconds if current else 120,
            )
            quote_root = right.text_input(
                "Quote JSON root", value=current.quote_root if current else "",
                placeholder="data.quote",
            )
            history_root = left.text_input(
                "History JSON root", value=current.history_root if current else "",
                placeholder="data.bars",
            )
            account_root = right.text_input(
                "Account JSON root", value=current.account_root if current else "",
                placeholder="data.account",
            )
            quote_mapping_text = st.text_area(
                "Quote field mapping (JSON)",
                json.dumps(current.quote_mapping if current else default_quote_mapping, indent=2),
            )
            history_mapping_text = st.text_area(
                "History OHLCV mapping (JSON)",
                json.dumps(current.history_mapping if current else default_history_mapping, indent=2),
            )
            history_params_text = st.text_area(
                "History parameter mapping (JSON)",
                json.dumps(
                    current.history_params if current else {
                        "start": "start", "end": "end", "interval": "interval"
                    }, indent=2,
                ),
            )
            submitted = st.form_submit_button(
                "Validasi & simpan profile", use_container_width=True
            )
        if submitted:
            try:
                profile = ProviderProfile(
                    id=current.id if current else provider_id.strip(), name=provider_name.strip(),
                    kind=kind, tier=tier, base_url=base_url.strip(),
                    quote_path=quote_path.strip(), history_path=history_path.strip(),
                    account_path=account_path.strip(), auth_type=auth_type,
                    auth_header=auth_header.strip(), auth_prefix=auth_prefix,
                    secret_name=secret_name.strip(), strip_suffix=strip_suffix.strip(),
                    timeout_seconds=int(timeout), stale_after_seconds=int(stale_after),
                    quote_root=quote_root.strip(), history_root=history_root.strip(),
                    account_root=account_root.strip(),
                    quote_mapping=_parse_json(quote_mapping_text, "Quote mapping"),
                    history_mapping=_parse_json(history_mapping_text, "History mapping"),
                    history_params=_parse_json(history_params_text, "History params"),
                )
                profile.validate()
                items = [
                    item for item in store.get("providers", [])
                    if item.get("id") != profile.id
                ]
                items.append(asdict(profile))
                store["providers"] = items
                save_provider_bridge(CONFIG_DIR, store)
                if secret_value:
                    if not profile.secret_name:
                        raise ValueError("Credential vault key wajib diisi untuk menyimpan token.")
                    save_secret(profile.secret_name, secret_value)
                notify_success(f"Profile {profile.name} tersimpan tanpa menulis secret ke JSON.")
                st.rerun()
            except Exception as error:
                notify_error(error, title="Gagal Menyimpan Profile")

    with architecture_tab:
        st.graphviz_chart(
            """
            digraph {
              rankdir=LR;
              ticker [label="Ticker aktif (.JK)"];
              normalize [label="Symbol mapper"];
              research [label="Research source"];
              licensed [label="Licensed feed"];
              broker [label="Broker quote/account"];
              gate [label="Freshness + deviation gate"];
              rl [label="NEZU RL engine (FinRL core)"];
              decision [label="Decision support"];
              ticker -> normalize;
              normalize -> research;
              normalize -> licensed;
              normalize -> broker;
              research -> gate [style=dashed];
              licensed -> gate;
              broker -> gate;
              licensed -> rl;
              gate -> decision;
              rl -> decision;
            }
            """,
            use_container_width=True,
        )
        st.markdown(
            """
**Production gate** dinyatakan siap hanya bila kontrak data mengizinkan penggunaan
yang dimaksud, quote memiliki timestamp dan tidak stale, mapping OHLCV tervalidasi,
corporate action/timezone/calendar sudah diuji, serta fallback tidak menaikkan
source research menjadi executable. Broker tetap read-only sampai adapter khusus
vendor melewati sandbox, risk limits, audit, reconciliation, dan kill-switch test.
            """
        )


def show_idx_analysis(config: ExperimentConfig) -> None:
    from finrl.analytics.indonesia import build_idx_analysis

    st.subheader("Analisa supermasif saham Indonesia")
    st.caption(
        "Screening lintas saham, momentum, risiko, likuiditas, trend, korelasi, "
        "drawdown, dan market breadth. Skor adalah alat riset, bukan rekomendasi investasi."
    )
    data = st.session_state.get("market_data")
    if data is None:
        st.info("Muat data pada tab Data terlebih dahulu.")
        return
    try:
        analysis = build_idx_analysis(data, risk_free_rate=config.risk_free_rate)
    except Exception as error:
        notify_error(error, title="Gagal Menjalankan Analisis IDX")
        return

    screener = analysis.screener.copy()
    if len(screener) == 1:
        top_n = 1
        st.info(
            "Mode analisis satu ticker aktif. Tambahkan ticker lain untuk "
            "mengaktifkan ranking dan perbandingan lintas saham."
        )
    else:
        minimum_top = 1 if len(screener) < 5 else 5
        top_n = st.slider(
            "Jumlah saham ditampilkan",
            min_value=minimum_top,
            max_value=len(screener),
            value=min(20, len(screener)),
        )
    minimum_value = st.number_input(
        "Minimum rata-rata nilai transaksi 20 hari (Rp)", min_value=0.0,
        value=0.0, step=1_000_000_000.0, format="%.0f",
    )
    filtered = screener[screener["avg_value_idr_20d"] >= minimum_value].head(top_n)
    if filtered.empty:
        st.warning("Tidak ada saham yang memenuhi filter likuiditas.")
        return

    latest_breadth = analysis.breadth.iloc[-1]
    a, b, c, d = st.columns(4)
    a.metric("Saham dianalisis", f"{len(screener)}")
    b.metric("Advancers", f"{latest_breadth['advancers_pct']:.1f}%")
    c.metric("Di atas SMA20", f"{latest_breadth['above_sma20_pct']:.1f}%")
    d.metric("Di atas SMA50", f"{latest_breadth['above_sma50_pct']:.1f}%")

    rank_tab, performance_tab, risk_tab, correlation_tab, breadth_tab = st.tabs(
        ("Ranking", "Performa", "Risk", "Korelasi", "Breadth")
    )
    with rank_tab:
        display_columns = [
            "composite_score", "price", "return_1m", "return_3m", "return_6m",
            "return_12m", "sharpe", "sortino", "annual_volatility",
            "max_drawdown", "avg_value_idr_20d", "above_sma20", "above_sma50",
            "above_sma200",
        ]
        st.dataframe(filtered[display_columns], use_container_width=True)
        st.bar_chart(filtered["composite_score"])
        st.download_button(
            "Unduh screener CSV", screener.to_csv().encode("utf-8"),
            "idx_screener.csv", "text/csv",
        )
    with performance_tab:
        selected = st.multiselect(
            "Bandingkan saham", filtered.index.tolist(), default=filtered.index[:5].tolist(),
        )
        if selected:
            st.line_chart(analysis.normalized_prices[selected].dropna(how="all"))
    with risk_tab:
        risk_columns = [
            "annual_volatility", "max_drawdown", "daily_var_95", "daily_cvar_95",
            "sharpe", "sortino",
        ]
        st.dataframe(filtered[risk_columns], use_container_width=True)
        st.line_chart(analysis.drawdown[filtered.index[:10]].dropna(how="all"))
    with correlation_tab:
        correlation_names = filtered.index[:20]
        matrix = analysis.correlation.loc[correlation_names, correlation_names].round(3)
        heatmap_data = (
            matrix.rename_axis("Ticker Y")
            .reset_index()
            .melt(id_vars="Ticker Y", var_name="Ticker X", value_name="Korelasi")
        )
        base = alt.Chart(heatmap_data).encode(
            x=alt.X("Ticker X:N", sort=correlation_names.tolist(), title=None),
            y=alt.Y("Ticker Y:N", sort=correlation_names.tolist(), title=None),
            tooltip=["Ticker X:N", "Ticker Y:N", alt.Tooltip("Korelasi:Q", format=".3f")],
        )
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "Korelasi:Q",
                scale=alt.Scale(domain=[-1, 0, 1], range=["#2166ac", "#f7f7f7", "#b2182b"]),
                title="Korelasi",
            )
        )
        labels = base.mark_text(fontSize=11).encode(
            text=alt.Text("Korelasi:Q", format=".2f"),
            color=alt.condition(
                "abs(datum.Korelasi) > 0.55", alt.value("white"), alt.value("black")
            ),
        )
        st.altair_chart((heatmap + labels).properties(height=520), use_container_width=True)
        st.caption("Merah = bergerak searah, biru = berlawanan, putih = hubungan linear lemah.")
        with st.expander("Lihat matriks angka"):
            st.dataframe(matrix, use_container_width=True)
    with breadth_tab:
        st.line_chart(analysis.breadth)


def show_training(config: ExperimentConfig, source_kwargs: dict[str, str]) -> None:
    st.subheader("Training")
    st.caption("Model disimpan ke path yang ditentukan pada konfigurasi.")
    if st.button("Mulai training", type="primary"):
        try:
            with st.spinner("Training sedang berjalan..."):
                st.session_state["trained_model"] = run_training(config, source_kwargs)
            notify_success(f"Training selesai. Artefak tersedia di {config.model_path}.")
            manifest_path = st.session_state.get("model_manifest_path")
            if manifest_path:
                st.caption(f"Konfigurasi model disimpan di {manifest_path}")
        except Exception as error:
            notify_error(error, title="Training Gagal")


def show_backtest(config: ExperimentConfig, source_kwargs: dict[str, str]) -> None:
    st.subheader("Model & evaluasi out-of-sample")
    st.caption(
        "Load dan validasi artefak model, lalu ukur hasil inferensi strategi terhadap benchmark."
    )
    artifact = _resolve_model_path(config.model_path)
    manifest = _load_manifest(config.model_path)
    mismatches = _manifest_mismatches(config, manifest) if manifest else []

    a, b, c = st.columns(3)
    a.metric("Artefak model", "Ditemukan" if artifact else "Tidak ditemukan")
    b.metric("Manifest", "Ditemukan" if manifest else "Tidak tersedia")
    c.metric("Kompatibilitas", "Cocok" if manifest and not mismatches else "Perlu dicek")

    if artifact:
        st.code(str(artifact), language=None)
    if mismatches:
        st.error(
            "Konfigurasi aktif berbeda dari manifest pada: " + ", ".join(mismatches)
            + ". Load manifest model sebelum menjalankan evaluasi."
        )
    elif manifest:
        st.success("Ticker, indikator, interval, risk feature, library, dan model cocok.")
    else:
        st.warning(
            "Manifest model tidak ditemukan. Model masih dapat diuji, tetapi kompatibilitas "
            "konfigurasinya tidak dapat dibuktikan otomatis."
        )

    load_col, test_col = st.columns(2)
    with load_col:
        load_clicked = st.button(
            "Load & validasi model",
            type="primary",
            disabled=artifact is None or bool(mismatches),
            use_container_width=True,
        )
    if load_clicked and artifact:
        try:
            with st.spinner("Memvalidasi artefak model..."):
                _verify_model_load(config, artifact)
            st.session_state["loaded_model_path"] = str(artifact)
            notify_success("Model berhasil dimuat dan dipilih sebagai model aktif.")
        except Exception as error:
            notify_error(error, title="Gagal Memuat Model")

    model_ready = st.session_state.get("loaded_model_path") == str(artifact)
    with test_col:
        run_clicked = st.button(
            "Jalankan evaluasi",
            disabled=artifact is None or bool(mismatches),
            use_container_width=True,
        )
    if run_clicked:
        try:
            with st.spinner("Menjalankan evaluasi..."):
                values = run_backtest(config, source_kwargs)
            st.session_state["equity_curve"] = values
            st.session_state["loaded_model_path"] = str(artifact)
            notify_success("Evaluasi out-of-sample selesai.")
        except Exception as error:
            notify_error(error, title="Evaluasi Gagal")

    if model_ready:
        st.caption("Model aktif telah melewati validasi loader pada sesi ini.")
    values = st.session_state.get("equity_curve")
    if values:
        comparison, metrics = _build_evaluation_report(values, config)
        strategy_return = metrics.loc["AI strategy", "Total return (%)"]
        strategy_drawdown = metrics.loc["AI strategy", "Max drawdown (%)"]
        strategy_sharpe = metrics.loc["AI strategy", "Sharpe"]
        a, b, c = st.columns(3)
        a.metric("Return strategi", f"{strategy_return:.2f}%")
        b.metric("Max drawdown", f"{strategy_drawdown:.2f}%")
        c.metric("Sharpe", f"{strategy_sharpe:.2f}" if np.isfinite(strategy_sharpe) else "N/A")

        performance_tab, risk_tab, evidence_tab = st.tabs(
            ("Performa vs benchmark", "Drawdown", "Bukti kelayakan")
        )
        with performance_tab:
            st.info(
                "Grafik ini adalah hasil inferensi kebijakan trading pada data test, "
                "bukan prediksi harga saham masa depan."
            )
            st.line_chart(comparison)
            st.dataframe(metrics.round(3), use_container_width=True)
        with risk_tab:
            drawdown = comparison.div(comparison.cummax()).sub(1).mul(100)
            st.line_chart(drawdown)
        with evidence_tab:
            checks = pd.DataFrame(
                {
                    "Pemeriksaan": [
                        "Artefak model tersedia",
                        "Manifest kompatibel",
                        "Periode test setelah train",
                        "Benchmark tersedia",
                        "Observasi memadai (>= 60)",
                        "Tidak ada nilai non-finite",
                    ],
                    "Status": [
                        bool(artifact),
                        bool(manifest) and not mismatches,
                        pd.Timestamp(config.test_start) >= pd.Timestamp(config.train_end),
                        "Benchmark equal-weight" in comparison.columns,
                        len(comparison) >= 60,
                        np.isfinite(comparison.to_numpy()).all(),
                    ],
                }
            )
            checks["Hasil"] = checks["Status"].map({True: "Lulus", False: "Perlu perhatian"})
            st.dataframe(checks[["Pemeriksaan", "Hasil"]], hide_index=True, use_container_width=True)
            if checks["Status"].all():
                st.success("Seluruh pemeriksaan dasar lulus. Tetap lakukan multi-period dan paper test.")
            else:
                st.warning("Hasil belum memenuhi seluruh pemeriksaan dasar kelayakan.")

        frame = pd.DataFrame({"portfolio_value": values})
        frame["return_pct"] = (frame["portfolio_value"] / frame["portfolio_value"].iloc[0] - 1) * 100
        st.download_button(
            "Unduh hasil backtest (CSV)",
            frame.to_csv(index=False).encode("utf-8"),
            "nezu_backtest.csv",
            "text/csv",
        )


def _build_ai_research_context(
    config: ExperimentConfig, ticker: str, external_context: str
) -> str:
    sections = [
        "KONFIGURASI EKSPERIMEN",
        json.dumps(
            {
                "ticker_focus": ticker,
                "universe": config.universe,
                "interval": config.interval,
                "train_period": [config.train_start, config.train_end],
                "test_period": [config.test_start, config.test_end],
                "indicators": config.indicators,
                "risk_feature": "VIX" if config.use_vix else "turbulence",
                "rl_library": config.drl_lib,
                "rl_model": config.model_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
    ]

    data = st.session_state.get("market_data")
    if data is not None and not data.empty and "tic" in data.columns:
        ticker_data = data[data["tic"] == ticker].copy()
        if not ticker_data.empty:
            date_col = "timestamp" if "timestamp" in ticker_data.columns else "date"
            ticker_data = ticker_data.sort_values(date_col)
            latest = ticker_data.iloc[-1]
            fields = [
                "close", "volume", "rsi_14", "rsi_30", "macd", "boll_ub",
                "boll_lb", "close_30_sma", "close_60_sma",
            ]
            snapshot = {
                "last_timestamp": str(latest.get(date_col)),
                **{
                    field: float(latest[field])
                    for field in fields
                    if field in latest and pd.notna(latest[field])
                },
                "observations": int(len(ticker_data)),
            }
            sections.extend(
                ["SNAPSHOT TEKNIKAL LOKAL (data yang dimuat pengguna)", json.dumps(snapshot, indent=2)]
            )

    values = st.session_state.get("equity_curve")
    if values:
        _, metrics = _build_evaluation_report(values, config)
        strategy_metrics = {
            key: None if pd.isna(value) else float(value)
            for key, value in metrics.loc["AI strategy"].items()
        }
        sections.extend(
            ["HASIL MODEL RL OUT-OF-SAMPLE", json.dumps(strategy_metrics, indent=2)]
        )

    if external_context.strip():
        sections.extend(
            [
                "KONTEKS EKSTERNAL DARI PENGGUNA (berita/fundamental/sentimen)",
                external_context.strip(),
            ]
        )
    else:
        sections.extend(
            [
                "KONTEKS EKSTERNAL",
                "Tidak diberikan. Jangan mengarang berita, fundamental, harga, atau peristiwa terbaru.",
            ]
        )
    company_context = st.session_state.get("ai_company_context")
    if company_context and company_context.get("ticker") == ticker:
        sections.extend(
            [
                "SNAPSHOT FUNDAMENTAL & BERITA DARI YAHOO FINANCE",
                json.dumps(company_context, indent=2, ensure_ascii=False, default=str),
            ]
        )
    return "\n\n".join(sections)


def show_ai_research(config: ExperimentConfig) -> None:
    from finrl.integrations.ai_router import RouterHTTPError
    from finrl.integrations.ai_router import chat_completion
    from finrl.integrations.ai_router import list_models
    from finrl.integrations.secure_store import load_secret
    from finrl.integrations.secure_store import save_secret

    if not st.session_state.get("ai_profile_initialized"):
        profile = _read_json_config("ai_research.json") or {}
        st.session_state.setdefault(
            "ai_base_url", profile.get("base_url", "http://localhost:20128/v1")
        )
        st.session_state.setdefault("ai_model_id", profile.get("model", ""))
        st.session_state.setdefault("ai_saved_model", profile.get("model", ""))
        st.session_state.setdefault("ai_saved_fallbacks", profile.get("fallback_models", []))
        st.session_state.setdefault("ai_temperature", float(profile.get("temperature", 0.2)))
        st.session_state.setdefault("ai_mode", profile.get("mode", "Analisis lengkap"))
        try:
            st.session_state.setdefault(
                "ai_api_key", load_secret("ai_router_api_key") or ""
            )
        except Exception:
            st.session_state.setdefault("ai_api_key", "")
        st.session_state["ai_profile_initialized"] = True

    st.subheader("AI Research Copilot")
    st.caption(
        "Gabungkan metrik teknikal dan hasil reinforcement learning dengan analisis "
        "model cloud melalui gateway OpenAI-compatible."
    )
    st.warning(
        "Copilot adalah alat riset, bukan penasihat keuangan. Verifikasi berita, angka "
        "fundamental, dan klaim terbaru ke sumber primer sebelum mengambil keputusan."
    )

    with st.expander("1. Koneksi AI Router", expanded=True):
        base_url = st.text_input(
            "Base URL",
            key="ai_base_url",
            help="Default 9Router lokal. Untuk OpenRouter gunakan https://openrouter.ai/api/v1.",
        )
        api_key = st.text_input(
            "API key router",
            type="password",
            key="ai_api_key",
            help="Disimpan hanya pada session Streamlit dan tidak masuk manifest/model.",
        )
        if st.button("Test koneksi & muat model", use_container_width=True):
            try:
                with st.spinner("Menghubungi router..."):
                    models = list_models(base_url, api_key)
                st.session_state["ai_available_models"] = models
                if models:
                    notify_success(f"Router terhubung. {len(models)} model tersedia.")
                else:
                    notify_warning("Router terhubung, tetapi daftar model kosong.")
            except Exception as error:
                notify_error(error, title="Koneksi AI Router Gagal")

        available_models = st.session_state.get("ai_available_models", [])
        if available_models:
            saved_model = st.session_state.get("ai_saved_model")
            selected_index = available_models.index(saved_model) if saved_model in available_models else 0
            model = st.selectbox("Model", available_models, index=selected_index)
            saved_fallbacks = [
                item for item in st.session_state.get("ai_saved_fallbacks", [])
                if item in available_models and item != model
            ]
            fallback_models = st.multiselect(
                "Fallback model (urutkan dari prioritas tertinggi)",
                [item for item in available_models if item != model],
                default=saved_fallbacks,
                help="Dicoba berurutan bila model utama ditolak, kehabisan kuota, atau provider error.",
            )
        else:
            model = st.text_input(
                "Model ID",
                key="ai_model_id",
                placeholder="contoh: provider/model atau model route 9Router",
            )
            fallback_text = st.text_input(
                "Fallback model IDs (pisahkan koma)",
                value=", ".join(st.session_state.get("ai_saved_fallbacks", [])),
                placeholder="provider/model-b, provider/model-c",
            )
            fallback_models = [
                item.strip() for item in fallback_text.split(",") if item.strip()
            ]
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.05,
            key="ai_temperature",
            help="Nilai rendah lebih konsisten; nilai tinggi lebih bervariasi dan spekulatif.",
        )

    st.markdown("### 2. Bahan analisis")
    mode = st.selectbox(
        "Mode riset",
        (
            "Analisis lengkap",
            "Berita & katalis",
            "Fundamental",
            "Sentimen",
            "Jelaskan hasil model RL",
            "Risk review",
        ),
        key="ai_mode",
    )
    save_profile_col, save_key_col = st.columns(2)
    if save_profile_col.button("Simpan profil Copilot", use_container_width=True):
        path = _write_json_config(
            "ai_research.json",
            {
                "base_url": base_url,
                "model": model,
                "fallback_models": fallback_models,
                "temperature": temperature,
                "mode": mode,
            },
        )
        st.session_state["ai_saved_model"] = model
        st.session_state["ai_saved_fallbacks"] = fallback_models
        notify_success(f"Profil disimpan di {path} (tanpa API key).")
    if save_key_col.button("Simpan API key ke secure vault", use_container_width=True):
        try:
            save_secret("ai_router_api_key", api_key)
            notify_success("API key AI router tersimpan di credential vault OS.")
        except Exception as error:
            notify_error(error, title="Gagal Menyimpan API Key")
    ticker = st.selectbox("Ticker fokus", config.tickers, key="ai_ticker")
    if st.button("Muat fundamental & berita Yahoo Finance", use_container_width=True):
        try:
            with st.spinner("Mengambil snapshot perusahaan dan headline..."):
                st.session_state["ai_company_context"] = load_company_research_context(ticker)
            context_loaded = st.session_state["ai_company_context"]
            notify_success(
                f"Snapshot dimuat: {len(context_loaded['fundamentals'])} field fundamental, "
                f"{len(context_loaded['news'])} headline."
            )
        except Exception as error:
            notify_error(error, title="Gagal Memuat Fundamental & Berita")
    company_context = st.session_state.get("ai_company_context")
    if company_context and company_context.get("ticker") == ticker:
        with st.expander("Snapshot Yahoo Finance yang aktif"):
            st.json(company_context)
    external_context = st.text_area(
        "Berita, laporan keuangan, tautan/sumber, atau catatan tambahan",
        height=180,
        placeholder=(
            "Tempelkan isi atau ringkasan sumber beserta tanggal dan URL. "
            "Router tidak otomatis memiliki data terbaru atau akses browsing."
        ),
    )
    question = st.text_area(
        "Pertanyaan riset",
        value=(
            f"Analisis {ticker} sebagai informasi tambahan. Pisahkan fakta, inferensi, "
            "risiko, skenario bullish/bearish, dan informasi yang masih perlu diverifikasi."
        ),
        height=120,
    )

    context = _build_ai_research_context(config, ticker, external_context)
    with st.expander("Preview data yang akan dikirim"):
        st.code(context, language="text")
    consent = st.checkbox(
        "Saya memahami ringkasan di atas akan dikirim ke layanan AI/router eksternal."
    )
    if st.button(
        "Jalankan AI research",
        type="primary",
        disabled=not consent or not model,
        use_container_width=True,
    ):
        system_prompt = """
Anda adalah copilot riset finansial berbahasa Indonesia. Gunakan hanya data dan
konteks yang diberikan. Jangan mengarang berita, laporan, harga, tanggal, sumber,
atau kemampuan real-time. Pisahkan dengan tegas: Fakta dari konteks, Inferensi,
Ketidakpastian/data yang hilang, Skenario bullish, Skenario bearish, Risiko, dan
Langkah verifikasi. Jelaskan hubungan dengan hasil reinforcement learning tanpa
menganggap backtest menjamin masa depan. Jangan memberi instruksi beli/jual yang
dipersonalisasi atau menjanjikan keuntungan. Bila konteks eksternal tidak ada,
katakan bahwa analisis berita/fundamental terbaru tidak dapat disimpulkan.
""".strip()
        user_prompt = f"MODE RISET: {mode}\n\n{context}\n\nPERTANYAAN:\n{question}"
        try:
            with st.spinner("AI router sedang menganalisis..."):
                response = None
                failures = []
                for candidate in [model, *fallback_models]:
                    try:
                        response = chat_completion(
                            base_url,
                            api_key,
                            candidate,
                            [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=temperature,
                        )
                        break
                    except RouterHTTPError as error:
                        failures.append(
                            f"{candidate}: HTTP {error.status_code} — {error.detail[:180]}"
                        )
                if response is None:
                    raise RouterHTTPError(
                        503,
                        "Semua model gagal. " + " | ".join(failures),
                    )
            entry = {
                "ticker": ticker,
                "mode": mode,
                "model": response.model,
                "question": question,
                "answer": response.content,
                "usage": response.usage,
                "timestamp": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
            }
            st.session_state.setdefault("ai_research_history", []).append(entry)
            if failures:
                notify_warning(
                    "Model utama gagal; fallback berhasil. Percobaan sebelumnya: "
                    + " | ".join(failures)
                )
            else:
                notify_success(f"Analisis AI selesai menggunakan {response.model}.")
        except RouterHTTPError as error:
            if error.status_code == 403:
                notify_error(
                    "Provider menolak akses model (HTTP 403). Periksa koneksi akun/provider, "
                    "izin model, dan kuota di dashboard 9Router. Pilih model lain atau "
                    "konfigurasikan fallback combo.",
                    title="Akses AI Router Ditolak (403)",
                    details=error.detail[:2000] if error.detail else None,
                )
            else:
                notify_error(
                    f"AI router gagal (HTTP {error.status_code}). Coba model/provider lain "
                    "atau periksa status dan kuota router.",
                    title=f"AI Router Error ({error.status_code})",
                    details=error.detail[:2000] if error.detail else None,
                )
        except Exception as error:
            notify_error(error, title="AI Router Gagal")

    history = st.session_state.get("ai_research_history", [])
    if history:
        st.markdown("### 3. Hasil riset")
        latest = history[-1]
        st.caption(
            f"{latest['ticker']} • {latest['mode']} • {latest['model']} • {latest['timestamp']}"
        )
        st.markdown(latest["answer"])
        export = (
            f"# AI Research — {latest['ticker']}\n\n"
            f"- Mode: {latest['mode']}\n- Model: {latest['model']}\n"
            f"- Waktu: {latest['timestamp']}\n\n"
            f"## Pertanyaan\n\n{latest['question']}\n\n"
            f"## Jawaban\n\n{latest['answer']}\n"
        )
        col1, col2 = st.columns(2)
        col1.download_button(
            "Unduh hasil riset (Markdown)",
            export.encode("utf-8"),
            f"ai_research_{latest['ticker']}.md",
            "text/markdown",
            use_container_width=True,
        )
        if col2.button("Hapus riwayat sesi", use_container_width=True):
            st.session_state["ai_research_history"] = []
            st.rerun()


def show_paper_trading(config: ExperimentConfig) -> None:
    from finrl.integrations.alpaca import PAPER_BASE_URL
    from finrl.integrations.alpaca import normalize_alpaca_paper_url
    from finrl.integrations.alpaca import validate_alpaca_paper_connection
    from finrl.integrations.secure_store import load_secret
    from finrl.integrations.secure_store import save_secret

    st.subheader("Paper trading Alpaca")
    st.warning("Mode ini mengirim order ke akun paper Alpaca. Jangan gunakan kredensial akun live.")
    if any(ticker.endswith(".JK") for ticker in config.tickers):
        st.error(
            "Alpaca Paper tidak dapat mengeksekusi saham IDX (.JK). Gunakan "
            "backtest untuk model IDX atau ganti ke ticker yang didukung Alpaca."
        )
        return
    if config.drl_lib in ("elegantrl", "rllib") and config.model_name != "ppo":
        st.error(
            f"Paper trading {config.drl_lib} saat ini hanya mendukung PPO. "
            "Pilih PPO atau gunakan Stable-Baselines3."
        )
        return
    if not st.session_state.get("paper_profile_initialized"):
        profile = _read_json_config("paper_trading.json") or {}
        stored_api_url = profile.get("api_url", PAPER_BASE_URL)
        try:
            stored_api_url = normalize_alpaca_paper_url(stored_api_url)
        except ValueError:
            # Keep an invalid custom value visible so the user can correct it;
            # valid legacy values ending in /v2 are migrated in memory.
            pass
        st.session_state.setdefault(
            "paper_api_url", stored_api_url
        )
        st.session_state.setdefault(
            "paper_state_dim",
            int(profile.get("state_dim", 1 + 2 + 3 * len(config.tickers) + len(config.indicators) * len(config.tickers))),
        )
        st.session_state.setdefault(
            "paper_action_dim", int(profile.get("action_dim", len(config.tickers)))
        )
        try:
            st.session_state.setdefault("paper_api_key", load_secret("alpaca_paper_api_key") or "")
            st.session_state.setdefault("paper_api_secret", load_secret("alpaca_paper_api_secret") or "")
        except Exception:
            st.session_state.setdefault("paper_api_key", "")
            st.session_state.setdefault("paper_api_secret", "")
        st.session_state["paper_profile_initialized"] = True
    with st.form("paper-trading"):
        api_key = st.text_input("Alpaca API key", type="password", key="paper_api_key")
        api_secret = st.text_input("Alpaca API secret", type="password", key="paper_api_secret")
        api_url = st.text_input(
            "Alpaca paper base URL", key="paper_api_url",
            help=f"Gunakan {PAPER_BASE_URL} tanpa /v2; SDK menambahkan versi API sendiri.",
        )
        state_dim = st.number_input("State dimension", min_value=1, key="paper_state_dim")
        action_dim = st.number_input("Action dimension", min_value=1, key="paper_action_dim")
        save_credentials = st.checkbox(
            "Simpan API key dan secret ke credential vault OS"
        )
        confirmed = st.checkbox("Saya memahami bahwa ini akan membuat order paper-trading.")
        save_only = st.form_submit_button("Simpan konfigurasi")
        test_connection = st.form_submit_button("Uji koneksi read-only")
        submitted = st.form_submit_button("Mulai paper trading", type="primary")
    if save_only:
        try:
            normalized_api_url = normalize_alpaca_paper_url(api_url)
            path = _write_json_config(
                "paper_trading.json",
                {
                    "api_url": normalized_api_url,
                    "state_dim": int(state_dim),
                    "action_dim": int(action_dim),
                    "model_path": config.model_path,
                    "drl_lib": config.drl_lib,
                    "model_name": config.model_name,
                },
            )
            if save_credentials:
                save_secret("alpaca_paper_api_key", api_key)
                save_secret("alpaca_paper_api_secret", api_secret)
            notify_success(f"Konfigurasi paper trading disimpan di {path}.")
        except Exception as error:
            notify_error(error, title="Gagal Menyimpan Konfigurasi")
    if test_connection:
        try:
            normalized_api_url, account = validate_alpaca_paper_connection(
                api_key, api_secret, api_url
            )
            account_status = getattr(account, "status", "terhubung")
            notify_success(
                f"Koneksi read-only berhasil. Status account: {account_status}. "
                f"Endpoint dinormalisasi menjadi {normalized_api_url}."
            )
        except PermissionError as error:
            notify_error(error, title="Izin Paper Trading Ditolak")
        except (ValueError, ConnectionError) as error:
            notify_error(error, title="Koneksi / Parameter Gagal")
    if submitted:
        if not confirmed:
            notify_warning("Konfirmasi paper trading terlebih dahulu.")
            return
        try:
            normalized_api_url = normalize_alpaca_paper_url(api_url)
            _write_json_config(
                "paper_trading.json",
                {
                    "api_url": normalized_api_url,
                    "state_dim": int(state_dim),
                    "action_dim": int(action_dim),
                    "model_path": config.model_path,
                    "drl_lib": config.drl_lib,
                    "model_name": config.model_name,
                },
            )
            if save_credentials:
                save_secret("alpaca_paper_api_key", api_key)
                save_secret("alpaca_paper_api_secret", api_secret)
            from finrl.trade import trade
            with st.spinner("Paper trading aktif. Hentikan proses untuk menghentikannya."):
                trade(
                    config.test_start, config.test_end, config.tickers, config.data_source,
                    config.interval, config.indicators, config.drl_lib, StockTradingEnv,
                    config.model_name, api_key, api_secret, normalized_api_url, "paper_trading",
                    config.use_vix, cwd=config.model_path, state_dim=int(state_dim),
                    action_dim=int(action_dim),
                )
        except PermissionError as error:
            notify_error(error, title="Izin Paper Trading Ditolak")
        except (ValueError, ConnectionError) as error:
            notify_error(error, title="Koneksi / Parameter Paper Trading Gagal")
        except Exception as error:
            notify_error(error, title="Paper Trading Gagal")


def _monitor_pid() -> int | None:
    path = CONFIG_DIR / "telegram_monitor.pid"
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        command = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="], text=True
        )
        return pid if "finrl.monitoring.telegram_monitor" in command else None
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def show_monitoring(config: ExperimentConfig) -> None:
    from finrl.integrations.secure_store import load_secret
    from finrl.integrations.secure_store import save_secret
    from finrl.integrations.telegram import send_telegram_message
    from finrl.monitoring.telegram_monitor import MonitorConfig
    from finrl.monitoring.telegram_monitor import build_daily_report

    st.subheader("Monitoring & Notifikasi")
    st.caption(
        "Monitor kondisi teknikal IDX secara periodik dan kirim ringkasan riset ke Telegram."
    )
    st.warning(
        "Yahoo Finance bukan feed bursa real-time yang dijamin. Alert bersifat periodik/"
        "near-real-time sesuai ketersediaan data dan bukan rekomendasi transaksi."
    )

    if not st.session_state.get("monitor_profile_initialized"):
        profile = _read_json_config("telegram_monitor.json") or {}
        st.session_state.setdefault("monitor_chat_id", profile.get("chat_id", ""))
        st.session_state.setdefault("monitor_mode", profile.get("schedule_mode", "daily"))
        st.session_state.setdefault("monitor_daily_time", profile.get("daily_time", "17:00"))
        st.session_state.setdefault("monitor_interval", int(profile.get("interval_minutes", 60)))
        st.session_state.setdefault("monitor_lookback", int(profile.get("lookback_days", 420)))
        st.session_state.setdefault("monitor_top_n", int(profile.get("top_n", 5)))
        st.session_state.setdefault("monitor_use_ai", bool(profile.get("use_ai_summary", False)))
        try:
            st.session_state.setdefault(
                "telegram_bot_token", load_secret("telegram_bot_token") or ""
            )
        except Exception:
            st.session_state.setdefault("telegram_bot_token", "")
        st.session_state["monitor_profile_initialized"] = True

    with st.expander("1. Telegram & jadwal", expanded=True):
        bot_token = st.text_input(
            "Telegram bot token", type="password", key="telegram_bot_token"
        )
        chat_id = st.text_input("Telegram chat ID", key="monitor_chat_id")
        schedule_mode = st.selectbox(
            "Mode jadwal", ("daily", "interval"), key="monitor_mode"
        )
        if schedule_mode == "daily":
            daily_time = st.text_input(
                "Waktu kirim harian (HH:MM, waktu mesin)", key="monitor_daily_time"
            )
            interval_minutes = int(st.session_state.get("monitor_interval", 60))
        else:
            interval_minutes = st.number_input(
                "Interval monitoring (menit)", min_value=5, max_value=1440,
                step=5, key="monitor_interval",
            )
            daily_time = st.session_state.get("monitor_daily_time", "17:00")
        lookback_days = st.number_input(
            "Lookback data (hari)", min_value=280, max_value=2000,
            step=20, key="monitor_lookback",
        )
        if int(st.session_state.get("monitor_top_n", 1)) > len(config.tickers):
            st.session_state["monitor_top_n"] = len(config.tickers)
        top_n = st.number_input(
            "Jumlah watchlist dikirim", min_value=1,
            max_value=max(1, len(config.tickers)), key="monitor_top_n",
        )
        use_ai = st.toggle("Tambahkan ringkasan AI Router", key="monitor_use_ai")

    ai_profile = _read_json_config("ai_research.json") or {}
    monitor_payload = {
        "tickers": config.tickers,
        "chat_id": chat_id,
        "schedule_mode": schedule_mode,
        "daily_time": daily_time,
        "interval_minutes": int(interval_minutes),
        "lookback_days": int(lookback_days),
        "top_n": min(int(top_n), len(config.tickers)),
        "risk_free_rate": config.risk_free_rate,
        "use_ai_summary": bool(use_ai),
        "ai_base_url": ai_profile.get("base_url", "http://localhost:20128/v1"),
        "ai_model": ai_profile.get("model", ""),
    }

    save_col, test_col = st.columns(2)
    if save_col.button("Simpan konfigurasi & token", use_container_width=True):
        try:
            pd.to_datetime(daily_time, format="%H:%M")
            save_secret("telegram_bot_token", bot_token)
            path = _write_json_config("telegram_monitor.json", monitor_payload)
            notify_success(f"Konfigurasi disimpan di {path}; token berada di secure vault.")
        except Exception as error:
            notify_error(error, title="Gagal Menyimpan Konfigurasi Monitoring")
    if test_col.button("Kirim pesan tes", use_container_width=True):
        try:
            send_telegram_message(
                bot_token,
                chat_id,
                "NEZU: koneksi notifikasi Telegram berhasil.",
            )
            notify_success("Pesan tes berhasil dikirim.")
        except Exception as error:
            notify_error(error, title="Pesan Tes Gagal")

    st.markdown("### 2. Preview dan eksekusi")
    if st.button("Bangun preview laporan", use_container_width=True):
        try:
            with st.spinner("Mengunduh data dan menghitung kondisi IDX..."):
                preview = build_daily_report(MonitorConfig(**monitor_payload))
            st.session_state["monitor_preview"] = preview
        except Exception as error:
            notify_error(error, title="Preview Gagal Dibuat")
    preview = st.session_state.get("monitor_preview")
    if preview:
        st.code(preview, language="text")
        if st.button("Kirim laporan sekarang", type="primary", use_container_width=True):
            try:
                send_telegram_message(bot_token, chat_id, preview)
                notify_success("Laporan berhasil dikirim ke Telegram.")
            except Exception as error:
                notify_error(error, title="Pengiriman Gagal")

    st.markdown("### 3. Standby monitor")
    pid = _monitor_pid()
    status_col, action_col = st.columns(2)
    status_col.metric("Status", f"Aktif (PID {pid})" if pid else "Tidak aktif")
    if pid:
        if action_col.button("Hentikan monitor", use_container_width=True):
            try:
                os.kill(pid, signal.SIGTERM)
                (CONFIG_DIR / "telegram_monitor.pid").unlink(missing_ok=True)
                notify_success("Monitor dihentikan.")
                st.rerun()
            except OSError as error:
                notify_error(error, title="Monitor Gagal Dihentikan")
    else:
        if action_col.button("Mulai standby monitor", type="primary", use_container_width=True):
            try:
                if not bot_token or not chat_id:
                    raise ValueError("Bot token dan chat ID wajib diisi.")
                pd.to_datetime(daily_time, format="%H:%M")
                save_secret("telegram_bot_token", bot_token)
                config_path = _write_json_config("telegram_monitor.json", monitor_payload)
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                log_path = LOG_DIR / "telegram-monitor.log"
                with log_path.open("a", encoding="utf-8") as log_handle:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "finrl.monitoring.telegram_monitor",
                            "--config",
                            str(config_path),
                        ],
                        cwd=PROJECT_ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                (CONFIG_DIR / "telegram_monitor.pid").write_text(
                    str(process.pid), encoding="utf-8"
                )
                notify_success(f"Standby monitor aktif (PID {process.pid}).")
                st.rerun()
            except Exception as error:
                notify_error(error, title="Monitor Gagal Dimulai")
    st.caption(f"Log monitor: {LOG_DIR / 'telegram-monitor.log'}")


def show_documentation(config: ExperimentConfig) -> None:
    """Render the user guide and collect downloadable experiment outputs."""
    st.subheader("Dokumentasi & pusat output")
    st.caption(
        "Baca panduan operasional, unduh artefak sesi, dan temukan lokasi model."
    )

    output_tab, guide_tab, paper_tab, diagram_tab, glossary_tab = st.tabs(
        (
            "Pusat output", "Panduan lengkap", "Roadmap paper", "Diagram engine",
            "Output apa yang bisa dipakai?",
        )
    )
    with output_tab:
        st.markdown("#### Konfigurasi eksperimen")
        config_json = json.dumps(asdict(config), indent=2, ensure_ascii=False)
        st.download_button(
            "Unduh konfigurasi (JSON)",
            config_json.encode("utf-8"),
            "nezu_experiment_config.json",
            "application/json",
            use_container_width=True,
        )

        model_path = Path(config.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        st.markdown("#### Model hasil training")
        st.code(str(model_path.resolve()), language=None)
        if model_path.exists() or model_path.with_suffix(".zip").exists():
            st.success("Artefak model ditemukan dan siap dipakai untuk backtest.")
        else:
            st.info("Model belum ditemukan. Jalankan tab Train terlebih dahulu.")
        manifest_path = Path(f"{config.model_path}.config.json")
        if manifest_path.exists():
            st.download_button(
                "Unduh manifest model",
                manifest_path.read_bytes(),
                manifest_path.name,
                "application/json",
                use_container_width=True,
            )

        st.markdown("#### Artefak sesi")
        data = st.session_state.get("market_data")
        equity = st.session_state.get("equity_curve")
        col1, col2 = st.columns(2)
        with col1:
            if data is not None:
                st.download_button(
                    "Unduh data pasar",
                    data.to_csv(index=False).encode("utf-8"),
                    "nezu_market_data.csv",
                    "text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Data pasar tersedia setelah tab Data dijalankan.")
        with col2:
            if equity:
                result = pd.DataFrame({"portfolio_value": equity})
                result["return_pct"] = (
                    result["portfolio_value"] / result["portfolio_value"].iloc[0] - 1
                ) * 100
                st.download_button(
                    "Unduh hasil backtest",
                    result.to_csv(index=False).encode("utf-8"),
                    "nezu_backtest.csv",
                    "text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Hasil tersedia setelah tab Backtest dijalankan.")

    with guide_tab:
        guide_path = PROJECT_ROOT / "docs" / "PANDUAN_FINRL_WORKBENCH_ID.md"
        if guide_path.exists():
            guide = guide_path.read_text(encoding="utf-8")
            st.download_button(
                "Unduh dokumentasi (Markdown)",
                guide.encode("utf-8"),
                guide_path.name,
                "text/markdown",
            )
            st.markdown(guide)
        else:
            st.warning(f"Dokumentasi tidak ditemukan di {guide_path}.")

    with paper_tab:
        paper_path = PROJECT_ROOT / "docs" / "ROADMAP_PAPER_DAN_TEORI.md"
        if paper_path.exists():
            paper_guide = paper_path.read_text(encoding="utf-8")
            st.download_button(
                "Unduh roadmap paper (Markdown)",
                paper_guide.encode("utf-8"),
                paper_path.name,
                "text/markdown",
            )
            st.markdown(paper_guide)
        else:
            st.warning(f"Roadmap paper tidak ditemukan di {paper_path}.")

    with diagram_tab:
        st.markdown("#### Arsitektur platform")
        st.graphviz_chart(
            """
            digraph architecture {
              rankdir=LR;
              node [shape=box, style="rounded"];
              user [label="Pengguna"];
              ui [label="Streamlit UI\nSession state"];
              config [label="ExperimentConfig"];
              data [label="DataProcessor\nYahoo / Alpaca / WRDS"];
              analytics [label="IDX Analytics"];
              env [label="StockTradingEnv"];
              agents [label="SB3 / ElegantRL / RLlib"];
              outputs [label="CSV / JSON / Model\nEquity curve"];
              broker [label="Alpaca Paper API"];
              user -> ui -> config;
              config -> data;
              data -> analytics -> outputs;
              data -> env -> agents -> outputs;
              agents -> broker [label="paper trade"];
              outputs -> user;
            }
            """,
            use_container_width=True,
        )

        st.markdown("#### Sesi dan kredensial")
        st.info(
            "Saat ini tidak ada login pengguna, database akun, role, atau session "
            "server permanen. API key Alpaca dimasukkan pada sesi Streamlit."
        )
        st.graphviz_chart(
            """
            digraph credentials {
              rankdir=LR;
              node [shape=box, style="rounded"];
              browser [label="Form password\ndi browser"];
              session [label="Streamlit session_state\nmemori sesi"];
              processor [label="AlpacaProcessor"];
              paper [label="Paper Trading Engine"];
              api [label="Alpaca Paper API"];
              end [label="Refresh / stop session\ninput perlu diisi ulang"];
              browser -> session [label="data source"];
              session -> processor;
              browser -> paper [label="form paper trade"];
              processor -> api [label="HTTPS + API credentials"];
              paper -> api [label="HTTPS + order paper"];
              session -> end [style=dashed];
            }
            """,
            use_container_width=True,
        )

        st.markdown("#### Siklus eksperimen E2E")
        st.graphviz_chart(
            """
            digraph e2e {
              rankdir=LR;
              node [shape=box, style="rounded"];
              cfg [label="1. Konfigurasi"];
              load [label="2. Download + clean"];
              feature [label="3. Indicator + risk"];
              inspect [label="4. Analisa IDX"];
              train [label="5. Train"];
              save [label="6. Simpan model"];
              test [label="7. Backtest OOS"];
              decide [label="8. Review output"];
              paper [label="9. Paper test"];
              cfg -> load -> feature;
              feature -> inspect -> decide;
              feature -> train -> save -> test -> decide -> paper;
              decide -> cfg [label="iterasi terkontrol", style=dashed];
            }
            """,
            use_container_width=True,
        )

    with glossary_tab:
        st.markdown(
            """
| Output | Format/lokasi | Kegunaan |
|---|---|---|
| Data pasar | CSV | Audit data, riset Excel/Python, dan validasi indikator |
| Screener IDX | CSV dari tab Analisa IDX | Shortlist saham, ranking, risk, momentum, dan likuiditas |
| Konfigurasi | JSON | Mengulang eksperimen dan mencatat parameter model |
| Model terlatih | Path pada konfigurasi | Backtest, perbandingan model, dan paper trading |
| Hasil backtest | CSV | Analisis equity curve, return, laporan, dan perbandingan eksperimen |
| Grafik dashboard | Tampilan interaktif | Inspeksi cepat performa, risiko, korelasi, dan breadth |

**Urutan pemakaian yang dianjurkan:** simpan konfigurasi JSON bersama model,
ekspor data dan screener untuk audit, lalu gunakan CSV backtest untuk membandingkan
beberapa eksperimen. Model hanya dipakai pada konfigurasi ticker, indikator, dan
dimensi state/action yang sama dengan saat training.
            """
        )


AUTH_USERS_PATH = CONFIG_DIR / "users.json"
AUTH_BRAND_NAME = "IDN Maker FINRLAB"
AUTH_BRAND_TAGLINE = "Decision intelligence workspace"
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 30

AUTH_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none;}
.idn-auth-hero {
  text-align:center; padding:1.7rem 1.5rem 1.3rem; border:1px solid var(--nezu-border);
  border-radius:18px; background:linear-gradient(150deg,#ffffff 0%, #edf5ef 100%); margin-bottom:1.1rem;
}
.idn-auth-mark {
  width:54px; height:54px; margin:0 auto .75rem; border-radius:15px; display:grid; place-items:center;
  background:var(--nezu-moss); color:#fff; font-weight:700; font-size:1.02rem; letter-spacing:-.03em;
  box-shadow:0 12px 26px rgba(53,106,82,.22);
}
.idn-auth-name {font-size:1.32rem; font-weight:700; letter-spacing:.02em; color:var(--nezu-ink);}
.idn-auth-tag {font-size:.76rem; color:var(--nezu-ink-muted); margin-top:.32rem; letter-spacing:.02em;}
.idn-auth-badge {
  display:inline-block; margin-top:.6rem; padding:.16rem .52rem; border-radius:999px;
  background:var(--nezu-moss-soft); color:var(--nezu-moss-dark); font-size:.66rem;
  font-weight:700; letter-spacing:.08em;
}
.idn-auth-note {font-size:.72rem; color:var(--nezu-ink-muted); text-align:center; margin-top:.9rem; line-height:1.6;}
</style>
"""


def _auth_hero(subtitle: str, badge: str = "SECURE ACCESS") -> str:
    return (
        '<div class="idn-auth-hero">'
        '<div class="idn-auth-mark">IF</div>'
        f'<div class="idn-auth-name">{escape(AUTH_BRAND_NAME)}</div>'
        f'<div class="idn-auth-tag">{escape(subtitle)}</div>'
        f'<div class="idn-auth-badge">{escape(badge)}</div>'
        "</div>"
    )


def _load_auth_store() -> dict[str, Any] | None:
    try:
        from finrl.db.bridge import load_auth_store

        return load_auth_store(CONFIG_DIR)
    except auth.AuthError as error:
        st.error(f"Penyimpanan user bermasalah: {error}")
        return None
    except Exception:
        try:
            return auth.load_users(AUTH_USERS_PATH)
        except auth.AuthError as error:
            st.error(f"Penyimpanan user bermasalah: {error}")
            return None


def _logout() -> None:
    """Drop the whole session so credentials and cached data do not persist."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _render_setup_page() -> None:
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            _auth_hero("Inisialisasi administrator", badge="FIRST-RUN SETUP"),
            unsafe_allow_html=True,
        )
        st.info("Belum ada akun. Buat akun admin pertama untuk mengamankan workspace ini.")
        with st.form("idn_maker_setup"):
            username = st.text_input("Username admin", value="admin")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Ulangi password", type="password")
            submitted = st.form_submit_button(
                "Buat akun", type="primary", use_container_width=True
            )
        if submitted:
            if password != confirm:
                notify_error("Konfirmasi password tidak sama.", title="Validasi Gagal")
                return
            try:
                from finrl.db.bridge import create_user_bridge

                create_user_bridge(CONFIG_DIR, username, password, role="admin")
            except auth.AuthError as error:
                notify_error(str(error), title="Gagal Membuat Akun")
                return
            except Exception as error:
                notify_error(f"Gagal membuat akun: {error}", title="Gagal Membuat Akun")
                return
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = username.strip()
            st.session_state["login_attempts"] = 0
            st.rerun()


def _render_login_page() -> None:
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            _auth_hero(f"{AUTH_BRAND_TAGLINE} · akses terbatas"),
            unsafe_allow_html=True,
        )
        remaining = st.session_state.get("login_locked_until", 0.0) - time.time()
        locked = remaining > 0
        if locked:
            notify_warning(
                f"Terlalu banyak percobaan. Coba lagi dalam {int(remaining) + 1} detik.",
                title="Akun Terkunci Sementara",
            )
        with st.form("idn_maker_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Masuk", type="primary", use_container_width=True, disabled=locked
            )
        if submitted and not locked:
            try:
                from finrl.db.bridge import authenticate_bridge

                ok = authenticate_bridge(CONFIG_DIR, username, password)
            except Exception:
                store = _load_auth_store()
                ok = store is not None and auth.authenticate(store, username, password)
            if ok:
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = username.strip()
                st.session_state["login_attempts"] = 0
                st.session_state["login_locked_until"] = 0.0
                st.rerun()
            else:
                attempts = st.session_state.get("login_attempts", 0) + 1
                if attempts >= MAX_LOGIN_ATTEMPTS:
                    st.session_state["login_locked_until"] = (
                        time.time() + LOGIN_LOCKOUT_SECONDS
                    )
                    st.session_state["login_attempts"] = 0
                else:
                    st.session_state["login_attempts"] = attempts
                notify_error("Username atau password salah.", title="Login Gagal")
        st.markdown(
            '<div class="idn-auth-note">Sesi berakhir saat tab ditutup atau server di-restart. '
            "Kredensial disimpan sebagai hash PBKDF2 di PostgreSQL (fallback configs/users.json).</div>",
            unsafe_allow_html=True,
        )


def _require_authentication() -> bool:
    """Render the auth gate and return True only for an authenticated session."""
    if st.session_state.get("authenticated") and st.session_state.get("auth_user"):
        return True
    try:
        from finrl.db.bridge import user_count_bridge

        count = user_count_bridge(CONFIG_DIR)
    except Exception:
        store = _load_auth_store()
        if store is None:
            return False
        count = auth.user_count(store)
        if count == 0:
            _render_setup_page()
        else:
            _render_login_page()
        return False
    if count == 0:
        _render_setup_page()
    else:
        _render_login_page()
    return False


def main() -> None:
    st.set_page_config(
        page_title="IDN Maker FINRLAB · NEZU", page_icon="◈", layout="wide"
    )
    st.markdown(NEZU_CSS, unsafe_allow_html=True)
    if not _require_authentication():
        return
    with st.sidebar:
        st.markdown(
            """
            <div class="nezu-brand">
              <div class="nezu-mark">N</div>
              <div><div class="nezu-brand-name">NEZU</div><div class="nezu-brand-sub">Decision intelligence</div></div>
            </div>
            <div class="nezu-side-label">NAVIGATION</div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="nezu_navigation"):
            page = _render_sidebar_navigation()
        st.markdown(
            """
            <div class="nezu-side-health">
              <div><span class="nezu-dot"></span><strong>Research engine</strong> · ready</div>
              <div><span class="nezu-dot neutral"></span><strong>Mode</strong> · decision support</div>
            </div>
            <div class="nezu-side-label">ACTIVE EXPERIMENT</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nezu-side-label">SESSION</div>', unsafe_allow_html=True)
        st.caption(f"Masuk sebagai {st.session_state.get('auth_user', '-')}")
        st.button(
            "Keluar",
            key="logout_button",
            icon=":material/logout:",
            use_container_width=True,
            on_click=_logout,
        )
    try:
        config = build_config()
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()

    with st.sidebar.expander("Konfigurasi aktif"):
        st.json(asdict(config))
    source_kwargs: dict[str, str] = st.session_state.get("source_kwargs", {})

    _render_page_header(page)
    if page == "Beranda":
        show_home(config)
    elif page == "Data Market":
        show_data(config, source_kwargs)
    elif page == "Data Sources & Broker":
        show_provider_hub(config)
    elif page == "Analisa IDX":
        show_idx_analysis(config)
    elif page == "Training":
        show_training(config, source_kwargs)
    elif page == "Model & Evaluasi":
        show_backtest(config, source_kwargs)
    elif page == "AI Research Copilot":
        show_ai_research(config)
    elif page == "Monitoring & Notifikasi":
        show_monitoring(config)
    elif page == "Paper Trading":
        show_paper_trading(config)
    else:
        show_documentation(config)


if __name__ == "__main__":
    main()
