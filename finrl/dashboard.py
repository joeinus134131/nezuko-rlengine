"""Interactive E2E workbench for configuring and evaluating FinRL experiments.

Run with: ``streamlit run finrl/dashboard.py``
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
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
    path = CONFIG_DIR / name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_config(name: str, payload: dict[str, Any]) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _apply_config_payload(payload: dict[str, Any]) -> None:
    """Apply a downloaded experiment config before widgets are instantiated."""
    required = {field.name for field in ExperimentConfig.__dataclass_fields__.values()}
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
    with st.sidebar:
        st.header("Konfigurasi eksperimen")
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
                st.error(f"Konfigurasi gagal dimuat: {error}")
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
            from finrl.integrations.provider_adapter import load_provider_store

            provider_path = CONFIG_DIR / "providers.json"
            try:
                provider_store = load_provider_store(provider_path)
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

    return ExperimentConfig(
        tickers=_tickers(ticker_text), data_source=source, interval=interval,
        train_start=train_start, train_end=train_end, test_start=test_start,
        test_end=test_end, indicators=indicators, use_vix=use_vix, drl_lib=library,
        model_name=model, model_path=model_path, timesteps=int(timesteps),
        agent_params=_parse_json(params, "Parameter agent"), universe=universe,
        risk_free_rate=float(risk_free_rate) / 100,
    )


def run_training(config: ExperimentConfig, source_kwargs: dict[str, str]) -> Any:
    from finrl.train import train

    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    shared = dict(source_kwargs, cwd=config.model_path)
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
        config.model_name, config.use_vix, cwd=config.model_path, **source_kwargs,
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
    st.subheader("Investment AI Research Workspace")
    st.caption(
        "Alur riset teknikal dan historis yang dapat direproduksi—bukan rekomendasi investasi otomatis."
    )
    artifact = _resolve_model_path(config.model_path)
    data = st.session_state.get("market_data")
    equity = st.session_state.get("equity_curve")
    a, b, c, d = st.columns(4)
    a.metric("Universe", config.universe)
    b.metric("Ticker", len(config.tickers))
    c.metric("Data", "Siap" if data is not None else "Belum dimuat")
    d.metric("Model", "Ditemukan" if artifact else "Belum tersedia")

    st.markdown("### Workflow eksperimen")
    st.graphviz_chart(
        """
        digraph workflow {
          rankdir=LR;
          node [shape=box, style="rounded"];
          config [label="1. Config"];
          data [label="2. Data & QA"];
          analysis [label="3. Analisa"];
          train [label="4. Train"];
          model [label="5. Load model"];
          test [label="6. OOS evaluation"];
          paper [label="7. Paper test"];
          config -> data -> analysis -> train -> model -> test -> paper;
        }
        """,
        width="stretch",
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
                "Buka Data Market",
                "Buka Training atau Model & Evaluasi",
                "Jalankan backtest out-of-sample",
            ],
        }
    )
    st.dataframe(status, hide_index=True, width="stretch")


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
            st.success(f"{len(data):,} baris siap digunakan untuk eksperimen.")
        except Exception as error:
            st.exception(error)
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
                st.dataframe(coverage, width="stretch")

        table_tab, history_tab = st.tabs(("Preview data", "Histori per ticker"))
        with table_tab:
            st.dataframe(data.tail(100), width="stretch")
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
                st.dataframe(ticker_data.tail(250), width="stretch")
        st.download_button(
            "Unduh seluruh data pasar (CSV)",
            data.to_csv(index=False).encode("utf-8"),
            "finrl_market_data.csv",
            "text/csv",
        )
        if {date_col, "tic", "close"}.issubset(data.columns):
            chart = data.pivot_table(index=date_col, columns="tic", values="close", aggfunc="last")
            st.line_chart(chart)


def show_provider_hub(config: ExperimentConfig) -> None:
    """Configure and compare research, licensed, and broker quote sources."""
    from finrl.integrations.provider_adapter import GenericRESTProvider
    from finrl.integrations.provider_adapter import ProviderProfile
    from finrl.integrations.provider_adapter import load_provider_store
    from finrl.integrations.provider_adapter import save_provider_store
    from finrl.integrations.provider_adapter import yahoo_research_quote
    from finrl.integrations.secure_store import save_secret

    path = CONFIG_DIR / "providers.json"
    try:
        store = load_provider_store(path)
        profiles = [
            ProviderProfile.from_dict(item) for item in store.get("providers", [])
        ]
    except Exception as error:
        st.error(f"Konfigurasi provider tidak dapat dibaca: {error}")
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
            save_provider_store(path, store)
            st.success("Source priority tersimpan.")

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
                ]], hide_index=True, width="stretch",
            )
        for error in st.session_state.get("provider_quote_errors", []):
            st.error(error)

        if active_broker and st.button("Uji account broker (read-only)"):
            try:
                account = GenericRESTProvider(by_id[active_broker]).get_account()
                st.success("Autentikasi account broker berhasil.")
                st.json(account)
            except Exception as error:
                st.error(str(error))
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
                save_provider_store(path, store)
                if secret_value:
                    if not profile.secret_name:
                        raise ValueError("Credential vault key wajib diisi untuk menyimpan token.")
                    save_secret(profile.secret_name, secret_value)
                st.success(f"Profile {profile.name} tersimpan tanpa menulis secret ke JSON.")
                st.rerun()
            except Exception as error:
                st.error(str(error))

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
              rl [label="FinRL history/training"];
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
            width="stretch",
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
        st.exception(error)
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
        st.altair_chart((heatmap + labels).properties(height=520), width="stretch")
        st.caption("Merah = bergerak searah, biru = berlawanan, putih = hubungan linear lemah.")
        with st.expander("Lihat matriks angka"):
            st.dataframe(matrix, width="stretch")
    with breadth_tab:
        st.line_chart(analysis.breadth)


def show_training(config: ExperimentConfig, source_kwargs: dict[str, str]) -> None:
    st.subheader("Training")
    st.caption("Model disimpan ke path yang ditentukan pada konfigurasi.")
    if st.button("Mulai training", type="primary"):
        try:
            with st.spinner("Training sedang berjalan..."):
                st.session_state["trained_model"] = run_training(config, source_kwargs)
            st.success(f"Training selesai. Artefak tersedia di {config.model_path}.")
            manifest_path = st.session_state.get("model_manifest_path")
            if manifest_path:
                st.caption(f"Konfigurasi model disimpan di {manifest_path}")
        except Exception as error:
            st.exception(error)


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
            st.success("Model berhasil dimuat dan dipilih sebagai model aktif.")
        except Exception as error:
            st.exception(error)

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
            st.success("Evaluasi out-of-sample selesai.")
        except Exception as error:
            st.exception(error)

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
            st.dataframe(metrics.round(3), width="stretch")
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
            st.dataframe(checks[["Pemeriksaan", "Hasil"]], hide_index=True, width="stretch")
            if checks["Status"].all():
                st.success("Seluruh pemeriksaan dasar lulus. Tetap lakukan multi-period dan paper test.")
            else:
                st.warning("Hasil belum memenuhi seluruh pemeriksaan dasar kelayakan.")

        frame = pd.DataFrame({"portfolio_value": values})
        frame["return_pct"] = (frame["portfolio_value"] / frame["portfolio_value"].iloc[0] - 1) * 100
        st.download_button(
            "Unduh hasil backtest (CSV)",
            frame.to_csv(index=False).encode("utf-8"),
            "finrl_backtest.csv",
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
                    st.success(f"Router terhubung. {len(models)} model tersedia.")
                else:
                    st.warning("Router terhubung, tetapi daftar model kosong.")
            except Exception as error:
                st.exception(error)

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
        st.success(f"Profil disimpan di {path} (tanpa API key).")
    if save_key_col.button("Simpan API key ke secure vault", use_container_width=True):
        try:
            save_secret("ai_router_api_key", api_key)
            st.success("API key AI router tersimpan di credential vault OS.")
        except Exception as error:
            st.error(f"API key gagal disimpan: {error}")
    ticker = st.selectbox("Ticker fokus", config.tickers, key="ai_ticker")
    if st.button("Muat fundamental & berita Yahoo Finance", use_container_width=True):
        try:
            with st.spinner("Mengambil snapshot perusahaan dan headline..."):
                st.session_state["ai_company_context"] = load_company_research_context(ticker)
            context_loaded = st.session_state["ai_company_context"]
            st.success(
                f"Snapshot dimuat: {len(context_loaded['fundamentals'])} field fundamental, "
                f"{len(context_loaded['news'])} headline."
            )
        except Exception as error:
            st.exception(error)
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
                st.warning(
                    "Model utama gagal; fallback berhasil. Percobaan sebelumnya: "
                    + " | ".join(failures)
                )
        except RouterHTTPError as error:
            if error.status_code == 403:
                st.error(
                    "Provider menolak akses model (HTTP 403). Periksa koneksi akun/provider, "
                    "izin model, dan kuota di dashboard 9Router. Pilih model lain atau "
                    "konfigurasikan fallback combo."
                )
            else:
                st.error(
                    f"AI router gagal (HTTP {error.status_code}). Coba model/provider lain "
                    "atau periksa status dan kuota router."
                )
            with st.expander("Detail teknis router"):
                st.code(error.detail[:2000], language="text")
        except Exception as error:
            st.error(f"AI router tidak dapat menyelesaikan request: {error}")

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
            st.success(f"Konfigurasi paper trading disimpan di {path}.")
        except Exception as error:
            st.error(f"Konfigurasi gagal disimpan: {error}")
    if test_connection:
        try:
            normalized_api_url, account = validate_alpaca_paper_connection(
                api_key, api_secret, api_url
            )
            account_status = getattr(account, "status", "terhubung")
            st.success(
                f"Koneksi read-only berhasil. Status account: {account_status}. "
                f"Endpoint dinormalisasi menjadi {normalized_api_url}."
            )
        except PermissionError as error:
            st.error(str(error))
        except (ValueError, ConnectionError) as error:
            st.error(str(error))
    if submitted:
        if not confirmed:
            st.error("Konfirmasi paper trading terlebih dahulu.")
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
            st.error(str(error))
        except (ValueError, ConnectionError) as error:
            st.error(str(error))
        except Exception as error:
            st.exception(error)


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
            st.success(f"Konfigurasi disimpan di {path}; token berada di secure vault.")
        except Exception as error:
            st.error(f"Konfigurasi monitoring gagal disimpan: {error}")
    if test_col.button("Kirim pesan tes", use_container_width=True):
        try:
            send_telegram_message(
                bot_token,
                chat_id,
                "FinRL Workbench: koneksi notifikasi Telegram berhasil.",
            )
            st.success("Pesan tes berhasil dikirim.")
        except Exception as error:
            st.error(f"Pesan tes gagal: {error}")

    st.markdown("### 2. Preview dan eksekusi")
    if st.button("Bangun preview laporan", use_container_width=True):
        try:
            with st.spinner("Mengunduh data dan menghitung kondisi IDX..."):
                preview = build_daily_report(MonitorConfig(**monitor_payload))
            st.session_state["monitor_preview"] = preview
        except Exception as error:
            st.error(f"Preview gagal dibuat: {error}")
    preview = st.session_state.get("monitor_preview")
    if preview:
        st.code(preview, language="text")
        if st.button("Kirim laporan sekarang", type="primary", use_container_width=True):
            try:
                send_telegram_message(bot_token, chat_id, preview)
                st.success("Laporan berhasil dikirim ke Telegram.")
            except Exception as error:
                st.error(f"Pengiriman gagal: {error}")

    st.markdown("### 3. Standby monitor")
    pid = _monitor_pid()
    status_col, action_col = st.columns(2)
    status_col.metric("Status", f"Aktif (PID {pid})" if pid else "Tidak aktif")
    if pid:
        if action_col.button("Hentikan monitor", use_container_width=True):
            try:
                os.kill(pid, signal.SIGTERM)
                (CONFIG_DIR / "telegram_monitor.pid").unlink(missing_ok=True)
                st.success("Monitor dihentikan.")
                st.rerun()
            except OSError as error:
                st.error(f"Monitor gagal dihentikan: {error}")
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
                st.success(f"Standby monitor aktif (PID {process.pid}).")
                st.rerun()
            except Exception as error:
                st.error(f"Monitor gagal dimulai: {error}")
    st.caption(f"Log monitor: {LOG_DIR / 'telegram-monitor.log'}")


def show_documentation(config: ExperimentConfig) -> None:
    """Render the user guide and collect downloadable experiment outputs."""
    st.subheader("Dokumentasi & pusat output")
    st.caption(
        "Baca panduan operasional, unduh artefak sesi, dan temukan lokasi model."
    )

    output_tab, guide_tab, diagram_tab, glossary_tab = st.tabs(
        (
            "Pusat output", "Panduan lengkap", "Diagram engine",
            "Output apa yang bisa dipakai?",
        )
    )
    with output_tab:
        st.markdown("#### Konfigurasi eksperimen")
        config_json = json.dumps(asdict(config), indent=2, ensure_ascii=False)
        st.download_button(
            "Unduh konfigurasi (JSON)",
            config_json.encode("utf-8"),
            "finrl_experiment_config.json",
            "application/json",
            width="stretch",
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
                    "finrl_market_data.csv",
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
                    "finrl_backtest.csv",
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
            width="stretch",
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
            width="stretch",
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
            width="stretch",
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


def main() -> None:
    st.set_page_config(page_title="FinRL Workbench", page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
        [data-testid="stMetric"] {padding: .75rem; border: 1px solid rgba(128,128,128,.18); border-radius: .6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("FinRL Workbench")
        st.caption("Research • Train • Validate")
        page = st.radio("Navigasi", NAVIGATION, key="navigation")
        st.divider()
    try:
        config = build_config()
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()

    with st.sidebar.expander("Konfigurasi aktif"):
        st.json(asdict(config))
    source_kwargs: dict[str, str] = st.session_state.get("source_kwargs", {})

    st.title(page)
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
