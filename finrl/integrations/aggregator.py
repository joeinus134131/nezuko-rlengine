"""Evidence-grounded multi-agent AI aggregation over 9Router (OpenAI-compatible).

Why this design (anti-"halu"):

1. The AI never sees raw prompts from the user alone; it receives an
   **evidence pack** produced by the quantitative engine (scores, gates,
   regime reasons, fundamentals, guard metrics, signal-study statistics, and
   user-supplied news with date + URL).
2. Roles are separated: a **bull** analyst, a **bear** analyst, and a
   **judge** that must weigh both and output strict JSON.
3. Every number in the judge's text is checked against the evidence pack
   (:func:`grounding_report`). Below the threshold (default 90%) the judge is
   asked once to correct itself; if it still fails, its score adjustment is
   discarded.
4. The AI may only *nudge* the quantitative score by at most ±10 points and
   only when the panel is grounded and agrees. Quant is the anchor.
5. Every call is written to an audit log (prompt hash, model, temperature,
   raw output) so any conclusion can be reproduced and challenged.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable

from finrl.integrations.ai_router import RouterHTTPError
from finrl.integrations.ai_router import chat_completion

ChatFn = Callable[[str, list[dict[str, str]], float], tuple[str, str]]

STANCES = ("POSITIF", "NETRAL", "NEGATIF")
REQUIRED_KEYS = (
    "sikap",
    "penyesuaian_skor",
    "confidence",
    "fakta",
    "inferensi",
    "bullish_case",
    "bearish_case",
    "risiko",
    "invalidation",
    "ketidakpastian",
    "verifikasi_lanjutan",
)
TEXT_KEYS = ("fakta", "inferensi", "bullish_case", "bearish_case", "risiko", "invalidation", "ketidakpastian")
RETRYABLE_STATUS = {401, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504}
MAX_ADJUSTMENT = 10.0
GROUNDING_THRESHOLD = 0.90

SYSTEM_RULES = (
    "Anda analis riset ekuitas Indonesia (IDX) yang disiplin dan skeptis. Aturan mutlak:\n"
    "1. Gunakan HANYA angka yang ada di EVIDENCE PACK. Dilarang mengarang angka, tanggal, target harga, "
    "laporan keuangan, berita, atau kutipan.\n"
    "2. Jika data tidak ada, tulis 'data tidak tersedia' dan masukkan ke ketidakpastian.\n"
    "3. Bedakan FAKTA (dari evidence) dan INFERENSI (penalaran Anda).\n"
    "4. Ini riset, bukan perintah beli/jual. Jangan menjanjikan keuntungan atau akurasi.\n"
    "5. Bahasa Indonesia, ringkas, spesifik, setiap klaim menyebut angka evidence-nya."
)


# --------------------------------------------------------------------------- evidence
def _clean(value: Any) -> Any:
    """JSON-safe, rounded values (NaN -> None) so the evidence stays compact."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 4)
    if hasattr(value, "isoformat"):
        return str(value)[:10]
    return value if isinstance(value, (str, int)) else str(value)


EVIDENCE_FIELDS = (
    "price", "score", "label", "confidence", "technical_score", "risk_score", "fundamental_score",
    "regime_score", "rsi14", "macd_hist", "sma20", "sma50", "sma200", "atr_pct", "momentum_12_1",
    "dist_52w_high", "return_1m", "return_3m", "return_6m", "return_12m", "annual_volatility", "sharpe",
    "sortino", "max_drawdown", "daily_cvar_95", "avg_value_idr_20d", "sector", "timing", "gates",
    "regime_tilt_reasons", "hist_prob_beat_median", "hist_prob_n", "evidence_strength", "fund_quality",
    "fund_completeness", "stale_days", "last_date",
)
FUNDAMENTAL_FIELDS = (
    "pe", "pb", "roe", "net_margin", "der", "eps_growth", "revenue_growth", "dividend_yield",
    "current_ratio", "fcf_yield", "value_score", "quality_score", "growth_score", "income_score",
    "data_quality", "source", "retrieved_at",
)


def build_evidence_pack(
    ticker: str | dict[str, Any] | Any,
    row: dict[str, Any] | Any = None,
    plan: dict[str, Any] | None = None,
    regime: Any = None,
    fundamentals: dict[str, Any] | None = None,
    signal_summary: dict[str, Any] | None = None,
    guard_summary: dict[str, Any] | None = None,
    news: list[dict[str, Any]] | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Assemble the only facts the AI is allowed to use."""
    if row is None and not isinstance(ticker, str):
        row = ticker
        if hasattr(row, "get"):
            ticker = str(row.get("ticker") or getattr(row, "name", "UNKNOWN"))
        else:
            ticker = str(getattr(row, "name", "UNKNOWN"))
    source = dict(row) if not isinstance(row, dict) else row
    quant = {key: source.get(key) for key in EVIDENCE_FIELDS if key in source}
    pack: dict[str, Any] = {
        **quant,
        "ticker": ticker,
        "as_of": as_of or str(source.get("last_date") or "")[:10],
        "kuantitatif": quant,
        "rencana_posisi": plan or {},
        "fundamental": {k: (fundamentals or {}).get(k) for k in FUNDAMENTAL_FIELDS if k in (fundamentals or {})},
        "rezim": {},
        "studi_sinyal_historis": signal_summary or {},
        "uji_model_rl": guard_summary or {},
        "berita_pengguna": [],
    }
    if regime is not None:
        if isinstance(regime, dict):
            regime_label = regime.get("label") or regime.get("regime")
            regime_score = regime.get("score")
            regime_reasons = regime.get("reasons", [])
            regime_caps = regime.get("caps", [])
        else:
            # Macro RegimeAssessment exposes `.regime`; older integrations used
            # `.label`. Accept both so the evidence boundary stays stable.
            regime_label = getattr(regime, "label", None) or getattr(regime, "regime", None)
            regime_score = getattr(regime, "score", None)
            regime_reasons = getattr(regime, "reasons", [])
            regime_caps = getattr(regime, "caps", [])
        pack["rezim"] = {
            "label": regime_label,
            "skor": regime_score,
            "alasan": regime_reasons,
            "batasan": regime_caps,
        }
    for item in news or []:
        text = str(item.get("text") or item.get("ringkasan") or "").strip()
        if not text:
            continue
        pack["berita_pengguna"].append(
            {
                "tanggal": str(item.get("date") or item.get("tanggal") or "tanpa tanggal"),
                "sumber": str(item.get("url") or item.get("sumber") or "tanpa sumber"),
                "ringkasan": text[:1200],
            }
        )
    return _clean(pack)


# --------------------------------------------------------------------------- prompts
JSON_SCHEMA_HINT = {
    "sikap": "POSITIF | NETRAL | NEGATIF",
    "penyesuaian_skor": "angka -10..10 terhadap skor kuantitatif (0 jika ragu)",
    "confidence": "0..100, keyakinan pada sikap (bukan probabilitas profit)",
    "fakta": ["klaim + angka persis dari evidence"],
    "inferensi": ["penalaran yang ditarik dari fakta"],
    "bullish_case": ["alasan positif terkuat beserta angka"],
    "bearish_case": ["alasan negatif terkuat beserta angka"],
    "risiko": ["risiko utama"],
    "invalidation": ["kondisi terukur yang membatalkan tesis"],
    "ketidakpastian": ["data yang tidak tersedia / asumsi"],
    "verifikasi_lanjutan": ["apa yang harus dicek manual sebelum eksekusi"],
}


def _evidence_text(pack: dict[str, Any]) -> str:
    return json.dumps(pack, ensure_ascii=False, indent=1)


def bull_messages(pack: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {
            "role": "user",
            "content": (
                "PERAN: Analis BULLISH. Susun maksimal 6 argumen terkuat MENDUKUNG saham ini, masing-masing "
                "mengutip angka evidence (teknikal, risiko, fundamental, makro/rezim, bukti historis). "
                "Akui juga 1 kelemahan terbesar argumen Anda.\n\nEVIDENCE PACK:\n" + _evidence_text(pack)
            ),
        },
    ]


def bear_messages(pack: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {
            "role": "user",
            "content": (
                "PERAN: Analis BEARISH / risk manager. Susun maksimal 6 argumen terkuat MENOLAK atau menunda "
                "saham ini, masing-masing mengutip angka evidence. Fokus: gates, drawdown, likuiditas, "
                "rezim, valuasi, data yang hilang, bukti historis lemah.\n\nEVIDENCE PACK:\n" + _evidence_text(pack)
            ),
        },
    ]


def judge_messages(pack: dict[str, Any], bull: str = "", bear: str = "") -> list[dict[str, str]]:
    debate = ""
    if bull or bear:
        debate = f"\n\nARGUMEN BULL:\n{bull.strip()}\n\nARGUMEN BEAR:\n{bear.strip()}"
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {
            "role": "user",
            "content": (
                "PERAN: Investment committee (hakim). Timbang evidence"
                + (" dan kedua argumen" if debate else " dari sisi bull dan bear")
                + ". Skor kuantitatif adalah jangkar; Anda hanya boleh menyesuaikan -10..10 bila ada alasan "
                "kuat yang belum tertangkap model. Abaikan argumen yang tidak didukung angka evidence.\n"
                "Balas HANYA JSON valid (tanpa markdown) dengan kunci persis:\n"
                + json.dumps(JSON_SCHEMA_HINT, ensure_ascii=False, indent=1)
                + "\n\nEVIDENCE PACK:\n"
                + _evidence_text(pack)
                + debate
            ),
        },
    ]


# --------------------------------------------------------------------------- parsing
def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text`` (tolerates code fences/prose)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Respons AI kosong.")
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth, in_string, escape = 0, False, False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
                    break
        start = cleaned.find("{", start + 1)
    raise ValueError("Tidak menemukan objek JSON valid di respons AI.")


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[-+−]?\d[\d.,]*%?(?![A-Za-z_]*\d)")


def _parse_number_token(token: str) -> tuple[list[float], int]:
    """Return candidate values for a numeric token (handles ID/EN separators)."""
    raw = token.replace("−", "-").rstrip("%").rstrip(".,")
    sign = -1.0 if raw.startswith("-") else 1.0
    raw = raw.lstrip("+-")
    if not raw:
        return [], 0
    candidates: list[tuple[float, int]] = []
    if "." in raw and "," in raw:
        decimal = "," if raw.rfind(",") > raw.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        number = raw.replace(thousands, "").replace(decimal, ".")
        candidates.append((float(number), len(raw) - raw.rfind(decimal) - 1))
    elif "," in raw or "." in raw:
        sep = "," if "," in raw else "."
        parts = raw.split(sep)
        grouped = all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3
        if len(parts) > 2:
            if grouped:
                candidates.append((float("".join(parts)), 0))
        else:
            candidates.append((float(parts[0] + "." + parts[1]) if parts[1] else float(parts[0]), len(parts[1])))
            if grouped:
                candidates.append((float("".join(parts)), 0))
    else:
        candidates.append((float(raw), 0))
    values = [sign * value for value, _ in candidates]
    decimals = max((d for _, d in candidates), default=0)
    return values, decimals


def extract_numbers(text: str) -> list[dict[str, Any]]:
    found = []
    for match in NUMBER_PATTERN.finditer(text or ""):
        token = match.group(0)
        try:
            values, decimals = _parse_number_token(token)
        except ValueError:
            continue
        if values:
            found.append({"token": token, "values": values, "decimals": decimals, "percent": token.endswith("%")})
    return found


def _evidence_numbers(pack: Any) -> list[float]:
    numbers: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, bool) or value is None:
            return
        elif isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                numbers.append(float(value))
        elif isinstance(value, str):
            for item in extract_numbers(value):
                numbers.extend(item["values"])

    walk(pack)
    return numbers


SCALES = (1.0, 100.0, 0.01, 1e-3, 1e-6, 1e-9, 1e-12)


def _is_grounded(item: dict[str, Any], evidence: list[float]) -> bool:
    for value in item["values"]:
        if float(value).is_integer() and 0 <= abs(value) <= 12:
            return True  # counts, months, horizons
        rounding = 0.51 * 10 ** (-item["decimals"])
        for reference in evidence:
            for scale in SCALES:
                target = reference * scale
                tolerance = max(0.02 * abs(target), rounding)
                if abs(abs(value) - abs(target)) <= tolerance:
                    return True
    return False


def grounding_report(judgement: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    """Share of numbers in the AI's text that can be traced to the evidence pack."""
    evidence = _evidence_numbers(pack)
    texts: list[str] = []
    for key in TEXT_KEYS + ("verifikasi_lanjutan",):
        value = judgement.get(key)
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend(str(v) for v in value)
    numbers = [item for text in texts for item in extract_numbers(text)]
    ungrounded = [item["token"] for item in numbers if not _is_grounded(item, evidence)]
    total = len(numbers)
    ratio = 1.0 if total == 0 else (total - len(ungrounded)) / total
    return {"total_numbers": total, "ungrounded": ungrounded, "ratio": round(ratio, 4)}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def validate_judgement(raw: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    """Normalise schema, clamp numbers, attach grounding report and issues."""
    issues: list[str] = []
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        issues.append(f"Kunci hilang: {', '.join(missing)}")
    stance = str(raw.get("sikap", "NETRAL")).strip().upper()
    if stance not in STANCES:
        issues.append(f"Sikap tidak valid '{stance}', dipaksa NETRAL")
        stance = "NETRAL"
    try:
        adjustment = float(raw.get("penyesuaian_skor", 0) or 0)
    except (TypeError, ValueError):
        adjustment, _ = 0.0, issues.append("penyesuaian_skor bukan angka")
    if not math.isfinite(adjustment):
        adjustment = 0.0
    if abs(adjustment) > MAX_ADJUSTMENT:
        issues.append(f"penyesuaian_skor {adjustment:+.1f} dipangkas ke ±{MAX_ADJUSTMENT:.0f}")
        adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adjustment))
    try:
        confidence = float(raw.get("confidence", 50) or 0)
    except (TypeError, ValueError):
        confidence = 50.0
    confidence = max(0.0, min(100.0, confidence if math.isfinite(confidence) else 50.0))
    judgement = {
        "sikap": stance,
        "penyesuaian_skor": round(adjustment, 2),
        "confidence": round(confidence, 1),
        **{key: _as_list(raw.get(key)) for key in REQUIRED_KEYS if key not in ("sikap", "penyesuaian_skor", "confidence")},
    }
    if stance == "POSITIF" and adjustment < 0 or stance == "NEGATIF" and adjustment > 0:
        issues.append("Arah penyesuaian bertentangan dengan sikap")
    if not judgement["bearish_case"]:
        issues.append("Tidak ada bearish case (analisa satu sisi)")
    report = grounding_report(judgement, pack)
    judgement["grounding"] = report
    if report["ratio"] < GROUNDING_THRESHOLD:
        issues.append(
            f"Hanya {report['ratio']:.0%} angka tertelusur ke evidence (< {GROUNDING_THRESHOLD:.0%}): "
            + ", ".join(report["ungrounded"][:8])
        )
    judgement["issues"] = issues
    judgement["valid"] = not missing and report["ratio"] >= GROUNDING_THRESHOLD
    return judgement


# --------------------------------------------------------------------------- execution
def make_router_chat(base_url: str, api_key: str, max_tokens: int = 1800, timeout: int = 90) -> ChatFn:
    def _chat(model: str, messages: list[dict[str, str]], temperature: float) -> tuple[str, str]:
        response = chat_completion(base_url, api_key, model, messages, temperature, max_tokens, timeout)
        return response.content, response.model

    return _chat


def _audit(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def call_with_fallback(
    chat: ChatFn,
    models: list[str],
    messages: list[dict[str, str]],
    temperature: float,
    role: str,
    audit_path: Path | None = None,
    ticker: str = "",
) -> tuple[str, str, list[str]]:
    """Try models in order; skip to the next on quota/permission/upstream errors."""
    errors: list[str] = []
    prompt_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()
    for model in [m for m in models if m and m.strip()]:
        started = time.time()
        try:
            content, used = chat(model, messages, temperature)
        except RouterHTTPError as error:
            errors.append(f"{model}: HTTP {error.status_code}")
            if error.status_code not in RETRYABLE_STATUS:
                break
            continue
        except ConnectionError as error:
            errors.append(f"{model}: {error}")
            continue
        _audit(
            audit_path,
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "ticker": ticker,
                "role": role,
                "model": used,
                "temperature": temperature,
                "prompt_sha256": prompt_hash,
                "latency_s": round(time.time() - started, 2),
                "response": content,
            },
        )
        return content, used, errors
    raise ConnectionError("Semua model gagal: " + "; ".join(errors or ["daftar model kosong"]))


@dataclass
class PanelResult:
    ticker: str
    judgements: list[dict[str, Any]] = field(default_factory=list)
    bull: str = ""
    bear: str = ""
    errors: list[str] = field(default_factory=list)
    consensus: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def consensus_of(judgements: list[dict[str, Any]]) -> dict[str, Any]:
    """Majority stance, median adjustment/confidence, agreement over valid judges."""
    valid = [j for j in judgements if j.get("valid")]
    if not valid:
        return {
            "sikap": "NETRAL", "penyesuaian_skor": 0.0, "confidence": 0.0, "agreement": 0.0,
            "valid_judges": 0, "total_judges": len(judgements),
            "catatan": "Tidak ada penilaian AI yang lolos validasi grounding; skor kuantitatif dipakai apa adanya.",
        }
    counts = Counter(j["sikap"] for j in valid)
    stance, votes = counts.most_common(1)[0]
    if list(counts.values()).count(votes) > 1:
        stance = "NETRAL"  # tie -> no edge from the panel
    agreement = votes / len(valid)
    aligned = [j for j in valid if j["sikap"] == stance] or valid
    return {
        "sikap": stance,
        "penyesuaian_skor": round(statistics.median(j["penyesuaian_skor"] for j in aligned), 2),
        "confidence": round(statistics.median(j["confidence"] for j in aligned), 1),
        "agreement": round(agreement, 3),
        "valid_judges": len(valid),
        "total_judges": len(judgements),
        "grounding_min": min(j["grounding"]["ratio"] for j in valid),
        "catatan": "",
    }


def apply_ai_adjustment(quant_score: float, consensus: dict[str, Any], min_agreement: float = 0.6) -> dict[str, Any]:
    """Bounded nudge: AI can move the quant score by at most ±10, only if grounded and agreed."""
    adjustment = float(consensus.get("penyesuaian_skor", 0.0) or 0.0)
    reason = "Disesuaikan oleh panel AI (grounded, sepakat)"
    if consensus.get("valid_judges", 0) == 0:
        adjustment, reason = 0.0, "Panel AI tidak valid: tanpa penyesuaian"
    elif consensus.get("valid_judges", 0) > 1 and consensus.get("agreement", 0) < min_agreement:
        adjustment, reason = 0.0, f"Panel AI tidak sepakat (< {min_agreement:.0%}): tanpa penyesuaian"
    adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adjustment))
    final = max(0.0, min(100.0, float(quant_score) + adjustment))
    return {"quant_score": round(float(quant_score), 1), "ai_adjustment": round(adjustment, 2),
            "final_score": round(final, 1), "reason": reason}


def run_panel(
    pack: dict[str, Any],
    chat: ChatFn,
    judge_models: list[str],
    fallback_models: list[str] | None = None,
    debate: bool = True,
    temperature: float = 0.2,
    audit_path: Path | None = None,
) -> PanelResult:
    """Bull + bear (optional) then one judge per model, validated and aggregated."""
    ticker = str(pack.get("ticker", ""))
    result = PanelResult(ticker=ticker)
    fallback = [m for m in (fallback_models or []) if m]
    primary = [m for m in judge_models if m]
    if not primary:
        raise ValueError("Pilih minimal satu model AI.")
    if debate:
        for role, builder in (("bull", bull_messages), ("bear", bear_messages)):
            try:
                text, _, errors = call_with_fallback(
                    chat, [primary[0], *fallback], builder(pack), temperature, role, audit_path, ticker
                )
                setattr(result, role, text)
                result.errors.extend(errors)
            except ConnectionError as error:
                result.errors.append(f"{role}: {error}")
    for model in primary:
        messages = judge_messages(pack, result.bull, result.bear)
        judgement: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                text, used, errors = call_with_fallback(
                    chat, [model, *fallback], messages, temperature if attempt == 0 else 0.0,
                    "judge", audit_path, ticker,
                )
                result.errors.extend(errors)
            except ConnectionError as error:
                result.errors.append(f"judge {model}: {error}")
                break
            try:
                judgement = validate_judgement(extract_json(text), pack)
            except ValueError as error:
                judgement = {"valid": False, "sikap": "NETRAL", "penyesuaian_skor": 0.0, "confidence": 0.0,
                             "issues": [str(error)], "grounding": {"ratio": 0.0, "ungrounded": [], "total_numbers": 0},
                             "raw": text[:2000]}
            judgement["model"] = used
            judgement["attempt"] = attempt + 1
            if judgement.get("valid"):
                break
            messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Koreksi: " + "; ".join(judgement.get("issues", []))
                        + ". Hapus/ubah setiap angka yang tidak ada di EVIDENCE PACK dan balas ulang HANYA JSON valid."
                    ),
                },
            ]
        if judgement is not None:
            result.judgements.append(judgement)
    result.consensus = consensus_of(result.judgements)
    return result


run_debate = run_panel
