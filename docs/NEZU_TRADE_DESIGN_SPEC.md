# nezu-trade — Product UI Redesign Specification

Figma file: https://www.figma.com/design/V7sKWvVsjXZidGzftLAY1A

## Product character

`nezu-trade` is a calm decision-support workspace, not a casino-style trading
terminal. The interface should feel human, neutral, natural, and credible. It
must make uncertainty, data provenance, and operational status visible without
turning every metric into an alarm.

Design principles:

1. **Calm before dense** — show the decision summary first; expose detailed
   configuration progressively.
2. **Evidence before recommendation** — every conclusion links back to data,
   model, period, and source.
3. **Research is not execution** — clearly distinguish research, licensed, and
   executable data tiers.
4. **Human remains in control** — AI and RL outputs are supporting evidence;
   live-order language is never implied when no broker is connected.
5. **Risk is a first-class object** — drawdown, stale data, model mismatch, and
   uncertainty appear beside return metrics.

## Information architecture

### Primary navigation

- **Today** — portfolio context, market state, watchlist, active alerts, and the
  next safe action.
- **Stocks** — ticker search, technical history, fundamentals, news sentiment,
  correlation, and AI explanation.
- **Experiments** — dataset configuration, indicators, agent, training runs,
  model registry, and out-of-sample evaluation.
- **Monitor** — daily IDX monitor, notification history, and Telegram delivery.

### Secondary navigation

- **Connections** — market-data providers, broker paper account, and AI router.
- **Documentation** — operating guide, engine diagram, paper roadmap, and output
  catalog.
- **Settings** — saved configurations, secrets status, defaults, and appearance.

### Mapping from the existing dashboard

| Existing page | nezu-trade destination |
|---|---|
| Beranda | Today |
| Data Market | Experiments / Data |
| Data Sources & Broker | Connections / Data & Broker |
| Analisa IDX | Stocks / Market Scanner |
| Training | Experiments / Training |
| Model & Evaluasi | Experiments / Models |
| AI Research Copilot | Stocks / Research |
| Monitoring & Notifikasi | Monitor |
| Paper Trading | Connections / Paper Trading |
| Dokumentasi & Output | Documentation |

## Visual system

### Typography

- Product typeface: **DM Sans**.
- Display: 32/40, SemiBold.
- Page title: 24/32, SemiBold.
- Section title: 18/26, SemiBold.
- Body: 14/22, Regular.
- Label: 12/18, Medium.
- Numeric metrics: tabular numerals where available.

### Color direction

| Token | Value | Purpose |
|---|---:|---|
| Canvas | `#F5F6F2` | warm neutral page background |
| Surface | `#FFFFFF` | primary cards and panels |
| Surface muted | `#EEF1EC` | grouped controls and secondary panels |
| Ink | `#202923` | primary text |
| Ink muted | `#68716B` | secondary text |
| Border | `#DDE2DC` | quiet structural separation |
| Moss | `#356A52` | primary action and trusted/healthy state |
| Moss soft | `#DDEBE3` | selected navigation and positive context |
| Amber | `#A97732` | caution, stale, and review-required state |
| Amber soft | `#F5E9D7` | caution surface |
| Clay | `#A95954` | destructive/error only |
| Clay soft | `#F4E1DF` | error surface |
| Blue gray | `#566A79` | neutral informational status |

Green and red must not be the only carriers of meaning. Every status also uses a
label and icon so the product remains legible for color-vision differences.

### Shape and depth

- Card radius: 16 px.
- Input/button radius: 10 px.
- Pills: full radius.
- Default spacing unit: 4 px; common gaps 8, 12, 16, 24, and 32 px.
- Shadows are restrained: `0 8 24 rgba(32,41,35,.06)` for floating surfaces.
- Prefer borders and tonal grouping over heavy shadows.

## Screen 01 — Today

Desktop frame: 1440 × 1024.

Structure:

1. 248 px left navigation with nezu-trade wordmark, primary items, connection
   status, and user/settings footer.
2. 72 px top bar with current market session, global ticker search, data-source
   health, and last refresh.
3. Main content with a gentle greeting and “What deserves attention today?”
4. Four compact context cards: Market regime, Data freshness, Active model, and
   Monitor status.
5. Two-column layout:
   - left: watchlist with price, daily move, RSI, trend, data tier, and evidence
     freshness;
   - right: “Decision notes” card combining technical, historical risk, news,
     and explicit uncertainty.
6. Bottom row: equity curve preview and recent notification timeline.

Primary action: **Review market**. Secondary action: **Run daily analysis**.

## Screen 02 — Stock Intelligence

Header includes ticker `BBCA`, company name, exchange, price, timestamp, source
tier, and a visible `Research only`/`Licensed` badge.

Tabs:

- Overview
- Technical
- Fundamentals
- News & sentiment
- Model evidence
- Relationships

Overview composition:

1. Price chart with volume and optional indicator overlays.
2. Compact signals strip: RSI, MACD, SMA regime, volatility, drawdown.
3. Evidence stack with three separately labeled perspectives:
   - Technical condition
   - Historical/model evidence
   - Fundamental/news context
4. “What could invalidate this view?” block beside the AI summary.
5. Correlation heatmap and nearest related stocks.

Never show a bare BUY/SELL badge. Use language such as `Momentum strengthening`,
`Risk elevated`, or `Evidence mixed`, followed by timestamp and rationale.

## Screen 03 — Experiment Lab

Use a three-step layout instead of one long sidebar:

1. **Data & universe** — saved configuration, ticker universe, source, interval,
   train/validation/test windows, and data-quality gate.
2. **Agent & features** — indicators, risk feature, algorithm, timesteps, seed,
   transaction-cost assumptions, and advanced JSON parameters.
3. **Review & run** — configuration diff, state/action dimensions, warnings,
   estimated workload, and run button.

The right side contains a persistent experiment summary. Training runs appear in
a table with status, agent, universe, period, seed, OOS Sharpe, max drawdown, and
artifact path. Model comparison defaults to distribution/median across seeds,
not the best single result.

## Screen 04 — Connections

Provider cards are grouped by responsibility:

- Market data
- Broker / paper account
- AI router
- Notification channels

Each card shows status, environment, trust tier, last successful check, secret
presence (never the secret), and one explicit action. Broker order capability is
shown as `Disabled`, `Paper`, or `Live gated`.

The paper-trading panel displays the correct Alpaca endpoint without `/v2`, a
read-only connection-test action, and a warning when ticker symbols are `.JK`.

## Reusable components

- App shell / sidebar
- Navigation item: default, hover, selected, disabled
- Status badge: neutral, healthy, caution, error, research, licensed, executable
- Metric card with value, context, timestamp, and optional trend
- Evidence card with source, confidence, age, and invalidation note
- Ticker row
- Source-health row
- Button: primary, secondary, quiet, danger
- Input, select, date range, multi-select, slider, toggle
- Empty, loading, stale, error, and disconnected states
- Chart card and legend
- Data table with sticky header and density control
- Confirmation dialog for paper actions

## Required UX states

Every data-dependent screen must define:

- loading;
- empty/no ticker;
- partial source failure;
- stale data;
- unauthorized provider;
- model/config mismatch;
- unsupported ticker/broker combination; and
- success with provenance and retrieval timestamp.

## Accessibility and tone

- Minimum body contrast 4.5:1.
- Keyboard focus is always visible.
- Click targets are at least 40 × 40 px.
- Charts cannot rely on color alone.
- Avoid “AI says buy”. Prefer “Model evidence”, “Technical condition”, and
  “Factors to review”.
- Error copy explains the next safe action in plain Indonesian.
