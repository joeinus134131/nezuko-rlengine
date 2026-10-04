# Investment Planning Engine — FinRL Workbench

> Dokumen rancangan agar FinRL Workbench menjadi platform perencanaan investasi
> yang kuat, jujur secara matematis, dan mudah dipakai orang awam yang paham
> teknologi dan sudah ikut kelas saham advanced.
>
> Hasil analisis dan model adalah alat riset, bukan rekomendasi investasi.
> Tidak ada klaim akurasi 90% tanpa terbantahkan di pasar saham.

## 1. Tujuan

Membangun satu menu baru: **Perencana Investasi / Cek Momen** yang menjawab:

1. Model RL apa yang diuji dan apakah bertahan out-of-sample?
2. Emiten apa yang cocok masuk saat ini per momen?
3. Bagaimana teknikal + fundamental + makro/mikro dikorelasikan?
4. Bagaimana 9Router mengagregasi banyak knowledge jadi satu kesimpulan rigid?
5. Bagaimana semua itu tetap user-friendly?

Prinsip utama:

- Semua klaim AI harus terikat ke angka + tanggal + sumber.
- Semua skor harus transparan bobotnya.
- Semua keputusan harus punya kondisi pembatal (invalidation).
- Tidak ada angka akurasi tunggal yang dijanjikan.

## 2. Arsitektur Ringkas

```text
DATA -> KUANT + RL -> FUNDAMENTAL -> REZIM MAKRO/MIKRO -> AGREGATOR AI 9Router
  |          |             |                |                      |
  OHLCV   backtest OOS   PER/PBV/      BI rate, inflasi,      bull/bear/judge
  +indikator walk-forward ROE/DER/      IDR, yield,            + grounding JSON
  +risk      multi-seed   FCF/div       komoditas,             + audit log
  +turbulence biaya+slip  likuiditas    breadth IHSG
                                    |
                                    v
              Kartu Keputusan: skor 0-100 + label + bull vs bear + pembatal
```

Fondasi yang sudah ada dan dipakai ulang:

| Komponen lama | Dipakai untuk |
|---|---|
| `finrl/analytics/indonesia.py` | Screener teknikal, risk, korelasi, breadth |
| `finrl/meta/data_processor.py` | Download, cleaning, indikator, VIX/turbulence |
| `finrl/train.py`, `finrl/test.py`, `finrl/trade.py` | Training, backtest, paper trading |
| `finrl/integrations/ai_router.py` | Client `/models` + `/chat/completions` untuk 9Router/OpenRouter |
| `finrl/integrations/provider_adapter.py` | Tier `research/licensed/executable` |
| `finrl/dashboard.py` `show_ai_research` | Preview data, consent, fallback model, suhu rendah |
| `configs/ai_research.json` | Profil Copilot tersimpan |

## 3. Lapisan 1 — Data

### 3.1 Harga dan teknikal

Tetap pakai alur sekarang:

- Sumber default IDX: `yahoofinance` + suffix `.JK`, interval `1D`.
- Indikator di `finrl/config.py`: `macd, boll_ub, boll_lb, rsi_14, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma`.
- Fitur risiko: VIX mati untuk IDX, pakai turbulence kovarians otomatis.
- Wajib cek: tanggal awal/akhir per ticker, missing close, aksi korporasi, satuan volume.

### 3.2 Fundamental IDX

File baru: `finrl/fundamentals/idx.py`

Sumber tahap awal:

1. Snapshot Yahoo yang sudah ada di Copilot: valuasi, pertumbuhan, profitabilitas, leverage, target harga, headline.
2. Input manual/CSV untuk field yang sering bolong di API gratis:
   `PER, PBV, ROE, DER, FCF yield, dividend yield, EPS growth, NPM, current ratio`.
3. Cache 15 menit, tampilkan tanggal snapshot.

Output:

- `fundamental_score 0-100` berbasis percentile lintas universe, bukan nilai absolut.
- Flag kualitas: `data_lengkap / sebagian / manual`.
- Tabel per ticker: valuasi vs median sektor, profitabilitas, leverage, dividen, likuiditas 20 hari.

### 3.3 Makro dan mikro

File baru: `finrl/macro/regime.py`

Field awal (bisa manual dulu, otomatis nanti):

- Makro negara: `BI rate, inflasi, IDR/USD, yield obligasi 10Y, PMI manufaktur`.
- Komoditas relevan IDX: `batubara, CPO, nikel, minyak`.
- Mikro pasar: `tren IHSG, breadth advancers, % di atas SMA20/50, VIX atau turbulence`.
- Kalender: `musim dividen, musim laporan Q1-Q4, keputusan BI/Fed`.

Output:

```text
rezim: RISK_ON / NEUTRAL / RISK_OFF + alasan list
```

Contoh aturan awal:

- `IDR melemah tajam + yield naik = batasi skor bank 55, prefer komoditas ekspor`.
- `breadth lemah + IHSG di bawah SMA50 = max label NEUTRAL`.
- `musim dividen = boost bank BUMN likuid dengan dividend yield tinggi`.

## 4. Lapisan 2 — Kuant dan RL yang Tidak Menipu Diri

RL di sini menjawab policy, bukan ramalan harga:

> Tindakan apa yang diharapkan menaikkan reward berdasarkan state yang dilihat?

State sekarang: kas, turbulence, flag turbulence, harga, kepemilikan, cooldown, indikator teknikal.
Action: kontinu `[-1, 1]` per ticker, diubah jadi jumlah beli/jual.
Reward: perubahan total nilai aset setelah biaya `0,1%`.

File baru: `finrl/decision/backtest_guard.py`

Evaluasi wajib sebelum model dipakai sebagai sinyal:

1. Split waktu: contoh `train 2014-2019, validation 2020, test 2021-2023`.
2. Walk-forward dan 3-5 random seed.
3. Biaya + slippage realistis IDX: lot 100, fee sekuritas, spread.
4. Metrik wajib:
   `total return, CAGR, volatility, max drawdown, Sharpe, Sortino, VaR95, CVaR95, win rate, profit factor, expectancy, equity curve`.
5. Guard:
   - Tolak jika test hanya bagus di 1 periode.
   - Tolak jika return hanya dari 1 ticker / 1 lonjakan.
   - Flag overfit sederhana: walk-forward efficiency + konsistensi antar seed.
   - Bandingkan vs benchmark equal-weight buy-and-hold.

## 5. Lapisan 3 — Skor Gabungan Momen

File baru: `finrl/decision/scoring.py`

Formula awal yang transparan:

```text
composite_moment = 0.30*teknikal + 0.25*risk_adj + 0.25*fundamental + 0.20*rezim_makro
```

Gating (lebih penting dari bobot):

- Jika `rezim=RISK_OFF` maka skor max `55`.
- Jika `avg_value_idr_20d < threshold` maka diskualifikasi likuiditas.
- Jika `max_drawdown > toleransi user` maka `-15` poin.
- Jika data fundamental tidak lengkap maka beri penalti ketidakpastian, bukan dikarang.

Profil user awam:

| Profil | Efek |
|---|---|
| Konservatif | Penalti volatilitas besar, max 1-2 emiten volatil |
| Seimbang | Bobot default |
| Agresif | Toleransi drawdown lebih besar, tetap ada guard likuiditas |

Output per emiten:

- `skor 0-100`
- `label: STRONG WATCH POSITIVE / NEUTRAL / CAUTION`
- `return_1m/3m/6m/12m, Sharpe, maxDD, RSI, posisi vs SMA20/50/200`
- `valuasi vs sektor, likuiditas, kecocokan rezim`

Label adalah klasifikasi teknikal relatif, bukan instruksi beli/jual.

## 6. Lapisan 4 — Agregator AI 9Router Anti-Halu

File baru: `finrl/integrations/aggregator.py`

### 6.1 Grounding wajib

Prompt hanya berisi:

- JSON angka dari lapisan 1-3.
- Berita/laporan yang ditempel user + tanggal + URL/penerbit.
- Konfigurasi eksperimen + periode + risk feature.
- Hasil backtest OOS bila sudah ada.

Dilarang mengirim klaim harga tanpa sumber. Sediakan `Preview data yang akan dikirim` dan checkbox persetujuan seperti Copilot sekarang.

### 6.2 Debat 3 peran

Gunakan 9Router `http://localhost:20128/v1` atau gateway kompatibel OpenAI:

1. `Analis Bull`: cari alasan mendukung dari data.
2. `Analis Bear / Risk Reviewer`: cari yang membatalkan, skenario rugi, data yang kurang.
3. `Hakim`: gabungkan jadi kesimpulan + skor keyakinan `0-100` + kondisi pembatal.

Konfigurasi:

- Model utama + 1-2 fallback berurutan di `configs/ai_research.json`.
- Untuk Combo 9Router, pakai nama combo sebagai Model ID.
- Suhu: `0.1-0.2` untuk laporan stabil, `0.2-0.35` untuk analisis rutin. Suhu tidak mengubah RL/backtest.
- Retry otomatis saat HTTP `403/429/5xx` ke fallback berikutnya.

### 6.3 Skema JSON ketat

Setiap jawaban AI wajib dipisah:

```json
{
  "fakta": [],
  "inferensi": [],
  "ketidakpastian": [],
  "bullish_case": [],
  "bearish_case": [],
  "risiko": [],
  "invalidation": [],
  "verifikasi_lanjutan": [],
  "confidence": 0
}
```

Validasi:

- Angka yang tidak ada di input ditolak dan di-retry.
- Tanpa sumber tanggal untuk berita, tandai sebagai `unverified`.
- Simpan audit log: prompt hash, model, suhu, waktu, respons mentah.

### 6.4 Contoh kartu keputusan

```text
BBCA.JK — 78 / STRONG WATCH POSITIVE — confidence 72
Kenapa: di atas SMA50, RSI 58 netral kuat, Sharpe 1.2, PBV di bawah median bank besar, likuiditas top 5, rezim netral mendukung bank.
Risiko: IDR melemah + yield naik bisa tekan valuasi; drawdown historis -18%.
Batal jika: close < SMA50 3 hari + volume naik, atau laporan Q2 NIM turun 2 kuartal beruntun.
Verifikasi: cek laporan keuangan Q terbaru + kalender BI rate.
```

## 7. UX untuk Orang Awam

Satu halaman baru di dashboard: `show_planner()` — **Perencana Investasi / Cek Momen**.

Tiga tombol besar:

1. `Cek Momen Hari Ini`: pilih universe, klik, tunggu. Tampilkan ranking + label warna.
2. `Kenapa?`: kartu per ticker berisi grafik normalized price, drawdown, tabel bull vs bear, metrik kunci.
3. `Apa yang membatalkan?`: list kondisi exit/invalidasi + checklist verifikasi manual.

Detail UX:

- Bahasa Indonesia semua, tooltip tiap istilah.
- Tidak perlu edit JSON untuk mulai; JSON hanya untuk advanced tuning.
- Unduh 1 paket: konfigurasi + CSV data + CSV screener + CSV backtest + prompt + jawaban AI.
- Mode konservatif/seimbang/agresif sebagai slider, bukan parameter mentah.
- Peringatan nyata: biaya, lot 100, spread, pajak, kebutuhan keuangan pribadi tidak dimodelkan penuh.

## 8. Struktur File yang Diusulkan

```text
finrl/
  fundamentals/
    __init__.py
    idx.py              # skor fundamental + flag kelengkapan data
  macro/
    __init__.py
    regime.py           # rezim RISK_ON/NEUTRAL/RISK_OFF + alasan
  decision/
    __init__.py
    scoring.py          # composite_moment + gating + profil risiko
    backtest_guard.py   # walk-forward, multi-seed, overfit flag
    explainer.py        # susun kartu bull/bear/invalidation untuk UI
  integrations/
    aggregator.py       # debat bull/bear/judge via 9Router + validasi JSON
  dashboard.py          # tambah show_planner()
configs/
  planner.json          # bobot skor, threshold likuiditas, profil risiko (non-secret)
  ai_research.json      # base_url, model utama, fallback, suhu, mode (non-secret)
```

Secret tetap di OS vault / DB terenkripsi, tidak di JSON/Git.

## 9. Implementasi & Verifikasi yang Telah Selesai (Completed Engine)

Seluruh lapisan arsitektur telah diimplementasikan, diuji, dan terintegrasi ke dalam FinRL Workbench:

1. **`finrl/fundamentals/idx.py`**:
   - Skoring fundamental komparatif median sektor (PER, PBV, ROE, DER, FCF yield, Dividend yield).
   - Penghitungan metrik kelengkapan data (*completeness factor*) sebagai proteksi ketidakpastian.
2. **`finrl/macro/regime.py`**:
   - Klasifikasi rezim pasar transparan (`RISK_ON`, `NEUTRAL`, `RISK_OFF`) berbasis data makro (BI Rate, inflasi YoY, yield SUN 10Y, kurs USD/IDR, harga komoditas ekspor) dan *market internals* (% saham > SMA50/SMA200, *advance-decline breadth*).
   - Penyesuaian skor spesifik sektoral dan emiten eksportir USD.
3. **`finrl/decision/scoring.py`**:
   - Skoring 4-pilar komposit dinamis: Teknikal (30%), Risiko (25%), Fundamental (25%), Makro/Rezim (20%).
   - *Hard gating rules*: ambang batas likuiditas BEI minimum Rp 2 Miliar/hari, penalti *max drawdown*, pencegahan *falling knife* (di bawah SMA200 + momentum negatif), serta pembatasan skor saat rezim `RISK_OFF`.
   - *Trade Plan builder*: penyesuaian fraksi harga resmi BEI (Rp 1, Rp 2, Rp 5, Rp 10, Rp 25), pembulatan 1 lot = 100 lembar, stop loss berbasis ATR, target profit $\ge 2.0 \times$ Risk, dan diversifikasi bobot portofolio *inverse-volatility*.
4. **`finrl/decision/signal_study.py`**:
   - Evaluasi kekuatan prediktif sinyal secara *Point-in-Time* (PIT) lintas waktu.
   - Metrik rigid: Spearman Rank Information Coefficient (IC), IC Information Ratio (ICIR), empirical hit rate dengan Wilson 95% Confidence Interval, serta Brier score untuk kalibrasi probabilitas.
5. **`finrl/decision/backtest_guard.py`**:
   - Pelindung overfit model RL: Bailey & Lopez de Prado *Probabilistic Sharpe Ratio* (PSR) dan *Deflated Sharpe Ratio* (DSR).
   - *Stationary Block Bootstrap* 95% confidence interval untuk Sharpe ratio.
   - Rasio *Walk-Forward Efficiency* (WFE = $SR_{OOS} / SR_{IS}$).
   - Uji konsentrasi return top 1% hari perdagangan untuk mendeteksi kinerja semu dari anomali sesaat.
6. **`finrl/integrations/aggregator.py`**:
   - Panel AI multi-agen via 9Router: *Bullish Analyst* vs *Bearish Risk Manager* vs *Hakim Komite Investasi*.
   - **Grounding Verification Ratio $\ge 90\%$ (Anti-Halu)**: setiap angka yang dikutip AI diverifikasi secara matematis terhadap *Evidence Pack* kuantitatif. Jika rasio grounding $< 90\%$, penyesuaian skor AI otomatis dibatalkan/dinetralkan.
   - Pembatasan nudging skor AI maksimal $\pm 10$ poin terhadap jangkar kuantitatif.
7. **`finrl/decision/explainer.py`**:
   - Sintesis argumen pendukung dan risiko, perumusan pemicu invalidasi objektif (stop loss tertembus, breakdown SMA50 3 hari, atau pemburukan makro), serta checklist aksi korporasi BEI sebelum order dipasang.
   - Generator ekspor laporan terstruktur dalam format Markdown.
8. **`finrl/decision/ledger.py`**:
   - Buku catatan keputusan (*Decision Ledger*) persisten untuk forward-testing empiris.
   - Evaluasi hasil riil pada horizon T+5, T+20, dan T+60 hari bursa, penghitungan *realized hit rate*, dan kalibrasi Brier score.
9. **`finrl/decision/planner_view.py` & `finrl/dashboard.py`**:
   - Antarmuka Streamlit interaktif yang ramah pengguna awam berwawasan teknologi pada menu **"💡 Perencana Investasi"** dengan 6 tab terstruktur.

---

## 10. Analisis Lanjutan & Metodologi Matematis

### 10.1 Analisis Kekuatan Sinyal & Kalibrasi Probabilitas (Point-in-Time Study)
Untuk memastikan model tidak hanya "halusinasi prediktif", platform menerapkan verifikasi empiris lintas cross-section saham:
- **Information Coefficient (IC)**:
  $$\text{IC}_t = \text{Corr}_{\text{Spearman}}(\text{Score}_t, \text{Return}_{t \to t+h})$$
- **Information Ratio (ICIR)**:
  $$\text{ICIR} = \frac{\mu(\text{IC})}{\sigma(\text{IC})}$$
  Sinyal dikategorikan **Kuat** bila $\text{IC} > 0.05$ dan $\text{ICIR} > 0.5$, **Sedang** bila $\text{IC} > 0.02$, dan **Lemah/Terbalik** bila $\text{IC} \le 0.02$.
- **Wilson Score Interval (95% CI)**:
  Memberikan batas atas dan batas bawah konservatif untuk *hit rate* historis pada sampel kecil, mencegah over-confidence.
- **Brier Calibration Score**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - o_i)^2$$
  Mengukur apakah probabilitas yang diberikan model benar-benar terkalibrasi dengan realitas frekuensi relatif di bursa.

### 10.2 Guard Model RL terhadap Overfitting (Bailey & Lopez de Prado)
Model Reinforcement Learning yang dilatih dengan ribuan iterasi rentan terhadap *selection bias under multiple testing*. Platform menerapkan:
- **Probabilistic Sharpe Ratio (PSR)**:
  Menghitung probabilitas bahwa Sharpe ratio riil lebih besar dari threshold benchmark ($SR^*$), mengoreksi *skewness* ($\gamma_3$) dan *kurtosis* ($\gamma_4$) return harian:
  $$\text{PSR}(SR^*) = Z\left( \frac{(SR - SR^*)\sqrt{T-1}}{\sqrt{1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4} SR^2}} \right)$$
- **Deflated Sharpe Ratio (DSR)**:
  Mengoreksi inflasi Sharpe ratio akibat banyaknya percobaan model ($N$ konfigurasi/trials) dan varians antar uji coba:
  $$\text{DSR} = \text{PSR}(SR_0), \quad SR_0 = \sqrt{\mathbb{V}[\{SR_n\}]} \left( (1-\gamma) Z^{-1}\left(1 - \frac{1}{N}\right) + \gamma Z^{-1}\left(1 - \frac{1}{N e}\right) \right)$$
  Jika DSR $< 0.95$, model ditandai sebagai indikasi *overfit / data mining fluke*.

### 10.3 Mekanisme Anti-Halusinasi AI Multi-Agen (9Router)
Interaksi AI tidak mengizinkan prompt bebas tanpa batas (*unconstrained prompt*):
1. **Evidence Pack Builder**: Data numerik dikompilasi ke dalam format JSON yang bersih dan dinormalisasi.
2. **Numeric Extraction & Token Parsing**: Parser regex mendeteksi setiap angka, persen, skala satuan, serta pemisah desimal internasional maupun Indonesia (titik/koma).
3. **Grounding Verification ($\ge 90\%$)**: Setiap angka yang dihasilkan pada bagian analisis Bullish, Bearish, dan Hakim dibandingkan dengan angka yang tersedia di *Evidence Pack* dengan toleransi presisi desimal dan skala perkalian ($100\times$, $0.01\times$, $10^9$, dll.).
4. **Sanksi Ketidaksesuaian**: Jika rasio angka valid $< 90\%$, penyesuaian skor AI otomatis dinolkan (`penyesuaian_skor = 0`) dan sistem memberikan status peringatan *unverified / hallucination detected*.

### 10.4 Regulasi Perdagangan Bursa Efek Indonesia (IDX)
Perencanaan posisi mematuhi ketentuan mikrostruktur BEI:
- **Tick Size Rule**:
  - Harga $< \text{Rp } 200$: Fraksi Rp 1
  - Harga $\text{Rp } 200 - \text{Rp } 500$: Fraksi Rp 2
  - Harga $\text{Rp } 500 - \text{Rp } 2.000$: Fraksi Rp 5
  - Harga $\text{Rp } 2.000 - \text{Rp } 5.000$: Fraksi Rp 10
  - Harga $\ge \text{Rp } 5.000$: Fraksi Rp 25
- **Lot Sizing**:
  Seluruh volume order dihitung dalam satuan bulat kelipatan 100 lembar saham.
- **Dynamic ATR Stop & Reward:Risk Ratio**:
  Level proteksi stop loss dipatok minimum $1.5 \times \text{ATR}(14)$ di bawah harga entry, dan target profit dihitung agar memenuhi rasio $\text{Reward : Risk} \ge 2.0$.

---

## 11. Panduan Operasional Pengguna Awam

1. **Memulai Analisis**:
   - Buka menu **"💡 Perencana Investasi"** di dashboard FinRL.
   - Pilih preset semesta saham (misal: *LQ45*, *Big Banks*, atau masukkan ticker pilihan Anda).
   - Tentukan profil risiko Anda: **Konservatif** (fokus proteksi modal & dividen), **Seimbang** (bobot standar), atau **Agresif** (toleransi drawdown lebih tinggi).
2. **Menilai Sinyal**:
   - Perhatikan kolom **Skor** (0–100) dan **Label** (*STRONG WATCH POSITIVE*, *WATCH POSITIVE*, *NEUTRAL*, *CAUTION*, *ILLIQUID*).
   - Buka **Tab 2 (Kartu Detail)** untuk membaca ringkasan eksekutif, faktor teknikal & fundamental, serta **Kondisi Pembatal (Invalidation Triggers)**.
3. **Mengeksekusi Trade Plan**:
   - Masukkan modal alokasi Anda di **Tab 3 (Trade Plan)**.
   - Sistem langsung menampilkan: **Jumlah Lot**, **Harga Beli Limit**, **Level Stop Loss**, dan **Target Profit** yang sudah disesuaikan dengan fraksi harga resmi BEI.
4. **Menjalankan Debat AI**:
   - Di **Tab 4 (Panel AI)**, pilih model 9Router (misal: DeepSeek, Claude, atau GPT-4o).
   - Klik **"Jalankan Debat Panel AI"** untuk memperoleh sudut pandang Bull vs Bear beserta skor grounding-nya.
5. **Mencatat ke Decision Ledger**:
   - Klik tombol **"Catat ke Decision Ledger"** untuk menyimpan tesis investasi ke dalam rekam jejak forward-test sistematis.
