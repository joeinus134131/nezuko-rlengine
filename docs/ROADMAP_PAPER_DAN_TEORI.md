# Roadmap Paper dan Teori FinRL Workbench

Dokumen ini menghubungkan literatur akademik dengan elemen yang benar-benar ada
di aplikasi. Tujuannya bukan mencari paper yang menjanjikan profit, tetapi memahami
asumsi, cara pengujian, dan kondisi yang membuat suatu hasil layak dipercaya.

## 1. Peta mental platform

Alur inti aplikasi dapat dibaca sebagai berikut:

```text
OHLCV + indikator + risk feature
              ↓
state pasar → environment → agent RL → action beli/tahan/jual
                   ↑                     ↓
             reward perubahan nilai portofolio setelah biaya
                                         ↓
                   out-of-sample backtest → paper test → keputusan manusia
```

Pada implementasi sekarang:

- **state** berisi kas, turbulence, penanda turbulence, harga, kepemilikan,
  cooldown, dan indikator teknikal;
- **action** bersifat kontinu `[-1, 1]` per saham dan dikonversi menjadi jumlah
  saham maksimum;
- **reward** adalah perubahan total nilai aset yang diberi scaling;
- biaya beli dan jual default masing-masing `0,1%`;
- saat turbulence melewati threshold, environment melikuidasi posisi; dan
- output model harus dinilai pada periode yang tidak dipakai saat training.

Artinya, model tidak “meramal harga besok” secara langsung. Model belajar policy:
tindakan apa yang diharapkan meningkatkan reward berdasarkan state yang dilihat.

## 2. Urutan baca yang disarankan

### Tahap A — memahami mesin FinRL

1. **FinRL: Deep Reinforcement Learning Framework to Automate Trading in
   Quantitative Finance** — Liu et al. (2021),
   [arXiv:2111.09395](https://arxiv.org/abs/2111.09395).
   Baca lebih dahulu karena menjelaskan arsitektur data–environment–agent dan
   friksi pasar yang menjadi dasar aplikasi ini.
2. **FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading** —
   Liu et al. (2020),
   [arXiv:2011.09607](https://arxiv.org/abs/2011.09607).
   Fokus pada definisi state, action, reward, training, validation, dan trading.
3. **A Deep Reinforcement Learning Framework for the Financial Portfolio
   Management Problem** — Jiang, Xu, dan Liang (2017),
   [arXiv:1706.10059](https://arxiv.org/abs/1706.10059).
   Berguna untuk memahami portfolio vector, reward, dan pentingnya memasukkan
   commission dalam simulasi.

Setelah tahap ini, buka kode `env_stocktrading_np.py` dan cocokkan setiap istilah
paper dengan `get_state()`, `step()`, action space, dan reward.

### Tahap B — memahami algoritma yang tersedia

| Model di UI | Paper utama | Inti yang perlu dipahami |
|---|---|---|
| A2C | Mnih et al., **Asynchronous Methods for Deep Reinforcement Learning**, [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) | actor, critic, advantage, on-policy |
| DDPG | Lillicrap et al., **Continuous Control with Deep Reinforcement Learning**, [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) | deterministic actor, replay buffer, target network |
| PPO | Schulman et al., **Proximal Policy Optimization Algorithms**, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | clipped objective dan pembatasan perubahan policy |
| TD3 | Fujimoto et al., **Addressing Function Approximation Error in Actor-Critic Methods**, [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) | twin critics, delayed update, pengurangan overestimation |
| SAC | Haarnoja et al., **Soft Actor-Critic**, [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) | off-policy, entropy, eksplorasi, stabilitas |

Jangan menyimpulkan satu algoritma selalu terbaik. Bandingkan beberapa random
seed, periode, dan hyperparameter dengan data test yang sama.

### Tahap C — teknikal, portfolio, dan risiko

1. Markowitz, **Portfolio Selection** (1952),
   [DOI:10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).
   Dasar hubungan expected return, variance, covariance, dan diversifikasi.
2. Sharpe, **Mutual Fund Performance** (1966),
   [JSTOR:2351746](https://www.jstor.org/stable/2351746).
   Dasar interpretasi reward terhadap risiko yang digunakan pada evaluasi Sharpe.
3. Brock, Lakonishok, dan LeBaron, **Simple Technical Trading Rules and the
   Stochastic Properties of Stock Returns** (1992),
   [DOI:10.1111/j.1540-6261.1992.tb04681.x](https://doi.org/10.1111/j.1540-6261.1992.tb04681.x).
   Membantu menempatkan moving average dan sinyal teknikal sebagai hipotesis yang
   harus diuji, bukan kebenaran universal.
4. Kritzman dan Li, **Skulls, Financial Turbulence, and Risk Management** (2010),
   [DOI:10.2469/faj.v66.n5.3](https://doi.org/10.2469/faj.v66.n5.3).
   Referensi langsung untuk gagasan turbulence yang digunakan sebagai risk gate.
5. Fama, **Efficient Capital Markets: A Review of Theory and Empirical Work**
   (1970), [PDF](https://www.e-m-h.org/Fama70.pdf).
   Bacaan penyeimbang: sinyal berbasis histori harus mengalahkan biaya, risiko,
   dan benchmark sebelum dianggap informatif.

RSI berasal dari karya J. Welles Wilder, *New Concepts in Technical Trading
Systems* (1978), yang berbentuk buku, bukan paper. RSI sebaiknya dipakai sebagai
fitur state atau filter eksperimen; nilai 30/70 bukan jaminan reversal.

### Tahap D — backtest yang tidak menipu diri sendiri

Bagian ini sama pentingnya dengan paper RL:

1. White, **A Reality Check for Data Snooping** (2000),
   [DOI:10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152).
   Menjelaskan mengapa strategi terbaik dari banyak percobaan bisa menang hanya
   karena kebetulan.
2. Bailey dan López de Prado, **The Deflated Sharpe Ratio** (2014),
   [DOI:10.2139/ssrn.2460551](https://doi.org/10.2139/ssrn.2460551).
   Mengoreksi Sharpe akibat multiple testing, data non-normal, dan selection bias.
3. Hansen, **A Test for Superior Predictive Ability** (2005),
   [DOI:10.1198/073500105000000063](https://doi.org/10.1198/073500105000000063).
   Membandingkan apakah performa model benar-benar lebih baik dari benchmark.
4. Henderson et al., **Deep Reinforcement Learning that Matters** (2017),
   [arXiv:1709.06560](https://arxiv.org/abs/1709.06560).
   Dasar penggunaan banyak seed, confidence interval, dan pelaporan reproducible.
5. Engstrom et al., **Implementation Matters in Deep Policy Gradients** (2020),
   [arXiv:2005.12729](https://arxiv.org/abs/2005.12729).
   Menunjukkan bahwa detail implementasi PPO dapat memengaruhi hasil sebesar atau
   lebih besar daripada nama algoritmanya.

Implikasi praktis untuk aplikasi:

- train dan test harus berurutan secara waktu, tidak diacak;
- jangan memilih model berdasarkan test set berulang-ulang;
- sediakan validation period terpisah untuk memilih model;
- bandingkan dengan buy-and-hold dan benchmark indeks;
- catat semua eksperimen, termasuk yang gagal;
- uji beberapa seed dan laporkan distribusi, bukan hanya run terbaik;
- masukkan fee, pajak, spread, slippage, lot size, dan saham tersuspensi; dan
- hindari survivorship bias dengan universe yang sesuai konstituen pada waktu itu.

### Tahap E — biaya transaksi dan eksekusi

Almgren dan Chriss, **Optimal Execution of Portfolio Transactions** (2000/2001),
[PDF](https://www.math.nyu.edu/~chriss/optliq_f.pdf), menjelaskan trade-off antara
market impact dan risiko ketika order dieksekusi. Ini membantu memahami mengapa
hasil backtest berbasis harga penutupan tidak otomatis dapat direalisasikan.

Untuk IDX, nilai biaya default environment `0,1%` tidak boleh langsung dianggap
sesuai. Eksperimen serius perlu memasukkan fee broker aktual, levy, pajak jual,
spread, fraksi lot 100 saham, dan slippage berdasarkan likuiditas masing-masing
ticker.

### Tahap F — AI Research Copilot dan sentimen

1. Araci, **FinBERT: Financial Sentiment Analysis with Pre-trained Language
   Models** (2019), [arXiv:1908.10063](https://arxiv.org/abs/1908.10063).
   Menjelaskan mengapa bahasa finansial membutuhkan model/domain adaptation.
2. Huang, Wang, dan Yang, **FinBERT: A Large Language Model for Extracting
   Information from Financial Text** (2023),
   [DOI:10.1111/1911-3846.12832](https://doi.org/10.1111/1911-3846.12832).
   Berguna untuk memahami evaluasi out-of-sample pada sentiment classifier.

AI Copilot pada aplikasi berperan merangkum dan menjelaskan evidence, bukan
menjadi ground truth dan bukan menggantikan RL. Setiap klaim berita/fundamental
harus menyimpan sumber, waktu publikasi, waktu pengambilan, dan ticker yang benar.
Sentiment score tanpa evaluasi terhadap label serta periode mendatang belum dapat
disebut sinyal investasi.

## 3. Eksperimen belajar yang direkomendasikan

### Eksperimen 1 — baseline lebih dahulu

- Ticker: 5 saham IDX likuid.
- Data: harian, minimal beberapa rezim pasar.
- Model: PPO dengan parameter default.
- Pembanding: cash, equal-weight, dan buy-and-hold.
- Output: CAGR, volatility, Sharpe, max drawdown, turnover, dan biaya.

Tujuannya bukan mencari return tertinggi, melainkan memastikan pipeline, dimensi,
dan periode tidak bocor.

### Eksperimen 2 — ablation indikator

Jalankan konfigurasi identik dengan: OHLCV saja; OHLCV + RSI; seluruh indikator;
dan seluruh indikator + turbulence. Gunakan seed serta periode yang sama. Jika
penambahan fitur tidak konsisten memperbaiki hasil out-of-sample, fitur tersebut
belum terbukti berguna.

### Eksperimen 3 — stabilitas agent

Bandingkan PPO, SAC, dan TD3 pada sedikitnya 5 seed. Laporkan median dan rentang
hasil. Satu run terbaik tidak cukup untuk menyatakan agent unggul.

### Eksperimen 4 — stress biaya

Ulangi backtest dengan biaya dan slippage rendah, realistis, serta berat. Policy
yang runtuh karena sedikit kenaikan biaya kemungkinan mengeksploitasi turnover,
bukan edge yang kuat.

### Eksperimen 5 — walk-forward

Gunakan beberapa jendela train/validation/test yang maju secara kronologis.
Periksa apakah hasil bertahan pada bullish, bearish, volatil, dan sideways regime.

## 4. Checklist membaca hasil aplikasi

Sebelum menjadikan output sebagai tambahan keputusan, jawab:

1. Apa definisi state, action, dan reward pada eksperimen ini?
2. Apakah ticker dan urutannya sama dengan manifest model?
3. Apakah test benar-benar tidak pernah dipakai memilih hyperparameter?
4. Berapa banyak model/konfigurasi yang dicoba sebelum hasil ini dipilih?
5. Apakah biaya, spread, slippage, lot size, dan corporate action masuk?
6. Apakah hasil mengalahkan benchmark setelah biaya?
7. Apakah performa konsisten pada beberapa seed dan jendela waktu?
8. Apakah drawdown masih dapat diterima secara finansial dan psikologis?
9. Apakah data source, timestamp, serta adjustment dapat diaudit?
10. Apakah kesimpulan AI memiliki sumber dan dapat diverifikasi?

Jika beberapa jawaban masih “tidak”, model tetap berguna sebagai bahan riset,
tetapi confidence untuk keputusan nyata harus diturunkan.
