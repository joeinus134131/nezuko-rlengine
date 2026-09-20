import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/user/Projects/FinRL";
const SKILL_DIR = "/Users/user/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const sourceTemplate = "/Users/user/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-dark-mode/assets/reference.pptx";
const FINAL_PPTX = path.join(workspaceDir, "output", "Laporan_Riset_Emiten_Indonesia_FinRL.pptx");
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const W = 1280, H = 720;
const C = { bg: "#000000", white: "#FFFFFF", muted: "#B7B7B7", panel: "#292929", line: "#454545", cyan: "#6FD3FF", blue: "#1689E8", red: "#FF5B5B", green: "#58D68D", amber: "#F3C969" };
const FONT = "Arial";

const assets = {
  rank: "/Users/user/Downloads/visualization (1).png",
  prices: "/Users/user/Downloads/visualization.png",
  drawdowns: "/Users/user/Downloads/visualization (3).png",
  breadth: "/Users/user/Downloads/visualization (4).png",
  perf: "/Users/user/Downloads/visualization (5).png",
  ddModel: "/Users/user/Downloads/visualization (6).png",
  data: "/Users/user/Desktop/ss/Screenshot 2026-09-20 at 01.54.08.png",
  ranking: "/Users/user/Desktop/ss/Screenshot 2026-09-20 at 01.55.39.png",
  positive: "/Users/user/Desktop/ss/Screenshot 2026-09-20 at 02.29.27.png",
  positiveDD: "/Users/user/Desktop/ss/Screenshot 2026-09-20 at 02.30.18.png",
  negative: "/Users/user/Desktop/ss/Screenshot 2026-09-20 at 10.39.11.png",
};
const bytes = {};
for (const [k, v] of Object.entries(assets)) bytes[k] = new Uint8Array(await fs.readFile(v));

function box(slide, x, y, w, h, fill=C.panel, radius=12, line="none") {
  return slide.shapes.add({ geometry: radius ? "roundRect" : "rect", position: { left:x, top:y, width:w, height:h }, fill, line: { fill: line, width: line === "none" ? 0 : 1 } });
}
function txt(slide, text, x, y, w, h, size=22, color=C.white, bold=false, opts={}) {
  const s = slide.shapes.add({ geometry:"textbox", position:{left:x,top:y,width:w,height:h}, fill:"none", line:{fill:"none",width:0} });
  s.text = text;
  s.text.style = { typeface: FONT, fontSize:size, bold, color, autoFit:"shrinkText", verticalAlignment:opts.vAlign ?? "top", textAlign:opts.align ?? "left" };
  return s;
}
function line(slide, x, y, w, color=C.line, width=1) {
  return slide.shapes.add({ geometry:"rect", position:{left:x,top:y,width:w,height:width}, fill:color, line:{fill:"none",width:0} });
}
function image(slide, key, x, y, w, h, fit="contain", alt=key) {
  return slide.images.add({ blob:bytes[key], contentType:"image/png", alt, fit, position:{left:x,top:y,width:w,height:h}, geometry:"roundRect", borderRadius:10 });
}
function slideBase(title, section="FINRL RESEARCH") {
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  txt(s, section, 42, 22, 400, 18, 12, C.muted, false);
  txt(s, title, 42, 48, 1180, 58, 30, C.white, false);
  line(s, 42, 112, 1196, C.line, 1);
  txt(s, String(presentation.slides.count).padStart(2,"0"), 1194, 682, 44, 18, 11, C.muted, false, {align:"right"});
  return s;
}
function note(slide, extra="") {
  slide.speakerNotes.textFrame.setText(`Sumber: hasil FinRL Workbench dan visualisasi yang diberikan pengguna. Angka merefleksikan konfigurasi serta periode pada tangkapan layar. ${extra}`.trim());
}
function stat(slide, x, y, w, value, label, color=C.white, sub="") {
  box(slide,x,y,w,152,C.panel,10);
  txt(slide,label,x+18,y+16,w-36,24,14,C.muted,false);
  txt(slide,value,x+18,y+47,w-36,55,34,color,false);
  if (sub) txt(slide,sub,x+18,y+112,w-36,25,12,C.muted,false);
}

// 1 Cover
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  txt(s,"FINRL RESEARCH",42,28,260,20,12,C.muted,false);
  txt(s,"Riset kuantitatif\nemiten Indonesia",42,172,700,122,45,C.white,false);
  txt(s,"Screening teknikal, risiko, dan evaluasi strategi deep reinforcement learning",42,320,680,66,20,C.muted,false);
  txt(s,"Universe 11 emiten  |  Data harian  |  Laporan eksperimen",42,614,700,24,14,C.white,false);
  image(s,"prices",820,0,460,720,"cover","Perbandingan harga saham ternormalisasi");
  note(s,"Deck ini merupakan laporan riset, bukan rekomendasi investasi.");
}

// 2 Executive summary
{
  const s = slideBase("Ringkasan eksekutif");
  txt(s,"Sinyal lintas saham mendukung seleksi kandidat, tetapi evaluasi strategi menunjukkan hasil yang berubah antarjendela uji.",42,132,1190,70,25,C.white,false);
  stat(s,42,236,280,"11","Emiten dianalisis",C.white,"LQ45 pilihan pengguna");
  stat(s,342,236,280,"45,5%","Advancers",C.amber,"Breadth belum dominan");
  stat(s,642,236,280,"78,18","Skor tertinggi",C.cyan,"ADRO.JK");
  stat(s,942,236,296,"−4,04%","Hasil uji lemah",C.red,"Jendela evaluasi terbaru");
  txt(s,"Implikasi riset",42,430,250,28,18,C.muted,true);
  txt(s,"Gunakan ranking sebagai prioritas riset, bukan sinyal beli tunggal. Stabilitas model perlu diuji lintas periode sebelum paper trading.",42,466,1120,78,24,C.white,false);
  line(s,42,580,1196,C.line,1);
  txt(s,"Temuan utama: kualitas pemeringkatan emiten lebih konsisten daripada performa strategi A2C pada satu periode tertentu.",42,604,1180,42,20,C.cyan,true);
  note(s,"Skor 78,18 berasal dari tabel ranking. Hasil −4,04% berasal dari bukti kelayakan pada tangkapan layar terakhir.");
}

// 3 Methodology
{
  const s = slideBase("Ruang lingkup dan metodologi","DESAIN EKSPERIMEN");
  const cols=[42,348,654,960];
  const heads=["Universe","Data dan fitur","Model","Evaluasi"];
  const bodies=[
    "ADRO, ANTM, BBCA, BBNI, BBRI, BBTN, BMRI, JPFA, PTBA, TLKM, WIKA",
    "Yahoo Finance, interval 1D. MACD, RSI 14, RSI 30, dan CCI 30 pada konfigurasi awal.",
    "Stable Baselines3, algoritma A2C. Contoh konfigurasi mencatat learning rate 0,0003.",
    "Return, volatilitas, Sharpe, max drawdown, benchmark equal weight, dan pemeriksaan kelayakan."
  ];
  for(let i=0;i<4;i++){
    txt(s,String(i+1).padStart(2,"0"),cols[i],157,70,32,15,C.cyan,true);
    txt(s,heads[i],cols[i],202,250,30,20,C.white,true);
    line(s,cols[i],244,250,C.line,1);
    txt(s,bodies[i],cols[i],268,250,180,17,C.muted,false);
  }
  txt(s,"Catatan periode",42,514,220,25,16,C.white,true);
  txt(s,"Tangkapan layar memuat beberapa konfigurasi train dan test yang berbeda. Karena itu, hasil backtest harus dibaca sebagai eksperimen terpisah, bukan satu seri performa yang sama.",42,550,1160,72,21,C.white,false);
  note(s,"Periode yang terlihat mencakup train mulai 2014 dan beberapa test window 2024 sampai 2026, tergantung konfigurasi.");
}

// 4 breadth
{
  const s = slideBase("Breadth pasar menunjukkan kondisi campuran","SCREENING PASAR");
  stat(s,42,137,270,"45,5%","Advancers",C.amber);
  stat(s,332,137,270,"54,5%","Di atas SMA20",C.cyan);
  stat(s,622,137,270,"63,6%","Di atas SMA50",C.green);
  stat(s,912,137,326,"11","Saham dianalisis",C.white);
  image(s,"breadth",42,326,780,300,"contain","Breadth historis: advancers dan persentase di atas SMA");
  txt(s,"Pembacaan",860,340,300,28,18,C.white,true);
  txt(s,"Lebih banyak saham berada di atas SMA50 daripada SMA20. Kombinasi ini konsisten dengan tren menengah yang masih lebih baik daripada momentum jangka pendek.",860,382,336,160,20,C.muted,false);
  txt(s,"Breadth 45,5% belum menunjukkan partisipasi kenaikan yang luas.",860,566,336,54,18,C.amber,true);
  note(s,"Interpretasi teknikal berbasis persentase yang terlihat pada dashboard.");
}

// 5 ranking
{
  const s = slideBase("Lima emiten memimpin composite score","RANKING EMITEN");
  const rows=[
    ["01","ADRO.JK","78,18","+60,2%","−75,2%"],
    ["02","BBRI.JK","69,70","−20,8%","−60,0%"],
    ["03","PTBA.JK","67,73","+31,9%","−70,0%"],
    ["04","JPFA.JK","64,24","+22,2%","−81,9%"],
    ["05","BBCA.JK","64,09","−20,7%","−55,5%"],
  ];
  txt(s,"Peringkat",42,145,100,20,13,C.muted,true); txt(s,"Ticker",150,145,190,20,13,C.muted,true); txt(s,"Skor",430,145,130,20,13,C.muted,true); txt(s,"Return 12 bln",650,145,190,20,13,C.muted,true); txt(s,"Max drawdown",940,145,200,20,13,C.muted,true);
  line(s,42,177,1196,C.line,1);
  rows.forEach((r,i)=>{ const y=195+i*75; txt(s,r[0],42,y,80,32,15,C.cyan,true); txt(s,r[1],150,y,190,32,21,C.white,true); txt(s,r[2],430,y,130,32,21,C.white,false); txt(s,r[3],650,y,190,32,21,r[3].startsWith("+")?C.green:C.red,false); txt(s,r[4],940,y,200,32,21,C.red,false); line(s,42,y+49,1196,C.line,1); });
  txt(s,"ADRO memimpin skor dan return 12 bulan. JPFA memiliki drawdown terdalam di kelompok lima besar, sehingga ranking tetap perlu dilengkapi batas risiko.",42,598,1170,46,19,C.white,false);
  note(s,"Nilai dibaca dari tabel ranking pada screenshot. Persentase dibulatkan satu desimal pada slide.");
}

// 6 price paths
{
  const s = slideBase("Performa jangka panjang terpisah antar-emiten","PERFORMA HARGA");
  image(s,"prices",42,140,835,420,"contain","Harga ternormalisasi ADRO, BBCA, BBRI, JPFA, dan PTBA");
  txt(s,"Pola yang terlihat",920,151,270,26,18,C.white,true);
  txt(s,"BBCA dan BBRI sempat mencatat kenaikan relatif terbesar pada sebagian besar horizon. ADRO menunjukkan lonjakan siklikal yang tajam. JPFA dan PTBA bergerak lebih datar dengan fase volatilitas yang berbeda.",920,196,290,205,19,C.muted,false);
  txt(s,"Implikasi",920,434,270,26,18,C.white,true);
  txt(s,"Universe ini tidak homogen. Normalisasi harga membantu membandingkan lintasan, tetapi tidak menggantikan analisis return, volatilitas, dan likuiditas.",920,476,290,120,18,C.white,false);
  note(s,"Grafik harga dinormalisasi sekitar 100 pada awal periode. Tanggal spesifik tidak tercantum lengkap pada gambar.");
}

// 7 drawdowns
{
  const s = slideBase("Drawdown besar muncul pada beberapa rezim pasar","RISIKO EMITEN");
  image(s,"drawdowns",42,142,870,410,"contain","Drawdown historis untuk emiten dalam universe");
  stat(s,948,153,290,"−55% s.d. −82%","Rentang max drawdown",C.red,"Lima skor teratas");
  txt(s,"Drawdown historis berulang dan tidak pulih seragam antar-emiten. Risiko ekor tetap material meski composite score tinggi.",948,345,275,120,20,C.white,false);
  txt(s,"Fokus kontrol",948,495,275,26,17,C.muted,true);
  txt(s,"Batas posisi, stop policy, dan evaluasi per rezim perlu diuji bersama model.",948,532,275,80,18,C.cyan,false);
  note(s,"Rentang max drawdown mengacu pada lima emiten teratas di tabel ranking.");
}

// 8 research watchlist
{
  const s = slideBase("Prioritas riset berdasarkan sinyal teknikal","WATCHLIST RISET");
  const cards=[
    ["ADRO.JK","Skor 78,18","Return 12 bulan +60,2%. Tren kuat, drawdown historis tetap tinggi.",C.cyan],
    ["PTBA.JK","Skor 67,73","Return 1 bulan +31,9% dan 3 bulan +27,1%. Momentum paling kuat pada tabel.",C.green],
    ["BBRI.JK","Skor 69,70","Skor tinggi, tetapi return 12 bulan −20,8%. Perlu konfirmasi perubahan tren.",C.amber],
    ["JPFA.JK","Skor 64,24","Return 3 bulan +22,8% dengan max drawdown −81,9%. Profil risiko agresif.",C.red],
  ];
  cards.forEach((c,i)=>{const x=42+(i%2)*600, y=145+Math.floor(i/2)*235; box(s,x,y,560,196,C.panel,10); txt(s,c[0],x+22,y+20,240,30,23,C.white,true); txt(s,c[1],x+330,y+22,195,28,17,c[3],true,{align:"right"}); txt(s,c[2],x+22,y+77,510,86,19,C.muted,false);});
  txt(s,"Watchlist ini mengatur urutan analisis lanjutan. Ia tidak memasukkan valuasi, kualitas laba, tata kelola, atau katalis korporasi.",42,630,1160,36,17,C.white,false);
  note(s,"Watchlist disusun hanya dari tabel screening dan metrik teknikal yang diberikan pengguna.");
}

// 9 positive window
{
  const s = slideBase("A2C menghasilkan return positif, tetapi tertinggal dari benchmark","EVALUASI MODEL 2026");
  stat(s,42,137,250,"12,75%","Return strategi",C.green,"179 observasi");
  stat(s,312,137,250,"−12,63%","Max drawdown",C.red);
  stat(s,582,137,250,"0,57","Sharpe",C.cyan);
  stat(s,852,137,386,"15,55%","Return benchmark",C.white,"Equal weight");
  image(s,"perf",42,330,780,286,"contain","Pertumbuhan nilai strategi AI dan benchmark equal weight");
  txt(s,"Trade-off",864,346,310,28,18,C.white,true);
  txt(s,"Strategi AI mencatat volatilitas tahunan 24,72%, lebih rendah daripada benchmark 35,23%. Namun return dan Sharpe benchmark sedikit lebih tinggi pada jendela ini.",864,392,330,160,20,C.muted,false);
  txt(s,"Kesimpulan periode",864,570,310,24,17,C.cyan,true);
  txt(s,"Ada reduksi risiko, tetapi belum ada alpha yang jelas.",864,603,330,44,18,C.white,false);
  note(s,"Angka berasal dari screenshot evaluasi model: strategi 12,747%, benchmark 15,545%, Sharpe 0,568 vs 0,586.");
}

// 10 drawdown model
{
  const s = slideBase("Profil drawdown AI lebih baik pada sebagian periode","EVALUASI RISIKO MODEL");
  image(s,"ddModel",42,142,840,390,"contain","Drawdown strategi AI dan benchmark equal weight");
  txt(s,"Strategi AI",930,160,260,22,15,C.muted,true);
  txt(s,"−12,63%",930,196,260,48,34,C.cyan,false);
  txt(s,"Benchmark",930,278,260,22,15,C.muted,true);
  txt(s,"−16,31%",930,314,260,48,34,C.white,false);
  line(s,930,388,260,C.line,1);
  txt(s,"Perbaikan drawdown sekitar 3,7 poin persentase pada jendela positif. Keunggulan ini belum cukup menutup selisih return terhadap benchmark.",930,420,275,130,19,C.muted,false);
  txt(s,"Pengukuran perlu diulang dengan biaya transaksi dan beberapa seed model.",930,582,275,54,17,C.amber,true);
  note(s,"Perbandingan drawdown mengacu pada tabel dan grafik evaluasi positif.");
}

// 11 negative window
{
  const s = slideBase("Hasil negatif pada jendela lain menegaskan risiko ketidakstabilan","UJI KETAHANAN");
  stat(s,42,139,330,"−4,04%","Return strategi",C.red);
  stat(s,392,139,330,"−25,43%","Max drawdown",C.red);
  stat(s,742,139,330,"−0,29","Sharpe",C.red);
  box(s,1102,139,136,152,C.panel,10); txt(s,"6/6",1122,182,95,44,31,C.green,false,{align:"center"}); txt(s,"cek dasar lulus",1114,236,112,26,12,C.muted,false,{align:"center"});
  image(s,"negative",42,335,680,286,"cover","Bukti kelayakan dan metrik evaluasi negatif");
  txt(s,"Mengapa penting",768,348,330,28,18,C.white,true);
  txt(s,"Loader, manifest, urutan periode, benchmark, jumlah observasi, dan nilai finite dapat lulus sementara performa ekonomi tetap lemah.",768,392,420,118,20,C.muted,false);
  txt(s,"Kelayakan teknis tidak sama dengan ketahanan strategi. Perbedaan hasil antarjendela menjadi temuan utama riset ini.",768,546,420,76,20,C.red,true);
  note(s,"Tangkapan layar menunjukkan seluruh pemeriksaan dasar lulus, dengan return −4,04%, max drawdown −25,43%, dan Sharpe −0,29.");
}

// 12 conclusion
{
  const s = slideBase("Kesimpulan dan langkah riset berikutnya","KESIMPULAN");
  txt(s,"Kesimpulan",42,144,260,30,19,C.white,true);
  txt(s,"Screening berhasil mempersempit fokus ke ADRO, PTBA, BBRI, JPFA, dan BBCA. Model A2C menunjukkan potensi pengendalian drawdown pada satu periode, tetapi belum stabil lintas jendela uji.",42,190,1120,92,25,C.white,false);
  line(s,42,315,1196,C.line,1);
  const next=[
    ["01","Walk-forward multi-periode","Pisahkan pemilihan model, validasi, dan evaluasi akhir agar tidak bergantung pada satu jendela."],
    ["02","Biaya transaksi dan slippage","Masukkan friksi pasar serta batas likuiditas untuk menguji apakah return tetap bertahan."],
    ["03","Robustness model","Ulangi beberapa random seed, bandingkan A2C dengan PPO dan baseline sederhana."],
    ["04","Overlay fundamental","Tambahkan valuasi, kualitas laba, leverage, dan katalis emiten sebelum membuat keputusan investasi."],
  ];
  next.forEach((r,i)=>{const y=348+i*75; txt(s,r[0],42,y,55,28,14,C.cyan,true); txt(s,r[1],120,y,300,28,18,C.white,true); txt(s,r[2],440,y,760,48,17,C.muted,false); if(i<3) line(s,42,y+58,1196,C.line,1);});
  txt(s,"Laporan ini adalah alat riset kuantitatif dan bukan rekomendasi investasi.",42,663,900,20,13,C.amber,true);
  note(s,"Rekomendasi metodologis didasarkan pada perbedaan hasil yang terlihat antarjendela evaluasi.");
}

const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR,"container_tools/artifact_tool_utils.mjs")).href);
const candidatePath = path.join(stagingDir,"finrl-research-candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const requirements = {
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  verifyArtifactToolImport: true,
};
const fontPolicy = { basis:"design", families:[FONT] };
const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath:FINAL_PPTX,
  pythonExecutable:"/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
  integrityValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit"],
  fontPolicy,
  receiptPath:path.join(stagingDir,"Laporan_Riset_Emiten_Indonesia_FinRL.validation.json"),
});
console.log(JSON.stringify({ final:FINAL_PPTX, result }, null, 2));
