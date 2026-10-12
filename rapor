#!/usr/bin/env python3
"""
Piyasa Raporu - MVP (Hafta 1)
BIST hisseleri, döviz, altın ve kripto için teknik analiz yapar,
her varlığa -100..+100 arası bir skor verir, sonuçları SQLite'a kaydeder
ve günlük HTML rapor üretir.

Kullanım:
    python piyasa_raporu.py            # canlı veri (internet gerekir)
    python piyasa_raporu.py --demo     # sentetik veriyle çevrimdışı test

UYARI: Bu bir karar destek aracıdır, yatırım tavsiyesi değildir.
"""
import argparse
import html
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    TR = ZoneInfo("Europe/Istanbul")
except Exception:  # tzdata yoksa sistem saatine düş
    TR = None

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# AYARLAR - burayı kendi listenle değiştir
# ----------------------------------------------------------------------
VARLIKLAR = {
    "Hisse": {
        "THYAO.IS": "Türk Hava Yolları",
        "GARAN.IS": "Garanti BBVA",
        "AKBNK.IS": "Akbank",
        "ASELS.IS": "Aselsan",
        "BIMAS.IS": "BİM",
        "EREGL.IS": "Ereğli Demir Çelik",
        "KCHOL.IS": "Koç Holding",
        "SISE.IS": "Şişecam",
        "TUPRS.IS": "Tüpraş",
        "FROTO.IS": "Ford Otosan",
    },
    "Endeks": {"XU100.IS": "BIST 100"},
    "Döviz": {"USDTRY=X": "Dolar/TL", "EURTRY=X": "Euro/TL"},
    "Emtia": {"GC=F": "Altın (ons, USD)", "SI=F": "Gümüş (ons, USD)"},
    "Kripto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"},
}
GECMIS_PERIYOT = "1y"
CIKTI_KLASORU = Path(__file__).parent / "raporlar"
DB_YOLU = Path(__file__).parent / "piyasa.db"
SITE_KLASORU = Path(__file__).parent / "site"  # telefondan açılacak yayın klasörü


def simdi_tr() -> datetime:
    return datetime.now(TR) if TR else datetime.now()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("piyasa")


# ----------------------------------------------------------------------
# 1) VERİ TOPLAMA
# ----------------------------------------------------------------------
def canli_veri_cek(sembol: str) -> pd.DataFrame:
    import yfinance as yf  # geç import: demo modunda gerekmesin

    df = yf.Ticker(sembol).history(period=GECMIS_PERIYOT, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("boş veri döndü")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def demo_veri_uret(sembol: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(sembol)) % (2**32))
    n = 260
    getiri = rng.normal(0.0005, 0.02, n)
    kapanis = 100 * np.exp(np.cumsum(getiri))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(
        {
            "Open": kapanis, "High": kapanis * 1.01, "Low": kapanis * 0.99,
            "Close": kapanis, "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


# ----------------------------------------------------------------------
# 2) GÖSTERGELER
# ----------------------------------------------------------------------
def rsi(kapanis: pd.Series, periyot: int = 14) -> pd.Series:
    fark = kapanis.diff()
    kazanc = fark.clip(lower=0).ewm(alpha=1 / periyot, adjust=False).mean()
    kayip = (-fark.clip(upper=0)).ewm(alpha=1 / periyot, adjust=False).mean()
    rs = kazanc / kayip.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(kapanis: pd.Series):
    ema12 = kapanis.ewm(span=12, adjust=False).mean()
    ema26 = kapanis.ewm(span=26, adjust=False).mean()
    hat = ema12 - ema26
    sinyal = hat.ewm(span=9, adjust=False).mean()
    return hat, sinyal, hat - sinyal


def gostergeleri_hesapla(df: pd.DataFrame) -> dict:
    k = df["Close"]
    if len(k) < 60:
        raise ValueError(f"yetersiz veri ({len(k)} gün)")
    sma20, sma50 = k.rolling(20).mean(), k.rolling(50).mean()
    sma200 = k.rolling(200).mean()
    _, _, hist = macd(k)
    hacim_ort = df["Volume"].rolling(20).mean()
    gunluk = k.pct_change()

    def getiri(gun):
        return float(k.iloc[-1] / k.iloc[-gun - 1] - 1) if len(k) > gun else np.nan

    son = lambda s: float(s.iloc[-1]) if pd.notna(s.iloc[-1]) else np.nan
    yil_yuksek, yil_dusuk = float(k.tail(252).max()), float(k.tail(252).min())
    return {
        "fiyat": son(k),
        "gunluk_degisim": float(gunluk.iloc[-1]),
        "getiri_1h": getiri(5),
        "getiri_1a": getiri(21),
        "getiri_3a": getiri(63),
        "sma20": son(sma20), "sma50": son(sma50), "sma200": son(sma200),
        "rsi": son(rsi(k)),
        "macd_hist": son(hist),
        "macd_hist_onceki": float(hist.iloc[-2]),
        "hacim_orani": (son(df["Volume"]) / son(hacim_ort)) if son(hacim_ort) else np.nan,
        "volatilite": float(gunluk.tail(60).std() * np.sqrt(252)),
        "yil_konum": (float(k.iloc[-1]) - yil_dusuk) / (yil_yuksek - yil_dusuk)
        if yil_yuksek > yil_dusuk else np.nan,
    }


# ----------------------------------------------------------------------
# 3) SKORLAMA (kural tabanlı, şeffaf: her puanın gerekçesi listelenir)
# ----------------------------------------------------------------------
def skorla(g: dict):
    puan, nedenler = 0, []

    def ekle(p, metin):
        nonlocal puan
        puan += p
        nedenler.append((p, metin))

    f = g["fiyat"]
    # Trend
    if f > g["sma20"]: ekle(5, "Fiyat 20 günlük ortalamanın üstünde")
    else: ekle(-5, "Fiyat 20 günlük ortalamanın altında")
    if f > g["sma50"]: ekle(15, "Fiyat 50 günlük ortalamanın üstünde (orta vade trend yukarı)")
    else: ekle(-15, "Fiyat 50 günlük ortalamanın altında (orta vade trend aşağı)")
    if not np.isnan(g["sma200"]):
        if g["sma50"] > g["sma200"]: ekle(15, "50 günlük ortalama 200 günlüğün üstünde (uzun vade pozitif)")
        else: ekle(-15, "50 günlük ortalama 200 günlüğün altında (uzun vade zayıf)")
    # RSI
    r = g["rsi"]
    if r < 30: ekle(12, f"RSI {r:.0f}: aşırı satım bölgesi, tepki yükselişi ihtimali")
    elif r > 75: ekle(-15, f"RSI {r:.0f}: aşırı alım, düzeltme riski yüksek")
    elif r > 70: ekle(-8, f"RSI {r:.0f}: aşırı alıma yaklaşıyor")
    elif 50 <= r <= 70: ekle(8, f"RSI {r:.0f}: sağlıklı yükseliş momentumu")
    elif r < 45: ekle(-5, f"RSI {r:.0f}: zayıf momentum")
    # MACD
    if g["macd_hist"] > 0:
        ekle(10, "MACD histogramı pozitif")
        if g["macd_hist"] > g["macd_hist_onceki"]: ekle(5, "MACD momentumu güçleniyor")
    else:
        ekle(-10, "MACD histogramı negatif")
        if g["macd_hist"] < g["macd_hist_onceki"]: ekle(-5, "MACD momentumu zayıflıyor")
    # Kısa vadeli getiri ve hacim
    if not np.isnan(g["getiri_1a"]):
        ekle(5 if g["getiri_1a"] > 0 else -5, f"Son 1 ay getiri: %{g['getiri_1a']*100:.1f}")
    if not np.isnan(g["hacim_orani"]) and g["hacim_orani"] > 1.5:
        yon = 5 if g["gunluk_degisim"] > 0 else -5
        ekle(yon, f"Hacim ortalamanın {g['hacim_orani']:.1f} katı, hareket {'yükseliş' if yon>0 else 'düşüş'} yönlü")

    puan = max(-100, min(100, puan))
    return puan, nedenler


def etiket(puan: int) -> tuple[str, str]:
    if puan >= 35: return "Güçlü pozitif", "pos2"
    if puan >= 15: return "Pozitif", "pos"
    if puan > -15: return "Nötr", "neu"
    if puan > -35: return "Negatif", "neg"
    return "Güçlü negatif", "neg2"


# ----------------------------------------------------------------------
# 4) VERİTABANI
# ----------------------------------------------------------------------
def db_hazirla() -> sqlite3.Connection:
    con = sqlite3.connect(DB_YOLU)
    con.execute(
        """CREATE TABLE IF NOT EXISTS sinyaller (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zaman TEXT, sembol TEXT, kategori TEXT,
            fiyat REAL, skor INTEGER, etiket TEXT, rsi REAL
        )"""
    )
    return con


def db_kaydet(con, kayitlar):
    zaman = simdi_tr().isoformat(timespec="seconds")
    con.executemany(
        "INSERT INTO sinyaller (zaman,sembol,kategori,fiyat,skor,etiket,rsi) VALUES (?,?,?,?,?,?,?)",
        [(zaman, k["sembol"], k["kategori"], k["g"]["fiyat"], k["skor"], k["etiket"], k["g"]["rsi"])
         for k in kayitlar],
    )
    con.commit()


# ----------------------------------------------------------------------
# 5) HTML RAPOR
# ----------------------------------------------------------------------
def yuzde(x):
    return "-" if x is None or np.isnan(x) else f"{x*100:+.1f}%"


def satir_html(k):
    g = k["g"]
    neden = "".join(
        f"<li class='{'p' if p > 0 else 'n'}'>{html.escape(m)}</li>" for p, m in k["nedenler"]
    )
    gun_sinif = "p" if g["gunluk_degisim"] >= 0 else "n"
    return f"""
<tr>
  <td><b>{html.escape(k['sembol'])}</b><br><small>{html.escape(k['ad'])}</small></td>
  <td class="num">{g['fiyat']:,.2f}</td>
  <td class="num {gun_sinif}">{yuzde(g['gunluk_degisim'])}</td>
  <td class="num">{yuzde(g['getiri_1a'])}</td>
  <td class="num">{g['rsi']:.0f}</td>
  <td class="num">{k['skor']:+d}</td>
  <td><span class="rozet {k['sinif']}">{k['etiket']}</span>
      <details><summary>Gerekçe</summary><ul>{neden}</ul></details></td>
</tr>"""


def rapor_olustur(kayitlar, hatalar, demo) -> str:
    simdi = simdi_tr().strftime("%d.%m.%Y %H:%M") + " (TSİ)"
    siralı = sorted(kayitlar, key=lambda k: k["skor"], reverse=True)
    en_iyi, en_kotu = siralı[:5], siralı[::-1][:5]
    ort = np.mean([k["skor"] for k in kayitlar]) if kayitlar else 0
    ruh = "olumlu" if ort >= 10 else "olumsuz" if ort <= -10 else "karışık"

    def kart(baslik, liste):
        oge = "".join(
            f"<li><b>{html.escape(k['sembol'])}</b> {html.escape(k['ad'])} "
            f"<span class='rozet {k['sinif']}'>{k['skor']:+d}</span></li>" for k in liste
        )
        return f"<div class='kart'><h3>{baslik}</h3><ul class='kisa'>{oge}</ul></div>"

    bolumler = ""
    for kat in VARLIKLAR:
        ks = [k for k in siralı if k["kategori"] == kat]
        if ks:
            bolumler += f"<h2>{kat}</h2><div class='tablo-kap'><table><thead><tr><th>Varlık</th><th>Fiyat</th><th>Gün</th><th>1 Ay</th><th>RSI</th><th>Skor</th><th>Durum</th></tr></thead><tbody>{''.join(satir_html(k) for k in ks)}</tbody></table></div>"

    hata_html = ""
    if hatalar:
        hata_html = "<div class='uyari'><b>Veri alınamayan varlıklar:</b> " + ", ".join(
            f"{html.escape(s)} ({html.escape(e)})" for s, e in hatalar) + "</div>"
    demo_html = "<div class='uyari'><b>DEMO MODU:</b> Veriler sentetiktir, gerçek piyasayı yansıtmaz.</div>" if demo else ""

    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Piyasa Raporu - {simdi}</title>
<style>
:root{{--bg:#f6f7f9;--kart:#fff;--yazi:#1a1d23;--soluk:#6b7280;--cizgi:#e5e7eb;--yesil:#0f7b3f;--kirmizi:#b42318;--sari:#8a6100}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1115;--kart:#181b21;--yazi:#e8eaed;--soluk:#9aa0a6;--cizgi:#2a2e36;--yesil:#4ade80;--kirmizi:#f87171;--sari:#fbbf24}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--yazi);font:15px/1.5 system-ui,sans-serif}}
.kap{{max-width:1100px;margin:auto}}
h1{{margin:0 0 4px}} h2{{margin:28px 0 8px}} .alt{{color:var(--soluk)}}
.izgara{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:16px}}
.kart{{background:var(--kart);border:1px solid var(--cizgi);border-radius:10px;padding:14px}}
.kart h3{{margin:0 0 8px;font-size:15px}} ul.kisa{{list-style:none;padding:0;margin:0}} ul.kisa li{{padding:4px 0;display:flex;justify-content:space-between;gap:8px}}
.tablo-kap{{overflow-x:auto;background:var(--kart);border:1px solid var(--cizgi);border-radius:10px}}
table{{border-collapse:collapse;width:100%;min-width:640px}} th,td{{padding:8px 10px;border-bottom:1px solid var(--cizgi);text-align:left;vertical-align:top}}
th{{font-size:12px;color:var(--soluk);text-transform:uppercase}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.p{{color:var(--yesil)}} .n{{color:var(--kirmizi)}}
.rozet{{padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid currentColor;white-space:nowrap}}
.pos2,.pos{{color:var(--yesil)}} .neg2,.neg{{color:var(--kirmizi)}} .neu{{color:var(--sari)}}
details{{margin-top:4px;font-size:13px}} summary{{cursor:pointer;color:var(--soluk)}} details ul{{padding-left:18px;margin:4px 0}}
.uyari{{background:var(--kart);border:1px solid var(--sari);border-radius:10px;padding:10px 14px;margin-top:12px}}
footer{{margin:32px 0 8px;color:var(--soluk);font-size:13px}}
</style></head><body><div class="kap">
<h1>Günlük Piyasa Raporu</h1>
<div class="alt">{simdi} · {len(kayitlar)} varlık analiz edildi · Genel görünüm: <b>{ruh}</b> (ort. skor {ort:+.0f})</div>
{demo_html}{hata_html}
<div class="izgara">{kart("En yüksek skorlular", en_iyi)}{kart("En düşük skorlular", en_kotu)}</div>
{bolumler}
<footer><b>Önemli:</b> Skorlar yalnızca geçmiş fiyat ve hacim verisine dayalı teknik göstergelerden üretilir; haber ve temel analiz henüz dahil değildir.
Bu rapor yatırım tavsiyesi değildir, getiri garantisi vermez. Kararlar size aittir.</footer>
</div></body></html>"""


# ----------------------------------------------------------------------
# ANA AKIŞ
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="sentetik veriyle çalış")
    args = ap.parse_args()

    kayitlar, hatalar = [], []
    for kategori, semboller in VARLIKLAR.items():
        for sembol, ad in semboller.items():
            try:
                df = demo_veri_uret(sembol) if args.demo else canli_veri_cek(sembol)
                g = gostergeleri_hesapla(df)
                skor, nedenler = skorla(g)
                et, sinif = etiket(skor)
                kayitlar.append(dict(sembol=sembol, ad=ad, kategori=kategori, g=g,
                                     skor=skor, nedenler=nedenler, etiket=et, sinif=sinif))
                log.info("%-10s skor %+d (%s)", sembol, skor, et)
            except Exception as e:  # bir varlık çökerse diğerleri devam eder
                log.warning("%s atlandı: %s", sembol, e)
                hatalar.append((sembol, str(e)))

    if not kayitlar:
        raise SystemExit("Hiç veri alınamadı. İnternet bağlantını ve yfinance kurulumunu kontrol et.")

    if not args.demo:
        con = db_hazirla()
        db_kaydet(con, kayitlar)
        con.close()

    CIKTI_KLASORU.mkdir(exist_ok=True)
    icerik = rapor_olustur(kayitlar, hatalar, args.demo)
    dosya = CIKTI_KLASORU / f"rapor_{simdi_tr():%Y-%m-%d_%H%M}.html"
    dosya.write_text(icerik, encoding="utf-8")
    log.info("Rapor yazıldı: %s", dosya)

    # Her zaman en güncel rapor: site/index.html (telefondaki link bunu gösterir)
    SITE_KLASORU.mkdir(exist_ok=True)
    (SITE_KLASORU / "index.html").write_text(icerik, encoding="utf-8")


if __name__ == "__main__":
    main()
