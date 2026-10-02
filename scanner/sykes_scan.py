"""
Sykes-style pre-market watchlist scanner.

Runs ~1 hour before the US open (on GitHub Actions) and writes:
  data/latest.json, data/YYYY-MM-DD.json, data/index.json

Rules (set by Husain):
  * US-listed (NASDAQ / NYSE / AMEX), price $1-$10, float < 20M shares
  * Fresh news catalyst required
  * Relative volume >= 5x (see rel_volume() for how pre-market RVOL is measured)
  * Setups: Supernova gapper, Morning-panic dip-buy (ran 50%+ in last 1-5 days),
    Breakout / multi-day runner
  * Long-side trade plans only
  * Red flags (dilution filings, reverse splits, recent foreign IPOs, SPACs/warrants)
    are FLAGGED, not excluded
  * Top 10 by score
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SEC_UA = "SykesWatchlist research h-4@outlook.com"

# ---------------------------------------------------------------- rules
RULES = {
    "price_min": 1.0,
    "price_max": 10.0,
    "float_max": 20_000_000,
    "rvol_min": 5.0,
    "gap_min_pct": 20.0,          # supernova gapper
    "runner_min_pct": 50.0,       # dip-buy: ran >= 50% ...
    "runner_lookback_days": 5,    # ... within the last 1-5 sessions
    "news_max_age_hours": 36,     # fresh catalyst for gappers
    "news_max_age_hours_runner": 120,  # catalyst for multi-day / dip setups
    "max_picks": 10,
    "pm_volume_share": 0.10,      # normal pre-market volume ~10% of a full day
}

# NYSE full-day holidays
HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})


def log(*a):
    print(datetime.now(ET).strftime("%H:%M:%S"), *a, flush=True)


def num(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").replace("%", "").strip()
            if x in ("", "NA", "N/A", "-"):
                return None
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None


def r2(x, n=2):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- universe
def universe_nasdaq() -> list[dict]:
    """All US-listed common stocks from Nasdaq's public screener."""
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
    r = S.get(url, timeout=40, headers={"Origin": "https://www.nasdaq.com",
                                         "Referer": "https://www.nasdaq.com/"})
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    out = []
    for row in rows:
        sym = (row.get("symbol") or "").strip().upper()
        if not sym or "^" in sym or "/" in sym:
            continue
        out.append({
            "symbol": sym.replace(".", "-"),
            "name": row.get("name") or "",
            "last": num(row.get("lastsale")),
            "country": row.get("country") or "",
            "ipo_year": num(row.get("ipoyear")),
            "industry": row.get("industry") or "",
            "sector": row.get("sector") or "",
            "volume": num(row.get("volume")),
        })
    return out


def universe_finviz() -> list[dict]:
    """Fallback: scrape Finviz free screener (price 1-10ish, float<20M)."""
    out, start = [], 1
    while start < 3000:
        url = ("https://finviz.com/screener.ashx?v=111&f=sh_float_u20,sh_price_u12"
               f"&r={start}")
        html = S.get(url, timeout=30).text
        syms = re.findall(r'quote\.ashx\?t=([A-Z\-\.]+)&', html)
        syms = list(dict.fromkeys(syms))
        if not syms:
            break
        new = [s for s in syms if s not in {o["symbol"] for o in out}]
        if not new:
            break
        out += [{"symbol": s, "name": "", "last": None, "country": "", "ipo_year": None,
                 "industry": "", "sector": "", "volume": None} for s in new]
        start += 20
        time.sleep(0.6)
    return out


def get_universe() -> list[dict]:
    try:
        u = universe_nasdaq()
        log(f"universe: nasdaq {len(u)}")
        if len(u) > 1000:
            return u
    except Exception as e:
        log("nasdaq universe failed:", e)
    u = universe_finviz()
    log(f"universe: finviz {len(u)}")
    return u


SPAC_RE = re.compile(r"acquisition corp|acquisition co\b|blank check", re.I)
WARRANT_RE = re.compile(r"\bwarrants?\b|\bunits?\b|\brights?\b", re.I)


def instrument_flags(u: dict) -> list[str]:
    f = []
    sym, name = u["symbol"], u["name"]
    if SPAC_RE.search(name) or "blank check" in u.get("industry", "").lower():
        f.append("SPAC")
    if WARRANT_RE.search(name) or (len(sym) == 5 and sym[-1] in "WUR"):
        f.append("Warrant/Unit/Right")
    if re.search(r"\bETF\b|\bETN\b|\bFund\b|\bTrust\b", name):
        f.append("ETF/Fund")
    return f


# ---------------------------------------------------------------- price data
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def daily_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for batch in chunks(symbols, 200):
        for attempt in range(3):
            try:
                df = yf.download(batch, period="3mo", interval="1d", group_by="ticker",
                                 auto_adjust=False, threads=True, progress=False)
                break
            except Exception as e:
                log("daily download retry", e)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for s in batch:
            try:
                d = df[s] if isinstance(df.columns, pd.MultiIndex) else df
                d = d.dropna(subset=["Close"])
                if len(d) >= 10:
                    out[s] = d
            except Exception:
                pass
        time.sleep(1)
    return out


def premarket(symbols: list[str], today) -> dict[str, dict]:
    """Pre-market last / high / low / volume / VWAP for today (04:00-09:30 ET)."""
    out = {}
    for batch in chunks(symbols, 80):
        try:
            df = yf.download(batch, period="2d", interval="1m", prepost=True,
                             group_by="ticker", auto_adjust=False, threads=True,
                             progress=False)
        except Exception as e:
            log("pm download failed", e)
            continue
        for s in batch:
            try:
                d = df[s] if isinstance(df.columns, pd.MultiIndex) else df
                d = d.dropna(subset=["Close"])
                if d.empty:
                    continue
                idx = d.index.tz_convert(ET) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(ET)
                d = d.set_axis(idx)
                pm = d[(d.index.date == today) & (d.index.time < datetime.strptime("09:30", "%H:%M").time())]
                if pm.empty:
                    continue
                vol = float(pm["Volume"].sum())
                tp = (pm["High"] + pm["Low"] + pm["Close"]) / 3
                vwap = float((tp * pm["Volume"]).sum() / vol) if vol > 0 else float(pm["Close"].iloc[-1])
                out[s] = {"last": float(pm["Close"].iloc[-1]), "high": float(pm["High"].max()),
                          "low": float(pm["Low"].min()), "volume": vol, "vwap": vwap}
            except Exception:
                pass
        time.sleep(1)
    return out


# ---------------------------------------------------------------- per-ticker detail
def yahoo_detail(sym: str) -> dict:
    t = yf.Ticker(sym)
    info, news, splits = {}, [], None
    try:
        info = t.get_info() or {}
    except Exception:
        pass
    try:
        news = t.get_news(count=15) or []
    except Exception:
        try:
            news = t.news or []
        except Exception:
            news = []
    try:
        splits = t.splits
    except Exception:
        pass
    return {"info": info, "news": news, "splits": splits}


def parse_news(raw: list) -> list[dict]:
    items = []
    for n in raw:
        c = n.get("content", n)
        title = c.get("title")
        ts = c.get("pubDate") or c.get("displayTime") or n.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
        else:
            continue
        link = None
        for key in ("clickThroughUrl", "canonicalUrl"):
            v = c.get(key)
            if isinstance(v, dict) and v.get("url"):
                link = v["url"]
                break
        link = link or n.get("link")
        prov = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else n.get("publisher")
        if title:
            items.append({"title": title, "url": link, "source": prov, "time": dt.isoformat()})
    items.sort(key=lambda x: x["time"], reverse=True)
    return items


def finviz_quote(sym: str) -> dict:
    """Float + news headlines from the Finviz quote page (fallback / supplement)."""
    out = {"float": None, "news": [], "status": None}
    try:
        r = S.get(f"https://finviz.com/quote.ashx?t={sym.replace('-', '.')}&p=d", timeout=20)
        out["status"] = r.status_code
        html = r.text
        m = re.search(r"Shs Float.{0,400}?>\s*([\d\.]+)\s*([KMB])\s*<", html, re.S)
        if m:
            mult = {"K": 1e3, "M": 1e6, "B": 1e9}[m.group(2)]
            out["float"] = float(m.group(1)) * mult
        tm_ = re.search(r'id="news-table"(.*?)</table>', html, re.S)
        if not tm_:
            return out
        now = datetime.now(ET)
        cur_date = None
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tm_.group(1), re.S)[:20]:
            td = re.search(r"<td[^>]*>(.*?)</td>", row, re.S)
            a = re.search(r"<a\s+([^>]*tab-link-news[^>]*)>(.*?)</a>", row, re.S)
            if not (td and a):
                continue
            stamp = re.sub(r"<[^>]+>", " ", td.group(1)).split()
            try:
                if len(stamp) >= 2:
                    d = now.date() if stamp[0].lower() == "today" else datetime.strptime(stamp[0], "%b-%d-%y").date()
                    cur_date, tm = d, stamp[1]
                else:
                    tm = stamp[0]
                if cur_date is None:
                    cur_date = now.date()
                dt = datetime.strptime(f"{cur_date} {tm}", "%Y-%m-%d %I:%M%p").replace(tzinfo=ET)
            except Exception:
                continue
            href = re.search(r'href="([^"]+)"', a.group(1))
            url = href.group(1) if href else None
            if url and url.startswith("/"):
                url = "https://finviz.com" + url
            title = re.sub(r"<[^>]+>", "", a.group(2)).strip()
            out["news"].append({"title": title, "url": url, "source": "Finviz",
                                "time": dt.astimezone(timezone.utc).isoformat()})
    except Exception as e:
        log("finviz quote fail", sym, e)
    return out


def yahoo_search_news(sym: str) -> list[dict]:
    """Headlines from Yahoo's public search endpoint (independent of yfinance)."""
    items = []
    for host in ("query2", "query1"):
        try:
            r = S.get(f"https://{host}.finance.yahoo.com/v1/finance/search",
                      params={"q": sym, "newsCount": 10, "quotesCount": 0}, timeout=15)
            if r.status_code != 200:
                continue
            for n in r.json().get("news", []):
                ts = n.get("providerPublishTime")
                if not (ts and n.get("title")):
                    continue
                if n.get("relatedTickers") and sym not in n["relatedTickers"]:
                    continue
                items.append({"title": n["title"], "url": n.get("link"), "source": n.get("publisher"),
                              "time": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()})
            break
        except Exception:
            continue
    return items


_CIK = None


def sec_cik_map() -> dict:
    global _CIK
    if _CIK is None:
        try:
            r = S.get("https://www.sec.gov/files/company_tickers.json", timeout=30,
                      headers={"User-Agent": SEC_UA})
            _CIK = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}
        except Exception as e:
            log("sec map fail", e)
            _CIK = {}
    return _CIK


DILUTION_FORMS = {"S-1", "S-1/A", "S-3", "S-3/A", "F-1", "F-1/A", "F-3", "F-3/A", "424B1",
                  "424B2", "424B3", "424B4", "424B5", "424B7", "S-3ASR", "EFFECT"}


def sec_flags(sym: str, today) -> list[str]:
    cik = sec_cik_map().get(sym)
    if not cik:
        return []
    try:
        r = S.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", timeout=20,
                  headers={"User-Agent": SEC_UA})
        rec = r.json()["filings"]["recent"]
    except Exception:
        return []
    flags, seen = [], set()
    cutoff = today - timedelta(days=120)
    for form, date in zip(rec.get("form", []), rec.get("filingDate", [])):
        try:
            d = datetime.strptime(date, "%Y-%m-%d").date()
        except Exception:
            continue
        if d < cutoff:
            break
        if form in DILUTION_FORMS and form not in seen:
            seen.add(form)
            kind = "Offering" if form.startswith("424B") else "Shelf/Registration"
            flags.append(f"{kind} {form} filed {date}")
    return flags[:4]


# ---------------------------------------------------------------- setups
def rel_volume(pm_vol: float | None, avg_vol: float, yday_vol: float) -> tuple[float, str]:
    """
    Pre-market relative volume = pre-market volume vs. a normal pre-market
    (assumed to be ~10% of the 20-day average daily volume).
    For setups judged on yesterday's action we also look at yesterday's RVOL.
    Returns the larger of the two with a label.
    """
    pm_rvol = (pm_vol or 0) / max(avg_vol * RULES["pm_volume_share"], 1)
    yd_rvol = yday_vol / max(avg_vol, 1)
    if pm_rvol >= yd_rvol:
        return pm_rvol, "pre-market"
    return yd_rvol, "yesterday"


def classify(sym, d: pd.DataFrame, pm: dict | None, today):
    d = d[d.index.date < today] if hasattr(d.index, "date") else d
    if len(d) < 22:
        return None
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    prev_close = float(c.iloc[-1])
    prev_high, prev_low = float(h.iloc[-1]), float(l.iloc[-1])
    avg_vol = float(v.iloc[-21:-1].mean()) or 1.0
    yday_vol = float(v.iloc[-1])
    last = pm["last"] if pm else prev_close
    gap = (last / prev_close - 1) * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.iloc[-14:].mean())

    # run-up in last 1-5 sessions: max high in window vs lowest close before the run
    lb = RULES["runner_lookback_days"]
    win = d.iloc[-lb:]
    base = float(c.iloc[-lb - 1:-lb].iloc[0]) if len(c) > lb else float(c.iloc[0])
    base = min(base, float(win["Low"].min()))
    peak = float(win["High"].max())
    run_pct = (peak / base - 1) * 100 if base > 0 else 0
    off_peak = (last / peak - 1) * 100 if peak else 0

    hi20 = float(h.iloc[-21:-1].max())
    hi5 = float(h.iloc[-5:].max())
    green_streak = 0
    for i in range(1, 6):
        if c.iloc[-i] > c.iloc[-i - 1]:
            green_streak += 1
        else:
            break

    rvol, rvol_basis = rel_volume(pm["volume"] if pm else 0, avg_vol, yday_vol)

    setups = []
    if gap >= RULES["gap_min_pct"]:
        setups.append("Supernova gapper")
    if run_pct >= RULES["runner_min_pct"] and off_peak <= -10:
        setups.append("Morning-panic dip-buy")
    if (prev_close >= hi20 * 0.98 or green_streak >= 2) and gap >= 0 and yday_vol >= 3 * avg_vol:
        setups.append("Breakout / multi-day runner")
    if not setups:
        return None

    return {
        "symbol": sym, "price": r2(last, 3), "prev_close": r2(prev_close, 3), "gap_pct": r2(gap, 1),
        "pm": {k: r2(vv, 3) for k, vv in pm.items() if k != "volume"} | {"volume": int(pm["volume"])} if pm else None,
        "avg_vol_20d": int(avg_vol), "yday_vol": int(yday_vol), "rvol": r2(rvol, 1), "rvol_basis": rvol_basis,
        "run_pct_5d": r2(run_pct, 1), "off_peak_pct": r2(off_peak, 1), "atr14": r2(atr, 3),
        "levels": {"prev_high": r2(prev_high, 3), "prev_low": r2(prev_low, 3), "prev_close": r2(prev_close, 3),
                   "high_5d": r2(hi5, 3), "high_20d": r2(hi20, 3), "run_peak": r2(peak, 3),
                   "pm_high": r2(pm["high"], 3) if pm else None, "pm_low": r2(pm["low"], 3) if pm else None,
                   "pm_vwap": r2(pm["vwap"], 3) if pm else None},
        "green_streak": green_streak, "setups": setups,
    }


def trade_plan(c: dict) -> dict:
    """Long-only plans, Sykes-style: small risk, quick cut, ~2:1 or better."""
    L, price, setup = c["levels"], c["price"], c["setups"][0]
    if setup == "Supernova gapper":
        entry = L["pm_high"] or price
        stop = max(L["pm_vwap"] or entry * 0.9, entry * 0.9)
        if stop >= entry:
            stop = entry * 0.93
        note = ("Wait for the open. Buy only on a clean break of the pre-market high with heavy volume; "
                "skip if it fades below pre-market VWAP. Sykes rule: take profits into the spike, cut fast.")
    elif setup == "Morning-panic dip-buy":
        peak = L["run_peak"]
        zone_hi = min(price, peak * 0.75)
        zone_lo = min(peak * 0.65, zone_hi * 0.92)
        entry = (zone_hi + zone_lo) / 2
        stop = zone_lo * 0.95
        note = (f"Former runner (+{c['run_pct_5d']}% recently). Wait for an early panic flush into "
                f"{zone_lo:.2f}-{zone_hi:.2f} ({(1 - zone_hi / peak) * 100:.0f}-{(1 - zone_lo / peak) * 100:.0f}% "
                f"off the {peak:.2f} peak), then buy the first "
                "strong bounce/reversal candle. No bounce = no trade.")
    else:  # breakout / multi-day runner
        entry = max(L["prev_high"], L["pm_high"] or 0)
        stop = max(L["prev_close"] * 0.97, entry * 0.92)
        if stop >= entry:
            stop = entry * 0.93
        note = ("Multi-day runner. Buy the break of yesterday's / pre-market high if it holds on volume; "
                "day 2-3 is the sweet spot, be cautious past day 3.")
    risk = entry - stop
    target = entry + 2 * risk
    if setup == "Morning-panic dip-buy":
        target = max(target, L["prev_close"])
    return {"entry": r2(entry, 3), "stop": r2(stop, 3), "target": r2(target, 3),
            "risk_pct": r2(risk / entry * 100, 1) if entry else None,
            "rr": r2((target - entry) / risk, 1) if risk > 0 else None, "note": note}


def score(c: dict) -> float:
    s = 0.0
    s += min(abs(c["gap_pct"] or 0), 150) * 0.4
    s += min(c["rvol"] or 0, 50) * 1.2
    if c.get("float"):
        s += (20e6 - min(c["float"], 20e6)) / 1e6 * 1.0
    if c.get("news_age_h") is not None:
        s += max(0, 36 - c["news_age_h"]) * 0.5
    s += len(c["setups"]) * 4
    s -= len(c.get("flags", [])) * 3
    return round(s, 1)


# ---------------------------------------------------------------- main
def main():
    now = datetime.now(ET)
    today = now.date()
    force = os.environ.get("FORCE_RUN") == "1"
    DATA.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SCHEDULED") == "1":
        # two UTC crons cover EDT and EST; only the one landing 07:45-08:59 ET runs
        mins = now.hour * 60 + now.minute
        if not (7 * 60 + 45 <= mins < 9 * 60):
            log(f"outside pre-market window ({now:%H:%M} ET), skipping")
            return
    if (today.weekday() >= 5 or today.isoformat() in HOLIDAYS) and not force:
        log("market closed today, nothing to do")
        write_output({"date": today.isoformat(), "generated_at": now.isoformat(), "market_closed": True,
                      "picks": [], "near_misses": [], "rules": RULES, "stats": {}})
        return

    uni = get_universe()
    by_sym = {u["symbol"]: u for u in uni}
    # coarse price pre-filter on last sale (wide band to catch pre-market gaps)
    pre = [u["symbol"] for u in uni
           if (u["last"] is None or 0.4 <= u["last"] <= 12) and "ETF/Fund" not in instrument_flags(u)]
    log("price pre-filter", len(pre))

    hist = daily_history(pre)
    log("daily history", len(hist))

    # second filter: needs some trading interest, then pre-market for the survivors
    stage2 = []
    for s, d in hist.items():
        d0 = d[d.index.date < today]
        if len(d0) < 22:
            continue
        pc = float(d0["Close"].iloc[-1])
        if not (0.5 <= pc <= 12):
            continue
        av = float(d0["Volume"].iloc[-21:-1].mean())
        yv = float(d0["Volume"].iloc[-1])
        rng5 = float(d0["High"].iloc[-5:].max()) / max(float(d0["Low"].iloc[-6:].min()), 1e-6)
        if av >= 20_000 or yv >= 200_000 or rng5 >= 1.4:
            stage2.append(s)
    log("stage2", len(stage2))
    pm = premarket(stage2, today)
    log("premarket bars", len(pm))

    cands = []
    for s in stage2:
        try:
            c = classify(s, hist[s], pm.get(s), today)
        except Exception:
            continue
        if not c:
            continue
        if not (RULES["price_min"] <= (c["price"] or 0) <= RULES["price_max"]):
            continue
        cands.append(c)
    log("setup candidates", len(cands))
    # keep the most active ones for the expensive per-ticker checks
    cands.sort(key=lambda c: (c["rvol"] or 0) + abs(c["gap_pct"] or 0) / 5, reverse=True)
    cands = cands[:45]

    def enrich(c):
        s = c["symbol"]
        u = by_sym.get(s, {})
        yd = yahoo_detail(s)
        info = yd["info"]
        fl = num(info.get("floatShares"))
        yn = parse_news(yd["news"])
        ys = yahoo_search_news(s)
        fv = finviz_quote(s)
        if fl is None:
            fl = fv["float"]
        c["sources"] = {"yf_raw": len(yd["news"]), "yf": len(yn), "yahoo_search": len(ys),
                        "finviz": len(fv["news"]), "finviz_http": fv["status"]}
        news = sorted({n["title"]: n for n in yn + ys + fv["news"]}.values(), key=lambda x: x["time"], reverse=True)
        c["name"] = info.get("shortName") or u.get("name") or s
        c["sector"] = info.get("sector") or u.get("sector")
        c["industry"] = info.get("industry") or u.get("industry")
        c["country"] = info.get("country") or u.get("country")
        c["exchange"] = info.get("exchange")
        c["float"] = fl
        c["short_pct_float"] = r2((num(info.get("shortPercentOfFloat")) or 0) * 100, 1) if info.get("shortPercentOfFloat") else None
        c["news"] = news[:4]
        if news:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(news[0]["time"])).total_seconds() / 3600
            c["news_age_h"] = round(age, 1)
        else:
            c["news_age_h"] = None
        flags = instrument_flags(u) if u else []
        flags += sec_flags(s, today)
        sp = yd["splits"]
        try:
            if sp is not None and len(sp):
                rec = sp[sp.index >= pd.Timestamp(today - timedelta(days=365), tz=sp.index.tz)]
                for dt, ratio in rec.items():
                    if ratio < 1:
                        flags.append(f"Reverse split 1:{round(1 / ratio)} on {dt.date()}")
        except Exception:
            pass
        ipo_year = u.get("ipo_year")
        first_trade = num(info.get("firstTradeDateEpochUtc") or info.get("firstTradeDateMilliseconds"))
        recent_ipo = False
        if first_trade:
            ft = first_trade / 1000 if first_trade > 1e11 else first_trade
            recent_ipo = (datetime.now(timezone.utc) - datetime.fromtimestamp(ft, timezone.utc)).days < 365
        elif ipo_year:
            recent_ipo = ipo_year >= today.year - 1
        if recent_ipo and (c["country"] or "United States") not in ("United States", "USA"):
            flags.append(f"Recent foreign IPO ({c['country']})")
        c["flags"] = flags
        return c

    with ThreadPoolExecutor(max_workers=6) as ex:
        enriched = list(ex.map(lambda c: _safe(enrich, c), cands))
    enriched = [c for c in enriched if c]

    picks, misses = [], []
    for c in enriched:
        reasons = []
        if c["float"] is None:
            reasons.append("float unknown")
        elif c["float"] > RULES["float_max"]:
            reasons.append(f"float {c['float'] / 1e6:.1f}M > 20M")
        max_age = RULES["news_max_age_hours"] if c["setups"][0] == "Supernova gapper" else RULES["news_max_age_hours_runner"]
        if c["news_age_h"] is None or c["news_age_h"] > max_age:
            reasons.append("no fresh news")
        if (c["rvol"] or 0) < RULES["rvol_min"]:
            reasons.append(f"RVOL {c['rvol']}x < 5x")
        c["plan"] = trade_plan(c)
        c["score"] = score(c)
        if reasons:
            c["failed"] = reasons
            misses.append(c)
        else:
            picks.append(c)
    picks.sort(key=lambda c: c["score"], reverse=True)
    misses = [m for m in misses if len(m["failed"]) == 1]
    misses.sort(key=lambda c: c["score"], reverse=True)

    result = {
        "date": today.isoformat(), "generated_at": datetime.now(ET).isoformat(), "market_closed": False,
        "rules": RULES, "picks": picks[:RULES["max_picks"]], "near_misses": misses[:5],
        "stats": {"universe": len(uni), "with_history": len(hist), "scanned_premarket": len(stage2),
                  "setup_candidates": len(cands), "qualified": len(picks),
                  "news_sources": {k: sum(1 for c in enriched if (c.get("sources") or {}).get(k))
                                   for k in ("yf_raw", "yf", "yahoo_search", "finviz")},
                  "finviz_http": sorted({str((c.get("sources") or {}).get("finviz_http")) for c in enriched})},
    }
    write_output(result)
    log(f"done: {len(picks)} picks, {len(misses)} near misses")


def _safe(fn, c):
    try:
        return fn(c)
    except Exception:
        traceback.print_exc()
        return None


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def write_output(result: dict):
    result = _clean(result)
    (DATA / "latest.json").write_text(json.dumps(result, indent=1, default=str))
    (DATA / f"{result['date']}.json").write_text(json.dumps(result, indent=1, default=str))
    idx_p = DATA / "index.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else []
    idx = [i for i in idx if i["date"] != result["date"]]
    idx.append({"date": result["date"], "count": len(result["picks"]),
                "tickers": [p["symbol"] for p in result["picks"]], "market_closed": result["market_closed"]})
    idx.sort(key=lambda i: i["date"], reverse=True)
    idx_p.write_text(json.dumps(idx, indent=1))


if __name__ == "__main__":
    main()
