import os
import time
import logging
import asyncio
import json
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import websockets
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("delta-ai-bot")

BASE_URL = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange").rstrip("/")
# Public Delta API calls can be briefly slow; retry transient GET failures.
TIMEOUT = 30
SESSION = requests.Session()
_RETRY = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
    raise_on_status=False,
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY, pool_connections=10, pool_maxsize=10)
SESSION.mount("https://", _ADAPTER)
SESSION.mount("http://", _ADAPTER)
SESSION.headers.update({"Accept": "application/json", "User-Agent": "Delta-AI-Trading-Bot-V2/1.0"})

def delta_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        response = SESSION.get(f"{BASE_URL}{path}", params=params or {}, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("success") is False:
            raise ValueError(str(payload))
        return payload.get("result", payload) if isinstance(payload, dict) else payload
    except (requests.RequestException, ValueError) as exc:
        log.exception("Delta API request failed: %s", path)
        raise HTTPException(status_code=502, detail=f"Delta API error: {exc}")

app = FastAPI(
    title="Delta AI Trading Bot V2",
    description="BTC/ETH market-data research, options analytics and paper-only signals. No live order execution.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("backtest_dashboard.html", "r", encoding="utf-8") as dashboard:
            return dashboard.read()
    except OSError:
        return HTMLResponse("<h1>Backtest dashboard file not found</h1>", status_code=500)

@app.get("/health")
def health():
    return {"status": "ok", "service": "delta-ai-trading-bot-v2", "mode": "PAPER_ONLY"}

@app.get("/market/tickers")
def market_tickers(asset: str = Query("BTC", pattern="^(BTC|ETH|XAUT)$")):
    return delta_get("/v2/tickers", {
        "underlying_asset_symbols": asset,
        "contract_types": "perpetual_futures,futures"
    })

@app.get("/market/options")
def options_chain(
    asset: str = Query("BTC", pattern="^(BTC|ETH|XAUT)$"),
    expiry_date: Optional[str] = Query(None, description="DD-MM-YYYY; omit to request current/future listed expiries")
):
    params = {
        "underlying_asset_symbols": asset,
        "contract_types": "call_options,put_options"
    }
    if expiry_date:
        params["expiry_date"] = expiry_date
    return delta_get("/v2/tickers", params)

@app.get("/market/candles")
def candles(
    asset: str = Query("BTC", pattern="^(BTC|ETH|XAUT)$"),
    resolution: str = Query("1h", pattern="^(1m|3m|5m|15m|30m|1h|2h|4h|6h|1d|1w)$"),
    days: int = Query(180, ge=1, le=180)
):
    symbol = f"{asset}USD"
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    # Delta returns at most 2,000 candles per request. Split the requested range safely.
    seconds = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,"4h":14400,"6h":21600,"1d":86400,"1w":604800}[resolution]
    chunk_seconds = seconds * 1900
    all_rows: List[dict] = []
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + chunk_seconds, end_ts)
        rows = delta_get("/v2/history/candles", {
            "resolution": resolution,
            "symbol": symbol,
            "start": cursor,
            "end": chunk_end
        })
        if isinstance(rows, list):
            all_rows.extend(rows)
        cursor = chunk_end
        if len(all_rows) > 50000:
            break
    unique = {row.get("time"): row for row in all_rows if isinstance(row, dict) and row.get("time") is not None}
    result = sorted(unique.values(), key=lambda x: x.get("time", 0))
    return {"symbol": symbol, "resolution": resolution, "days_requested": days, "count": len(result), "candles": result}

def number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0

@app.get("/analytics/options")
def options_analytics(
    asset: str = Query("BTC", pattern="^(BTC|ETH|XAUT)$"),
    expiry_date: Optional[str] = None
):
    rows = options_chain(asset, expiry_date)
    if not isinstance(rows, list):
        rows = rows.get("result", []) if isinstance(rows, dict) else []
    calls = [x for x in rows if "call" in str(x.get("contract_type", "")).lower()]
    puts = [x for x in rows if "put" in str(x.get("contract_type", "")).lower()]
    def oi(x):
        return number(x.get("open_interest") or x.get("oi") or x.get("openInterest"))
    call_oi = sum(oi(x) for x in calls)
    put_oi = sum(oi(x) for x in puts)
    pcr = (put_oi / call_oi) if call_oi else None
    strikes = sorted({number(x.get("strike_price") or x.get("strike")) for x in rows if number(x.get("strike_price") or x.get("strike")) > 0})
    max_pain = None
    if strikes and calls and puts:
        payouts = []
        for settlement in strikes:
            total = 0.0
            for option in calls:
                total += max(0.0, settlement - number(option.get("strike_price") or option.get("strike"))) * oi(option)
            for option in puts:
                total += max(0.0, number(option.get("strike_price") or option.get("strike")) - settlement) * oi(option)
            payouts.append((total, settlement))
        max_pain = min(payouts)[1]
    return {
        "asset": asset,
        "expiry_date": expiry_date,
        "contracts": len(rows),
        "call_contracts": len(calls),
        "put_contracts": len(puts),
        "call_open_interest": call_oi,
        "put_open_interest": put_oi,
        "pcr_by_open_interest": pcr,
        "max_pain_estimate": max_pain,
        "note": "Only available exchange OI/strike fields are used. Missing fields are not fabricated; Greeks/IV remain in raw chain response."
    }


@app.get("/backtest/options/{asset}")
def options_contract_backtest(
    asset: str,
    symbol: str = Query(..., min_length=5, max_length=80),
    resolution: str = Query("1h", pattern="^(1m|3m|5m|15m|30m|1h|2h|4h|6h|1d|1w)$"),
    days: int = Query(30, ge=1, le=180),
    taker_fee_pct: float = Query(0.05, ge=0, le=1),
    gst_pct: float = Query(18.0, ge=0, le=100),
    slippage_pct: float = Query(0.02, ge=0, le=1)
):
    """Backtest a selected currently listed option contract using its own historical candles; PAPER_ONLY."""
    asset = asset.upper()
    if asset not in ("BTC", "ETH", "XAUT"):
        raise HTTPException(status_code=400, detail="asset must be BTC, ETH or XAUT")
    chain = options_chain(asset)
    if not isinstance(chain, list):
        chain = chain.get("result", []) if isinstance(chain, dict) else []
    allowed = {str(row.get("symbol", "")) for row in chain if isinstance(row, dict)}
    if symbol not in allowed:
        raise HTTPException(status_code=400, detail="Choose a symbol from the current live option chain.")
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    intervals = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,"4h":14400,"6h":21600,"1d":86400,"1w":604800}
    interval = intervals[resolution]
    raw = []
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + interval * 1900, end_ts)
        batch = delta_get("/v2/history/candles", {"resolution": resolution, "symbol": symbol, "start": cursor, "end": chunk_end})
        if isinstance(batch, list): raw.extend(batch)
        cursor = chunk_end
    unique = {x.get("time"): x for x in raw if isinstance(x, dict) and x.get("time") is not None}
    rows = sorted(unique.values(), key=lambda x: x.get("time", 0))
    rows = [{"time": int(number(x.get("time"))), "open": number(x.get("open")), "close": number(x.get("close"))}
            for x in rows if number(x.get("time")) + interval <= end_ts and number(x.get("open")) > 0 and number(x.get("close")) > 0]
    if len(rows) < 30:
        return {"asset": asset, "symbol": symbol, "resolution": resolution, "candles": len(rows),
                "closed_trades": 0, "status": "INSUFFICIENT_HISTORICAL_OPTION_CANDLES",
                "message": "Exchange ne is contract ke liye paryapt historical candles nahi diye. Result fabricate nahi kiya gaya.",
                "mode": "PAPER_ONLY"}
    closes = [x["close"] for x in rows]
    fast, slow = [closes[0]], [closes[0]]
    for price in closes[1:]:
        fast.append((2/10)*price + (8/10)*fast[-1])
        slow.append((2/22)*price + (20/22)*slow[-1])
    fee = (taker_fee_pct/100) * (1 + gst_pct/100)
    slip = slippage_pct/100
    position, entry, returns, equity = 0, 0.0, [], [1.0]
    for i in range(22, len(rows)-1):
        previous = fast[i-1] - slow[i-1]
        current = fast[i] - slow[i]
        crossed = (previous <= 0 < current) or (previous >= 0 > current)
        desired = 1 if current > 0 else -1
        if crossed and desired != position:
            price = rows[i+1]["open"]
            if position:
                exit_fill = price*(1-slip) if position == 1 else price*(1+slip)
                gross = position*(exit_fill-entry)/entry
                net = gross-fee-fee*(exit_fill/entry)
                returns.append(net); equity.append(equity[-1]*(1+net))
            entry = price*(1+slip) if desired == 1 else price*(1-slip)
            position = desired
    if position:
        price = rows[-1]["close"]
        exit_fill = price*(1-slip) if position == 1 else price*(1+slip)
        gross = position*(exit_fill-entry)/entry
        net = gross-fee-fee*(exit_fill/entry)
        returns.append(net); equity.append(equity[-1]*(1+net))
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    peak, drawdown = 1.0, 0.0
    for value in equity[1:]:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak-value)/peak if peak else 0)
    total = 1.0
    for value in returns: total *= 1+value
    return {"asset": asset, "symbol": symbol, "resolution": resolution, "candles": len(rows),
            "closed_trades": len(returns), "win_rate_pct": round(100*len(wins)/len(returns),2) if returns else None,
            "net_return_pct": round((total-1)*100,4), "max_drawdown_pct": round(drawdown*100,4),
            "profit_factor": round(sum(wins)/abs(sum(losses)),4) if losses else ("infinite" if wins else None),
            "strategy": "EMA 9/21 on option premium; signal at close, entry next candle open",
            "fee_assumption": {"taker_fee_per_side_pct": taker_fee_pct, "gst_on_fee_pct": gst_pct, "slippage_per_side_pct": slippage_pct},
            "mode": "PAPER_ONLY",
            "warning": "Selected option premium candles ka simulation; historical option-chain/Greeks replay nahi. Assumed costs apply. Future profit guaranteed nahi."}


@app.get("/signals/{asset}")
def paper_signal(asset: str, resolution: str = "1h"):
    asset = asset.upper()
    if asset not in ("BTC", "ETH"):
        raise HTTPException(status_code=400, detail="asset must be BTC or ETH")
    data = candles(asset, resolution, 30)
    rows = data["candles"]
    closes = [number(x.get("close")) for x in rows if number(x.get("close")) > 0]
    if len(closes) < 30:
        return {"asset": asset, "signal": "INSUFFICIENT_DATA", "mode": "PAPER_ONLY", "candles": len(closes)}
    def ema(values, period):
        alpha = 2 / (period + 1)
        value = values[0]
        for item in values[1:]:
            value = alpha * item + (1 - alpha) * value
        return value
    fast = ema(closes[-100:], 9)
    slow = ema(closes[-200:], 21)
    changes = [closes[i] - closes[i-1] for i in range(max(1, len(closes)-15), len(closes))]
    gains = sum(max(x, 0) for x in changes) / max(len(changes), 1)
    losses = sum(abs(min(x, 0)) for x in changes) / max(len(changes), 1)
    rsi = 100 if losses == 0 else 100 - (100 / (1 + gains / losses))
    direction = "BUY" if fast > slow and rsi < 70 else "SELL" if fast < slow and rsi > 30 else "WAIT"
    return {
        "asset": asset, "signal": direction, "last_price": closes[-1],
        "ema_9": round(fast, 4), "ema_21": round(slow, 4), "rsi_14_approx": round(rsi, 2),
        "mode": "PAPER_ONLY",
        "reason": "EMA trend with RSI filter; educational baseline, not a validated strategy.",
        "warning": "Do not trade real money from this signal without independent testing and risk controls."
    }

@app.get("/backtest/{asset}")
def simple_backtest(
    asset: str,
    resolution: str = Query("1h", pattern="^(1m|3m|5m|15m|30m|1h|2h|4h|6h|1d|1w)$"),
    days: int = Query(180, ge=30, le=180),
    taker_fee_pct: float = Query(0.05, ge=0, le=1, description="Delta India futures taker fee per side, percent of notional; default 0.05%"),
    gst_pct: float = Query(18.0, ge=0, le=100, description="GST applied to trading fees; default 18%"),
    slippage_pct: float = Query(0.02, ge=0, le=1, description="Assumed adverse slippage per side, percent; not an exchange fee")
):
    """Historical EMA 9/21 long-short crossover test. PAPER_ONLY; never places orders."""
    asset = asset.upper()
    if asset not in ("BTC", "ETH", "XAUT"):
        raise HTTPException(status_code=400, detail="asset must be BTC, ETH or XAUT")

    raw_rows = candles(asset, resolution, days)["candles"]
    seconds_by_resolution = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
        "1d": 86400, "1w": 604800
    }
    interval = seconds_by_resolution[resolution]
    now = int(time.time())
    # Exclude a still-forming candle to avoid using incomplete information.
    valid_rows = []
    for row in raw_rows:
        stamp = int(number(row.get("time")))
        if stamp > 0 and stamp + interval <= now:
            op, cl = number(row.get("open")), number(row.get("close"))
            if op > 0 and cl > 0:
                valid_rows.append({"time": stamp, "open": op, "close": cl})
    valid_rows.sort(key=lambda row: row["time"])

    if len(valid_rows) < 30:
        return {
            "asset": asset, "resolution": resolution, "candles": len(valid_rows),
            "trades": 0, "message": "Not enough fully closed candles",
            "mode": "PAPER_ONLY"
        }

    closes = [row["close"] for row in valid_rows]
    alpha_fast, alpha_slow = 2 / 10, 2 / 22
    ema_fast = [closes[0]]
    ema_slow = [closes[0]]
    for price in closes[1:]:
        ema_fast.append(alpha_fast * price + (1 - alpha_fast) * ema_fast[-1])
        ema_slow.append(alpha_slow * price + (1 - alpha_slow) * ema_slow[-1])

    fee_rate = (taker_fee_pct / 100) * (1 + gst_pct / 100)
    slip_rate = slippage_pct / 100
    position = 0
    entry_fill = 0.0
    trade_returns = []
    gross_trade_returns = []
    equity_curve = [1.0]
    long_trades = 0
    short_trades = 0

    def close_position(raw_exit, index):
        nonlocal position, entry_fill
        if position == 0:
            return
        # Apply adverse slippage to the exit, then charge both entry and exit fees.
        exit_fill = raw_exit * (1 - slip_rate) if position == 1 else raw_exit * (1 + slip_rate)
        gross = position * (exit_fill - entry_fill) / entry_fill
        entry_fee = fee_rate
        exit_fee = fee_rate * (exit_fill / entry_fill)
        net = gross - entry_fee - exit_fee
        gross_trade_returns.append(gross)
        trade_returns.append(net)
        equity_curve.append(equity_curve[-1] * (1 + net))
        position = 0
        entry_fill = 0.0

    # A crossover confirmed at candle close is executed at the NEXT candle open.
    for i in range(22, len(valid_rows) - 1):
        previous_diff = ema_fast[i - 1] - ema_slow[i - 1]
        current_diff = ema_fast[i] - ema_slow[i]
        desired = 1 if current_diff > 0 else -1
        crossed = (previous_diff <= 0 < current_diff) or (previous_diff >= 0 > current_diff)
        if crossed and desired != position:
            next_open = valid_rows[i + 1]["open"]
            if position != 0:
                close_position(next_open, i + 1)
            # Enter in the newly confirmed direction at the next candle open.
            entry_fill = next_open * (1 + slip_rate) if desired == 1 else next_open * (1 - slip_rate)
            position = desired
            if desired == 1:
                long_trades += 1
            else:
                short_trades += 1

    if position != 0:
        close_position(valid_rows[-1]["close"], len(valid_rows) - 1)

    wins = [value for value in trade_returns if value > 0]
    losses = [value for value in trade_returns if value < 0]
    peak = 1.0
    max_drawdown = 0.0
    for value in equity_curve[1:]:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)

    gross_total = 1.0
    net_total = 1.0
    for value in gross_trade_returns:
        gross_total *= (1 + value)
    for value in trade_returns:
        net_total *= (1 + value)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (None if not wins else "infinite")

    return {
        "asset": asset,
        "resolution": resolution,
        "candles": len(valid_rows),
        "closed_trades": len(trade_returns),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_short": "Both long and short",
        "long_short_strategy": "EMA 9/21 crossover; signal at candle close, execution at next candle open",
        "win_rate_pct": round(100 * len(wins) / len(trade_returns), 2) if trade_returns else None,
        "gross_return_pct": round((gross_total - 1) * 100, 4),
        "net_return_pct": round((net_total - 1) * 100, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "profit_factor": round(profit_factor, 4) if isinstance(profit_factor, (int, float)) else profit_factor,
        "fee_assumption": {
            "taker_fee_per_side_pct": taker_fee_pct,
            "gst_on_fee_pct": gst_pct,
            "effective_fee_per_side_pct": round(fee_rate * 100, 5)
        },
        "slippage_assumption_per_side_pct": slippage_pct,
        "excluded_costs": ["funding", "spread beyond modeled slippage", "taxes/charges other than GST on trading fee"],
        "mode": "PAPER_ONLY",
        "warning": "Historical simulation only, not a validated strategy. Results depend on fee/slippage assumptions and do not predict future returns."
    }




# Multi-factor confluence setup. All calculations use closed candles; read-only.
@app.get("/trade/setup/{asset}")
def timeframe_trade_setup(
    asset: str,
    resolution: str = Query("5m", pattern="^(1m|3m|5m|15m|30m|1h|2h|4h|6h|1d)$"),
    target_points: float = Query(300, gt=0, le=100000)
):
    asset = asset.upper()
    if asset not in ("BTC", "ETH", "XAUT"):
        raise HTTPException(status_code=400, detail="asset must be BTC, ETH or XAUT")
    days_by_tf = {"1m":3,"3m":7,"5m":30,"15m":60,"30m":90,"1h":180,"2h":180,"4h":180,"6h":180,"1d":180}
    secs = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,"4h":14400,"6h":21600,"1d":86400}
    raw = candles(asset, resolution, days_by_tf[resolution]).get("candles", [])
    interval, now = secs[resolution], int(time.time())
    rows = [x for x in raw if number(x.get("time")) > 0 and number(x.get("time"))+interval <= now
            and all(number(x.get(k)) > 0 for k in ("open","high","low","close"))]
    rows.sort(key=lambda x:number(x.get("time")))
    if len(rows) < 60:
        return {"asset":asset,"timeframe":resolution,"signal":"WAIT","status":"INSUFFICIENT_CLOSED_CANDLES",
                "candles":len(rows),"mode":"PAPER_ONLY","reason":"Is timeframe par kam se kam 60 fully closed candles chahiye."}

    op=[number(x["open"]) for x in rows]; hi=[number(x["high"]) for x in rows]
    lo=[number(x["low"]) for x in rows]; cl=[number(x["close"]) for x in rows]
    vol=[max(0,number(x.get("volume"))) for x in rows]
    e9=_ema_series(cl,9)[-1]; e20=_ema_series(cl,20)[-1]; e50=_ema_series(cl,50)[-1]
    rsi=_rsi_series(cl,14)[-1]
    tr=[max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1])) for i in range(1,len(rows))]
    atr=sum(tr[-14:])/max(1,len(tr[-14:]))
    last=len(rows)-1; price=cl[-1]
    # Rolling 20-bar VWAP; exchange volume may be absent, in which case disclose fallback.
    vwindow=vol[-20:]; typical=[(hi[i]+lo[i]+cl[i])/3 for i in range(len(rows)-20,len(rows))]
    vwap_volume=sum(vwindow)
    vwap=sum(typical[j]*(vwindow[j] if vwap_volume>0 else 1) for j in range(20))/(vwap_volume if vwap_volume>0 else 20)
    volume_available=vwap_volume>0
    avg_vol=sum(vol[-21:-1])/20 if len(vol)>=21 else 0
    vol_ratio=(vol[-1]/avg_vol) if avg_vol>0 else None
    # Liquidity sweep: wick through prior 20-bar extreme then close back inside.
    prior_hi=max(hi[-21:-1]); prior_lo=min(lo[-21:-1])
    bull_sweep=lo[-1]<prior_lo and price>prior_lo
    bear_sweep=hi[-1]>prior_hi and price<prior_hi
    # Structure break confirmed by candle close beyond prior 10 completed bars.
    range_hi=max(hi[-11:-1]); range_lo=min(lo[-11:-1])
    bull_bos=price>range_hi; bear_bos=price<range_lo
    # Three-candle imbalance/FVG on latest closed candle.
    bull_fvg=lo[-1]>hi[-3]; bear_fvg=hi[-1]<lo[-3]
    # Latest candle rejection / body quality.
    body=abs(cl[-1]-op[-1]); lower_wick=min(op[-1],cl[-1])-lo[-1]; upper_wick=hi[-1]-max(op[-1],cl[-1])
    bull_reject=cl[-1]>op[-1] and lower_wick>=max(body*1.5,atr*0.08)
    bear_reject=cl[-1]<op[-1] and upper_wick>=max(body*1.5,atr*0.08)

    components=[]
    score=0
    def add(name, direction, weight, detail):
        nonlocal score
        points=direction*weight
        score+=points
        components.append({"factor":name,"reading":detail,"direction":"BULLISH" if direction>0 else "BEARISH" if direction<0 else "NEUTRAL",
                           "score":points,"max_score":weight})
    add("EMA 9 / EMA 20",1 if e9>e20 else -1 if e9<e20 else 0,2,
        f"EMA9={e9:.4f}; EMA20={e20:.4f}; {'9 above 20' if e9>e20 else '9 below 20' if e9<e20 else 'overlap'}")
    add("EMA 50 trend",1 if price>e50 else -1 if price<e50 else 0,2,
        f"Close={price:.4f}; EMA50={e50:.4f}")
    add("RSI 14",1 if 52<=rsi<=68 else -1 if 32<=rsi<=48 else 0,1,
        f"RSI={rsi:.2f}; bullish band 52–68, bearish band 32–48")
    add("VWAP 20",1 if price>vwap else -1 if price<vwap else 0,1,
        f"Close={price:.4f}; VWAP20={vwap:.4f}; volume {'available' if volume_available else 'missing, equal-weight fallback used'}")
    add("Liquidity sweep",1 if bull_sweep else -1 if bear_sweep else 0,3,
        "Bullish sweep: low pierced prior 20-bar low and reclaimed" if bull_sweep else "Bearish sweep: high pierced prior 20-bar high and closed back below" if bear_sweep else "No confirmed sweep of prior 20-bar high/low")
    add("Break of structure",1 if bull_bos else -1 if bear_bos else 0,2,
        "Close broke prior 10-bar high" if bull_bos else "Close broke prior 10-bar low" if bear_bos else "No 10-bar closing break")
    add("Fair Value Gap",1 if bull_fvg else -1 if bear_fvg else 0,1,
        "Bullish 3-candle gap" if bull_fvg else "Bearish 3-candle gap" if bear_fvg else "No fresh 3-candle gap")
    add("Volume expansion",1 if vol_ratio is not None and vol_ratio>=1.5 and price>op[-1] else -1 if vol_ratio is not None and vol_ratio>=1.5 and price<op[-1] else 0,1,
        f"Latest / prior-20 average volume={vol_ratio:.2f}x" if vol_ratio is not None else "Volume unavailable")
    add("Candle rejection",1 if bull_reject else -1 if bear_reject else 0,1,
        "Bullish lower-wick rejection" if bull_reject else "Bearish upper-wick rejection" if bear_reject else "No strong rejection candle")

    # Higher timeframe trend alignment; only closed candles count.
    htf=[]
    htf_frames=["15m","1h"] if resolution in ("1m","3m","5m") else ["1h","4h"] if resolution in ("15m","30m") else ["4h","1d"]
    for tf in htf_frames:
        try:
            hraw=candles(asset,tf,30).get("candles",[])
            hrows=[x for x in hraw if number(x.get("time"))>0 and number(x.get("time"))+secs[tf]<=now and number(x.get("close"))>0]
            hcl=[number(x["close"]) for x in hrows]
            if len(hcl)>=55:
                h20=_ema_series(hcl,20)[-1]; h50=_ema_series(hcl,50)[-1]
                hs=1 if hcl[-1]>h20>h50 else -1 if hcl[-1]<h20<h50 else 0
                htf.append({"timeframe":tf,"signal":"BULLISH" if hs>0 else "BEARISH" if hs<0 else "MIXED",
                            "close":round(hcl[-1],4),"ema20":round(h20,4),"ema50":round(h50,4)})
                add("Higher timeframe "+tf,hs,2,f"Close={hcl[-1]:.4f}; EMA20={h20:.4f}; EMA50={h50:.4f}")
            else:
                htf.append({"timeframe":tf,"signal":"INSUFFICIENT_DATA"})
        except Exception as exc:
            htf.append({"timeframe":tf,"signal":"UNAVAILABLE","detail":str(exc)[:120]})

    max_score=sum(x["max_score"] for x in components)
    threshold=max(7,round(max_score*0.45))
    # Avoid chasing after a large candle; require confirmation beyond current closed candle.
    direction="BUY" if score>=threshold else "SELL" if score<=-threshold else "WAIT"
    reasons=[f"{x['factor']}: {x['reading']} (score {x['score']:+d})" for x in components]
    result={"asset":asset,"timeframe":resolution,"signal":direction,"mode":"PAPER_ONLY",
        "status":"NO_TRADE" if direction=="WAIT" else "CONFLUENCE_SETUP",
        "last_closed_candle_time":int(number(rows[-1].get("time"))),"last_closed_price":round(price,4),
        "ema9":round(e9,4),"ema20":round(e20,4),"ema50":round(e50,4),"rsi14":round(rsi,2),
        "vwap20":round(vwap,4),"atr14":round(atr,4),"volume_ratio":round(vol_ratio,2) if vol_ratio is not None else None,
        "confluence_score":score,"max_confluence_score":max_score,"decision_threshold":threshold,
        "components":components,"higher_timeframes":htf,"logic":reasons,
        "liquidity_sweep":{"bullish":bull_sweep,"bearish":bear_sweep,"prior_20bar_high":round(prior_hi,4),"prior_20bar_low":round(prior_lo,4)},
        "structure":{"bullish_bos":bull_bos,"bearish_bos":bear_bos,"prior_10bar_high":round(range_hi,4),"prior_10bar_low":round(range_lo,4)},
        "target_points_requested":target_points}
    if direction=="WAIT":
        result.update({"entry":None,"stop_loss":None,"target_1":None,"target_2":None,
            "why_no_trade":"Weighted confluence score threshold tak nahi pahuncha. Sirf EMA/RSI ke aadhar par trade nahi diya jayega."})
        return result
    swing_stop=(price-prior_lo) if direction=="BUY" else (prior_hi-price)
    risk=max(1.2*atr,swing_stop)
    if risk<=0:
        result.update({"signal":"WAIT","status":"INVALID_RISK_LEVELS","why_no_trade":"ATR/swing stop invalid."})
        return result
    sl=price-risk if direction=="BUY" else price+risk
    t1=price+target_points if direction=="BUY" else price-target_points
    t2=price+2*target_points if direction=="BUY" else price-2*target_points
    rr=target_points/risk
    result.update({"status":"SETUP_READY" if rr>=1.5 else "WAIT_LOW_RISK_REWARD",
        "entry_type":"Only after next candle confirms direction; last close is reference, not a market order",
        "entry":round(price,4),"stop_loss":round(sl,4),"target_1":round(t1,4),"target_2":round(t2,4),
        "risk_points":round(risk,4),"reward_risk_to_target_1":round(rr,2),
        "why_trade":"Multiple independent factors align; review component scores and higher-timeframe context." if rr>=1.5 else "Confluence direction exists, but requested target gives less than 1:1.5 reward/risk; wait.",
        "trailing_stop_rule":"After Target 1, trail by 1 ATR from the best price reached; paper rule only.",
        "warning":"Confluence is a scoring model, not perfect or guaranteed. Candle close data only; fees/slippage may alter levels. No orders are placed."})
    return result

# Public real-time market feed. This relays Delta's public WebSocket only.
DELTA_PUBLIC_WS = "wss://public-socket.india.delta.exchange"

@app.get("/market/live/status")
def live_market_status():
    return {
        "status": "available",
        "provider": "Delta Exchange public WebSocket",
        "websocket_path": "/ws/market",
        "symbols": ["BTCUSD", "ETHUSD"],
        "channels": ["ticker", "mark_price", "funding_rate", "trades"],
        "mode": "READ_ONLY",
        "note": "Public market feed only. No account data or order placement."
    }

@app.get("/live", response_class=HTMLResponse)
def live_market_page():
    return HTMLResponse("""
<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delta AI Bot Live Market</title><style>
body{font-family:Arial,sans-serif;background:#101820;color:#f4f7fa;margin:0;padding:18px}
h2{margin:4px 0 8px}.muted{color:#aab7c4;font-size:13px}
.card{background:#1b2a36;border-radius:12px;padding:16px;margin:14px 0}
.price{font-size:30px;font-weight:bold;margin:8px 0}
.row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #344451}
#state{padding:9px;border-radius:8px;background:#344451;margin:12px 0}
</style></head><body>
<h2>Delta AI Bot — Live Market</h2>
<div class="muted">BTCUSD / ETHUSD · Delta public WebSocket · Read-only</div>
<div id="state">Connecting…</div>
<div class="card"><h3>BTC</h3><div class="price" id="BTC-price">Waiting…</div>
<div class="row"><span>Mark price</span><b id="BTC-mark">—</b></div>
<div class="row"><span>Spot price</span><b id="BTC-spot">—</b></div>
<div class="row"><span>24h change</span><b id="BTC-change">—</b></div>
<div class="row"><span>Open interest</span><b id="BTC-oi">—</b></div>
<div class="row"><span>Last update</span><b id="BTC-time">—</b></div></div>
<div class="card"><h3>ETH</h3><div class="price" id="ETH-price">Waiting…</div>
<div class="row"><span>Mark price</span><b id="ETH-mark">—</b></div>
<div class="row"><span>Spot price</span><b id="ETH-spot">—</b></div>
<div class="row"><span>24h change</span><b id="ETH-change">—</b></div>
<div class="row"><span>Open interest</span><b id="ETH-oi">—</b></div>
<div class="row"><span>Last update</span><b id="ETH-time">—</b></div></div>
<div class="muted">Prices are exchange market data, not trading advice. No orders are placed.</div>
<script>
const state=document.getElementById('state');
const scheme=location.protocol==='https:'?'wss':'ws';
const ws=new WebSocket(scheme+'://'+location.host+'/ws/market');
ws.onopen=()=>state.textContent='Connected — waiting for Delta feed';
ws.onclose=()=>state.textContent='Disconnected — refresh this page to reconnect';
ws.onerror=()=>state.textContent='Connection error — check Render logs';
ws.onmessage=(ev)=>{
 try{
  const x=JSON.parse(ev.data);
  if(x.type==='connection'){state.textContent=x.status;return;}
  const symbol=x.sy||x.symbol||'';
  const asset=symbol.startsWith('BTC')?'BTC':symbol.startsWith('ETH')?'ETH':null;
  if(!asset)return;
  if(x.type==='ticker' && Array.isArray(x.d) && x.d.length){
   const d=x.d[0], o=d.ohlc||[], oi=d.oi||[];
   document.getElementById(asset+'-price').textContent=o[3]??'—';
   document.getElementById(asset+'-mark').textContent=d.m??'—';
   document.getElementById(asset+'-spot').textContent=x.sp??'—';
   document.getElementById(asset+'-change').textContent=d.m24hc??'—';
   document.getElementById(asset+'-oi').textContent=oi[0]??'—';
   document.getElementById(asset+'-time').textContent=new Date(Number(x.ts||Date.now())/1000).toLocaleTimeString();
   state.textContent='Live feed active · last update '+new Date().toLocaleTimeString();
  } else if(x.type==='mark_price'){
   document.getElementById(asset+'-mark').textContent=x.p??'—';
   state.textContent='Live mark price update · '+new Date().toLocaleTimeString();
  }
 }catch(e){}
};
</script></body></html>
""")

@app.websocket("/ws/market")
async def live_market_websocket(client: WebSocket):
    await client.accept()
    try:
        async with websockets.connect(
            DELTA_PUBLIC_WS,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
            max_size=2**20
        ) as upstream:
            subscription = {
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {"name": "ticker", "symbols": ["BTCUSD", "ETHUSD"]},
                        {"name": "mark_price", "symbols": ["MARK:BTCUSD", "MARK:ETHUSD"]},
                        {"name": "funding_rate", "symbols": ["BTCUSD", "ETHUSD"]},
                        {"name": "trades", "symbols": ["BTCUSD", "ETHUSD"]}
                    ]
                }
            }
            await upstream.send(json.dumps(subscription))
            await client.send_json({
                "type": "connection",
                "status": "connected to Delta public market feed",
                "mode": "READ_ONLY"
            })

            async def relay_market():
                async for message in upstream:
                    await client.send_text(message)

            async def watch_disconnect():
                while True:
                    await client.receive_text()

            tasks = {
                asyncio.create_task(relay_market()),
                asyncio.create_task(watch_disconnect())
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("Live market WebSocket failed")
        try:
            await client.send_json({"type": "error", "message": "Live feed disconnected. Refresh to reconnect."})
        except Exception:
            pass
    finally:
        try:
            await client.close()
        except Exception:
            pass


# Read-only MCP tools for compatible AI clients. No order-placement tools are exposed.
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP("Delta AI Trading Bot V2")

@mcp_server.tool()
def mcp_market_tickers(asset: str = "BTC") -> dict:
    """Read BTC or ETH perpetual/futures market tickers from Delta Exchange."""
    asset = asset.upper()
    if asset not in ("BTC", "ETH"):
        raise ValueError("asset must be BTC or ETH")
    return {"asset": asset, "data": market_tickers(asset)}

@mcp_server.tool()
def mcp_options_analytics(asset: str = "BTC") -> dict:
    """Read available option-chain open interest, PCR and estimated max pain."""
    asset = asset.upper()
    if asset not in ("BTC", "ETH"):
        raise ValueError("asset must be BTC or ETH")
    return options_analytics(asset)

@mcp_server.tool()
def mcp_paper_signal(asset: str = "BTC", resolution: str = "1h") -> dict:
    """Get an educational paper-only EMA/RSI signal; this never places an order."""
    return paper_signal(asset, resolution)

@mcp_server.tool()
def mcp_backtest(asset: str = "BTC", resolution: str = "1h", days: int = 180) -> dict:
    """Run a paper-only long/short EMA 9/21 backtest with configurable taker fee, GST and slippage assumptions."""
    return simple_backtest(asset, resolution, days)


# Multi-strategy research engine. Signals and historical simulations are PAPER_ONLY.
def _sma(values, period):
    part = values[-period:]
    return sum(part) / len(part) if part else 0.0

def _ema_series(values, period):
    if not values: return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1-alpha) * out[-1])
    return out

def _rsi_series(values, period=14):
    out = [50.0] * len(values)
    for i in range(1, len(values)):
        start = max(1, i-period+1)
        changes = [values[j]-values[j-1] for j in range(start, i+1)]
        gains = sum(max(v,0) for v in changes) / max(len(changes),1)
        losses = sum(abs(min(v,0)) for v in changes) / max(len(changes),1)
        out[i] = 100.0 if losses == 0 and gains > 0 else (50.0 if losses == 0 else 100-(100/(1+gains/losses)))
    return out

def _strategy_directions(rows, options_pcr=None):
    closes=[number(x.get("close")) for x in rows]
    highs=[number(x.get("high")) for x in rows]
    lows=[number(x.get("low")) for x in rows]
    opens=[number(x.get("open")) for x in rows]
    volumes=[number(x.get("volume")) for x in rows]
    n=len(rows)
    e9,e21=_ema_series(closes,9),_ema_series(closes,21)
    rsi=_rsi_series(closes)
    out={k:[0]*n for k in ["ema_rsi","vwap","smc","fvg","liquidity_sweep","support_resistance","trendline","candlestick","options_oi"]}
    cum_pv=cum_vol=0.0
    active_day=None
    for i in range(n):
        # Reset VWAP at each UTC day boundary; do not anchor it across the full 180-day history.
        stamp=int(number(rows[i].get("time")))
        candle_day=stamp//86400 if stamp>0 else active_day
        if active_day is not None and candle_day != active_day:
            cum_pv=cum_vol=0.0
        active_day=candle_day
        typical=(highs[i]+lows[i]+closes[i])/3 if all([highs[i],lows[i],closes[i]]) else closes[i]
        # Missing exchange volume is disclosed separately; equal weighting is only a fallback.
        weight=volumes[i] if volumes[i]>0 else 1.0
        cum_pv += typical*weight
        cum_vol += weight
        vwap=cum_pv/cum_vol if cum_vol else closes[i]
        if i>=21:
            if e9[i]>e21[i] and 45<=rsi[i]<=70: out["ema_rsi"][i]=1
            elif e9[i]<e21[i] and 30<=rsi[i]<=55: out["ema_rsi"][i]=-1
            if closes[i]>vwap and e9[i]>e21[i] and rsi[i]>50: out["vwap"][i]=1
            elif closes[i]<vwap and e9[i]<e21[i] and rsi[i]<50: out["vwap"][i]=-1
            # Swing structure / break of structure using the prior 10 completed bars.
            prior_hi=max(highs[max(0,i-10):i]); prior_lo=min(lows[max(0,i-10):i])
            if closes[i]>prior_hi: out["smc"][i]=1
            elif closes[i]<prior_lo: out["smc"][i]=-1
            # Three-candle fair-value gap.
            if i>=2 and lows[i]>highs[i-2]: out["fvg"][i]=1
            elif i>=2 and highs[i]<lows[i-2]: out["fvg"][i]=-1
            # Sweep prior 10-bar liquidity then close back inside range.
            if lows[i]<prior_lo and closes[i]>prior_lo: out["liquidity_sweep"][i]=1
            elif highs[i]>prior_hi and closes[i]<prior_hi: out["liquidity_sweep"][i]=-1
            # Rolling support/resistance breakout.
            if closes[i]>prior_hi: out["support_resistance"][i]=1
            elif closes[i]<prior_lo: out["support_resistance"][i]=-1
            # Close beyond a rolling 20-bar regression endpoint.
            w=20; xs=list(range(w)); ys=closes[i-w:i]
            xm=sum(xs)/w; ym=sum(ys)/w
            den=sum((x-xm)**2 for x in xs)
            slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/den if den else 0
            projected=ym+slope*(w-1)
            if closes[i]>projected*1.001 and slope>0: out["trendline"][i]=1
            elif closes[i]<projected*0.999 and slope<0: out["trendline"][i]=-1
            # Hammer / inverted hammer with directional close confirmation.
            body=abs(closes[i]-opens[i]); rng=max(highs[i]-lows[i],1e-12)
            lower=min(opens[i],closes[i])-lows[i]; upper=highs[i]-max(opens[i],closes[i])
            if lower>=2*max(body,1e-12) and upper<=body and closes[i]>opens[i]: out["candlestick"][i]=1
            elif upper>=2*max(body,1e-12) and lower<=body and closes[i]<opens[i]: out["candlestick"][i]=-1
            # Current options PCR is deliberately NOT copied into historical candles.
            # Historical option-chain snapshots are not available in this backtest, so doing
            # so would introduce look-ahead bias. PCR is applied only to the current live vote.
    return out

@app.get("/strategies/{asset}")
def all_strategy_analysis(asset: str, resolution: str = Query("1h", pattern="^(1m|3m|5m|15m|30m|1h|2h|4h|6h|1d|1w)$"), days: int = Query(180, ge=1, le=180)):
    asset=asset.upper()
    if asset not in ("BTC","ETH","XAUT"):
        raise HTTPException(status_code=400, detail="asset must be BTC, ETH or XAUT")
    data=candles(asset,resolution,days)
    rows=data.get("candles",[])
    if len(rows)<30:
        return {"asset":asset,"mode":"PAPER_ONLY","status":"INSUFFICIENT_DATA","candles":len(rows),"strategies":[]}
    opt=None
    try:
        a=options_analytics(asset)
        opt=a.get("pcr_by_open_interest")
    except Exception as exc:
        log.warning("Options context unavailable for %s: %s",asset,exc)
    dirs=_strategy_directions(rows,opt)
    names={
        "ema_rsi":"EMA 9/21 + RSI","vwap":"VWAP + EMA + RSI","smc":"SMC / Break of Structure",
        "fvg":"Fair Value Gap","liquidity_sweep":"Liquidity Sweep","support_resistance":"Support & Resistance",
        "trendline":"Trendline Breakout","candlestick":"Hammer / Inverted Hammer","options_oi":"Options OI + PCR"
    }
    results=[]
    score=0
    for key,label in names.items():
        sig=dirs[key][-1]
        if key=="options_oi":
            # Current PCR is context only; it has no historical time series to backtest.
            sig=1 if opt is not None and opt>1.1 else -1 if opt is not None and opt<0.9 else 0
            signal="BUY" if sig==1 else "SELL" if sig==-1 else "WAIT"
            results.append({"strategy":label,"key":key,"signal":signal,"backtest_trades":0,
                "win_rate_pct":None,"net_return_pct":None,"max_drawdown_pct":None,
                "status":"LIVE_CONTEXT_ONLY_NO_HISTORICAL_PCR_REPLAY",
                "note":"Current option-chain PCR is not backtested against historical candles because historical PCR snapshots are unavailable."})
            if opt is not None: score+=sig
            continue
        signal="BUY" if sig==1 else "SELL" if sig==-1 else "WAIT"
        score+=sig
        # Signal is formed at candle close and evaluated from the next candle open to close.
        rets=[]
        for i in range(len(rows)-1):
            d=dirs[key][i]
            entry=number(rows[i+1].get("open"))
            exit_price=number(rows[i+1].get("close"))
            if d and entry>0 and exit_price>0: rets.append(d*(exit_price-entry)/entry)
        wins=[v for v in rets if v>0]
        equity=peak=1.0; dd=0.0
        for v in rets:
            equity*=1+v; peak=max(peak,equity); dd=max(dd,(peak-equity)/peak if peak else 0)
        total=1.0
        for v in rets: total*=1+v
        results.append({"strategy":label,"key":key,"signal":signal,"backtest_trades":len(rets),
            "win_rate_pct":round(100*len(wins)/len(rets),2) if rets else None,
            "net_return_pct":round((total-1)*100,4),"max_drawdown_pct":round(dd*100,4),
            "status":"ILLUSTRATIVE_PAPER_BACKTEST_NO_FEES"})
    combined="BUY" if score>=2 else "SELL" if score<=-2 else "WAIT"
    return {"asset":asset,"resolution":resolution,"days_requested":days,"candles":len(rows),
        "mode":"PAPER_ONLY","combined_signal":combined,"agreement_score":score,
        "options_pcr_by_open_interest":opt,"strategies":results,
        "warning":"Illustrative in-sample directional simulation, not validated. No fees, slippage, funding, spread or execution delay are modeled. No orders are placed."}


@app.get("/market/outlook/{asset}")
def multi_timeframe_outlook(asset: str, days: int = Query(180, ge=30, le=180)):
    """Multi-timeframe directional context; uses only fully closed exchange candles."""
    asset = asset.upper()
    if asset not in ("BTC", "ETH", "XAUT"):
        raise HTTPException(status_code=400, detail="asset must be BTC, ETH or XAUT")
    frames = {"5m": 30, "15m": 60, "1h": days, "4h": days, "1d": days}
    results = []
    weights = {"5m": 1, "15m": 1, "1h": 2, "4h": 3, "1d": 4}
    total_score = 0
    for resolution, lookback in frames.items():
        try:
            raw = candles(asset, resolution, lookback).get("candles", [])
            seconds = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[resolution]
            now = int(time.time())
            rows = [x for x in raw if number(x.get("time")) > 0
                    and number(x.get("time")) + seconds <= now
                    and number(x.get("close")) > 0]
            closes = [number(x.get("close")) for x in rows]
            if len(closes) < 55:
                results.append({"timeframe": resolution, "signal": "INSUFFICIENT_DATA",
                                "candles": len(closes), "status": "NOT_ENOUGH_CLOSED_CANDLES"})
                continue
            ema20 = _ema_series(closes, 20)[-1]
            ema50 = _ema_series(closes, 50)[-1]
            rsi = _rsi_series(closes)[-1]
            last = closes[-1]
            recent = closes[-21:-1]
            swing_high, swing_low = max(recent), min(recent)
            if last > ema20 > ema50 and rsi >= 52:
                score, signal = 1, "BULLISH"
            elif last < ema20 < ema50 and rsi <= 48:
                score, signal = -1, "BEARISH"
            else:
                score, signal = 0, "SIDEWAYS / MIXED"
            if last > swing_high:
                structure = "BREAKOUT_ABOVE_20_BAR_HIGH"
            elif last < swing_low:
                structure = "BREAKDOWN_BELOW_20_BAR_LOW"
            else:
                structure = "INSIDE_20_BAR_RANGE"
            total_score += score * weights[resolution]
            results.append({
                "timeframe": resolution, "signal": signal, "score": score,
                "last_closed_price": round(last, 4), "ema20": round(ema20, 4),
                "ema50": round(ema50, 4), "rsi14_approx": round(rsi, 2),
                "swing_high_20": round(swing_high, 4), "swing_low_20": round(swing_low, 4),
                "market_structure": structure, "candles": len(closes),
                "lookback_days": lookback, "status": "OK"
            })
        except Exception as exc:
            log.warning("Multi-timeframe outlook failed %s %s: %s", asset, resolution, exc)
            results.append({"timeframe": resolution, "signal": "UNAVAILABLE",
                            "status": "API_ERROR", "message": str(exc)[:240]})
    if total_score >= 5:
        overall = "BULLISH BIAS"
    elif total_score <= -5:
        overall = "BEARISH BIAS"
    else:
        overall = "MIXED / WAIT"
    return {
        "asset": asset, "overall_bias": overall, "weighted_score": total_score,
        "timeframes": results, "mode": "PAPER_ONLY",
        "method": "Closed candles only; EMA20/50 trend, approximate RSI14 and 20-bar structure. Higher timeframes carry more weight.",
        "warning": "This is a probabilistic market-bias dashboard, not a guaranteed price prediction or trade instruction."
    }


@app.get("/trading/status")
def trading_status():
    return {
        "mode": "PAPER_ONLY",
        "live_trading_enabled": False,
        "order_execution_available": False,
        "reason": "Live order placement is intentionally disabled until API credentials, product, position sizing and hard risk limits are configured and verified.",
        "mcp_endpoint": "/mcp"
    }


# ATR-based illustrative paper plan. This endpoint never places orders.
@app.get("/paper/plan/{asset}")
def paper_plan(asset: str):
    asset = asset.upper()
    if asset not in ("BTC", "ETH"):
        raise HTTPException(status_code=400, detail="asset must be BTC or ETH")

    signal_data = paper_signal(asset, "1h")
    if signal_data.get("signal") not in ("BUY", "SELL"):
        return {
            "asset": asset,
            "signal": signal_data.get("signal", "WAIT"),
            "mode": "PAPER_ONLY",
            "plan_status": "NO_TRADE",
            "reason": "The current EMA/RSI filter did not produce a BUY or SELL signal.",
            "signal_details": signal_data
        }

    candle_data = candles(asset, "1h", 30)
    rows = candle_data.get("candles", [])
    valid = [
        row for row in rows
        if all(number(row.get(key)) > 0 for key in ("high", "low", "close"))
    ]
    if len(valid) < 15:
        return {
            "asset": asset,
            "signal": signal_data["signal"],
            "mode": "PAPER_ONLY",
            "plan_status": "INSUFFICIENT_DATA",
            "candles_available": len(valid)
        }

    true_ranges = []
    for index in range(1, len(valid)):
        high = number(valid[index].get("high"))
        low = number(valid[index].get("low"))
        previous_close = number(valid[index - 1].get("close"))
        true_ranges.append(max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        ))
    atr_value = sum(true_ranges[-14:]) / min(14, len(true_ranges[-14:]))
    entry = number(valid[-1].get("close"))
    if entry <= 0 or atr_value <= 0:
        raise HTTPException(status_code=502, detail="Could not calculate a valid entry reference and ATR")

    risk_distance = 1.5 * atr_value
    reward_1 = 1.5 * risk_distance
    reward_2 = 2.0 * risk_distance

    if signal_data["signal"] == "BUY":
        stop_loss = entry - risk_distance
        target_1 = entry + reward_1
        target_2 = entry + reward_2
    else:
        stop_loss = entry + risk_distance
        target_1 = entry - reward_1
        target_2 = entry - reward_2

    return {
        "asset": asset,
        "timeframe": "1h",
        "signal": signal_data["signal"],
        "mode": "PAPER_ONLY",
        "plan_status": "ILLUSTRATIVE_NOT_EXECUTED",
        "entry_reference": round(entry, 2),
        "atr_14": round(atr_value, 2),
        "stop_loss": round(stop_loss, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "trailing_stop": {
            "enabled": True,
            "activation": "Only after target_1 is reached",
            "activation_price": round(target_1, 2),
            "trail_distance": round(atr_value, 2),
            "distance_atr_multiple": 1.0,
            "long_rule": "After activation: highest price since activation minus trail_distance",
            "short_rule": "After activation: lowest price since activation plus trail_distance",
            "mode": "PAPER_ONLY",
            "note": "This is a trailing-stop rule, not an automatically tracked or exchange-placed stop order."
        },
        "risk_reward": "1:1.5 and 1:2",
        "sizing": "Not calculated; account balance and maximum risk per trade are not configured.",
        "signal_details": signal_data,
        "warning": "Illustrative ATR-based levels only. No order is placed; signals and levels are not validated. Check fees, slippage and risk before any real trade."
    }

# Streamable HTTP MCP endpoint. All exposed tools are read-only.
app.mount("/mcp", mcp_server.streamable_http_app())
