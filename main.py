import os
import time
import logging
import asyncio
import json
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import websockets
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("delta-ai-bot")

BASE_URL = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange").rstrip("/")
TIMEOUT = 15
SESSION = requests.Session()
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

@app.get("/")
def home():
    return {
        "name": "Delta AI Trading Bot V2",
        "status": "online",
        "mode": "PAPER_ONLY",
        "assets": ["BTC", "ETH"],
        "docs": "/docs",
        "warning": "Signals are educational and unvalidated. No real-money orders are placed."
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "delta-ai-trading-bot-v2", "mode": "PAPER_ONLY"}

@app.get("/market/tickers")
def market_tickers(asset: str = Query("BTC", pattern="^(BTC|ETH)$")):
    return delta_get("/v2/tickers", {
        "underlying_asset_symbols": asset,
        "contract_types": "perpetual_futures,futures"
    })

@app.get("/market/options")
def options_chain(
    asset: str = Query("BTC", pattern="^(BTC|ETH)$"),
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
    asset: str = Query("BTC", pattern="^(BTC|ETH)$"),
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
    asset: str = Query("BTC", pattern="^(BTC|ETH)$"),
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
    resolution: str = "1h",
    days: int = Query(180, ge=30, le=180)
):
    asset = asset.upper()
    if asset not in ("BTC", "ETH"):
        raise HTTPException(status_code=400, detail="asset must be BTC or ETH")
    rows = candles(asset, resolution, days)["candles"]
    closes = [number(x.get("close")) for x in rows if number(x.get("close")) > 0]
    if len(closes) < 30:
        return {"asset": asset, "trades": 0, "message": "Not enough candles"}
    # Simple EMA crossover research baseline. Fees, slippage and funding are not included.
    position = 0
    entry = 0.0
    trade_returns = []
    for i in range(25, len(closes)):
        fast = sum(closes[i-8:i+1]) / 9
        slow = sum(closes[i-20:i+1]) / 21
        desired = 1 if fast > slow else 0
        if position and not desired:
            trade_returns.append((closes[i] - entry) / entry)
            position = 0
        elif not position and desired:
            entry = closes[i]
            position = 1
    if position:
        trade_returns.append((closes[-1] - entry) / entry)
    equity = 1.0
    for ret in trade_returns:
        equity *= (1 + ret)
    wins = sum(1 for x in trade_returns if x > 0)
    return {
        "asset": asset, "resolution": resolution, "candles": len(closes),
        "closed_trades": len(trade_returns), "win_rate_pct": round(100*wins/len(trade_returns), 2) if trade_returns else None,
        "gross_return_pct": round((equity-1)*100, 2),
        "assumptions": "Long-only EMA 9/21 crossover. Excludes fees, slippage, funding, spread and execution delay.",
        "warning": "Illustrative historical test only; past results do not predict future returns."
    }



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
    """Run the illustrative historical EMA crossover backtest; fees/slippage are excluded."""
    return simple_backtest(asset, resolution, days)

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
        "risk_reward": "1:1.5 and 1:2",
        "sizing": "Not calculated; account balance and maximum risk per trade are not configured.",
        "signal_details": signal_data,
        "warning": "Illustrative ATR-based levels only. No order is placed; signals and levels are not validated. Check fees, slippage and risk before any real trade."
    }

# Streamable HTTP MCP endpoint. All exposed tools are read-only.
app.mount("/mcp", mcp_server.streamable_http_app())
