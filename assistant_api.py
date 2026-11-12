import os
import logging
from typing import List, Literal
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

log = logging.getLogger("delta-ai-assistant")
router = APIRouter()

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=12)
    asset: Literal["BTC", "ETH", "XAUT"] = "BTC"
    timeframe: Literal["5m", "15m", "1h", "4h", "1d"] = "1h"

@router.get("/assistant", response_class=HTMLResponse)
def assistant_page():
    return HTMLResponse("""<!doctype html>
<html lang="hi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delta AI Assistant</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#080e1b;color:#edf4ff;font:15px system-ui}
header{padding:17px;background:#111c2e;border-bottom:1px solid #263a56}header a{color:#8dbaff;text-decoration:none}
main{max-width:850px;margin:auto;padding:14px}.bar,.chatbox{background:#111c2e;border:1px solid #263a56;border-radius:14px;padding:13px;margin-bottom:12px}
select,textarea,button{background:#203b60;color:#fff;border:1px solid #35557e;border-radius:9px;padding:11px;font:inherit}
select{margin:4px 6px 4px 0}#messages{height:55vh;min-height:320px;overflow:auto;display:flex;flex-direction:column;gap:10px}
.msg{white-space:pre-wrap;line-height:1.55;padding:12px;border-radius:12px;max-width:94%;overflow-wrap:anywhere}
.user{align-self:flex-end;background:#23456e}.assistant{align-self:flex-start;background:#17283d;border:1px solid #304866}
form{display:flex;gap:8px;margin-top:10px}textarea{flex:1;resize:vertical;min-height:55px;max-height:160px;background:#091321}
button{font-weight:700;cursor:pointer}button:disabled{opacity:.5}small,.muted{color:#91a5c2}.warn{color:#f5c66b;font-size:12px}
@media(max-width:520px){main{padding:9px}#messages{height:53vh}form{flex-direction:column}button{width:100%}}
</style></head><body><header><b>🤖 Delta AI Assistant</b><br><small>Live market context · Hindi / Hinglish · Read-only · PAPER_ONLY</small><p><a href="/">← Back to Dashboard</a></p></header>
<main><div class="bar"><label>Asset <select id="asset"><option>BTC</option><option>ETH</option><option>XAUT</option></select></label><label>Timeframe <select id="tf"><option>5m</option><option>15m</option><option selected>1h</option><option>4h</option><option>1d</option></select></label><button type="button" id="clear">New chat</button></div>
<div class="chatbox"><div id="messages"><div class="msg assistant">Namaste! Main Delta AI Assistant hoon. BTC, ETH ya XAUT select karke market, trend, liquidity sweep, SMC/FVG, OI/PCR, entry/SL/targets ya backtest ke baare mein poochho. Main sirf read-only analysis deta hoon; koi order place nahi hota.</div></div>
<form id="form"><textarea id="prompt" placeholder="Market abhi kya bol raha hai?..." required></textarea><button id="send">Send</button></form><p id="status" class="muted">Assistant ready · API responses may use paid OpenAI API credits.</p></div>
<p class="warn">Risk note: signals probabilistic hain, guaranteed nahi. Assistant live market data unavailable hone par clearly batayega. PAPER_ONLY — real trading disabled.</p></main>
<script>
const messages=document.getElementById('messages'),form=document.getElementById('form'),promptBox=document.getElementById('prompt'),statusBox=document.getElementById('status'),send=document.getElementById('send');
let history=[];
function add(text,role){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;messages.appendChild(d);messages.scrollTop=messages.scrollHeight}
document.getElementById('clear').onclick=()=>{history=[];messages.innerHTML='';add('Nayi chat shuru. Kya analyze karna hai?','assistant');statusBox.textContent='New chat';};
form.addEventListener('submit',async e=>{e.preventDefault();const message=promptBox.value.trim();if(!message)return;const asset=document.getElementById('asset').value, timeframe=document.getElementById('tf').value;
add(message,'user');promptBox.value='';send.disabled=true;statusBox.textContent='Live data aur AI analysis load ho raha hai…';
try{const r=await fetch('/assistant/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,history,asset,timeframe})});const d=await r.json();if(!r.ok)throw Error(d.detail||('HTTP '+r.status));add(d.answer,'assistant');history.push({role:'user',content:message},{role:'assistant',content:d.answer});history=history.slice(-12);statusBox.textContent='Analysis complete · '+(d.context_status||'market context checked')+' · PAPER_ONLY';}
catch(err){add('Maaf kijiye, assistant response nahi aa paya: '+err.message,'assistant');statusBox.textContent='Request failed';}
finally{send.disabled=false;promptBox.focus();}});
</script></body></html>""")

@router.post("/assistant/chat")
def assistant_chat(body: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY Render Environment mein set nahi hai.")
    # Pull context from the bot's own exchange-data and paper-analysis functions.
    import main as bot
    context = {"asset": body.asset, "timeframe": body.timeframe, "mode": "PAPER_ONLY"}
    context_status = []
    for label, fn in [
        ("ticker", lambda: bot.market_tickers(body.asset)),
        ("signal", lambda: bot.paper_signal(body.asset, body.timeframe if body.timeframe in ("5m","15m","1h","4h","1d") else "1h")),
        ("options", lambda: bot.options_analytics(body.asset)),
        ("multi_timeframe", lambda: bot.multi_timeframe_outlook(body.asset, 180)),
    ]:
        try:
            value = fn()
            if label == "ticker":
                rows = value if isinstance(value, list) else value.get("result", []) if isinstance(value, dict) else []
                context[label] = rows[:8]
            else:
                context[label] = value
            context_status.append(label + ":OK")
        except Exception as exc:
            context[label] = {"available": False, "reason": str(getattr(exc, "detail", exc))[:180]}
            context_status.append(label + ":UNAVAILABLE")
    instructions = (
        "You are Delta AI Assistant inside a cryptocurrency market research dashboard. "
        "Speak clear Hindi/Hinglish by default; answer the user's language. "
        "Use the supplied JSON market context as the source of truth. Do not invent prices, timestamps, OI, PCR, Greeks, candles, news or indicator values. "
        "If a field is missing or an API failed, say unavailable. Clearly distinguish observed data from inference. "
        "Explain multi-timeframe trend, momentum, liquidity sweep, BOS/structure, FVG, RSI/EMA and options OI/PCR only when supported by supplied data. "
        "If asked for entry/stop/targets, provide an illustrative PAPER_ONLY setup only when the context supports it; otherwise say WAIT/insufficient data and explain what is missing. "
        "Never promise 90-95% accuracy, guaranteed profit or certainty. Mention invalidation and risk/reward caveats. "
        "You are strictly read-only: never place, recommend automatic execution of, or claim to have placed orders. "
        "The bot is PAPER_ONLY and has no live order execution. Keep answers practical and concise."
    )
    conversation = []
    for item in body.history[-10:]:
        conversation.append({"role": item.role, "content": item.content})
    conversation.append({"role": "user", "content": "LIVE BOT CONTEXT JSON:\n" + __import__("json").dumps(context, ensure_ascii=False, default=str) + "\n\nUSER QUESTION:\n" + body.message})
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            json={"model": os.getenv("OPENAI_MODEL", "gpt-5.5"), "instructions": instructions,
                  "input": conversation, "store": False, "max_output_tokens": 900},
            timeout=60,
        )
        if response.status_code >= 400:
            log.error("OpenAI API error %s: %s", response.status_code, response.text[:500])
            raise HTTPException(status_code=502, detail="OpenAI API request failed. Render logs mein status check karein; API key expose nahi ki gayi.")
        payload = response.json()
        parts = []
        for item in payload.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        parts.append(part.get("text", ""))
        answer = "\n".join(x for x in parts if x).strip()
        if not answer:
            raise HTTPException(status_code=502, detail="AI ne empty response diya. Dobara try karein.")
        return {"answer": answer, "context_status": ", ".join(context_status), "mode": "PAPER_ONLY"}
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="AI response timeout. Thodi der baad dobara try karein.")
    except requests.RequestException:
        log.exception("OpenAI API connection failed")
        raise HTTPException(status_code=502, detail="OpenAI API se connection nahi ho paya.")
