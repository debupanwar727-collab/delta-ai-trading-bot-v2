# Delta AI Trading Bot V2

A read-only BTC/ETH market research API for Delta Exchange India. It starts in **PAPER_ONLY** mode and does not place orders.

## Deploy on Render
1. Create a new Web Service and connect this repository.
2. Runtime: Python; Build: `pip install -r requirements.txt`
3. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Health check: `/health`

A `render.yaml` blueprint is included.

## Endpoints
- `/` service status
- `/health` health check
- `/market/tickers?asset=BTC` BTC/ETH futures tickers
- `/market/options?asset=BTC&expiry_date=DD-MM-YYYY` option chain
- `/analytics/options?asset=BTC&expiry_date=DD-MM-YYYY` OI PCR and estimated max pain
- `/market/candles?asset=BTC&resolution=1h&days=180` historical OHLC (up to 6 months)
- `/signals/BTC?resolution=1h` baseline EMA/RSI paper signal
- `/backtest/BTC?resolution=1h&days=180` basic EMA crossover historical test
- `/docs` interactive API docs

## Limitations and safety
- Public Delta endpoints are used; no API key is required for these market-data calls.
- Greeks/IV are not fabricated. Raw option-chain ticker fields are returned by Delta; analytics use available strike and OI fields.
- The backtest is a simple baseline, not a validated strategy. It excludes fees, spread, slippage, funding and execution delays.
- No live order placement, account access, or automatic trading is implemented.
- Six-month history is requested in chunks because Delta documents a maximum of 2,000 candles per response. Actual returned history depends on product listing and API availability.

Always paper-test and independently verify data before making financial decisions.
