# Deribit Volatility Service

This project provides a small Flask application that fetches option data from the [Deribit](https://www.deribit.com/) cryptocurrency derivatives exchange and exposes a simple API for retrieving implied volatility.

## Deribit API usage

The application communicates with Deribit using a handful of public REST endpoints:

- `https://www.deribit.com/api/v2/public/get_index_price?index_name=<asset>_usd` – retrieves the current index price for the requested asset.
- `https://www.deribit.com/api/v2/public/get_instruments?currency=any&kind=option&expired=false` – lists available option instruments so the service can filter for near-the-money contracts.
- `https://www.deribit.com/api/v2/public/ticker?instrument_name=<instrument>` – returns details for a specific option, including the implied volatility value used in calculations.

These URLs are combined using the base URL `https://www.deribit.com/api/v2/` defined in `app.py`.

## Running the service

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python app.py
   ```

## API endpoints

- `GET /` – health check message.
- `GET /volatility?asset=BTC` – returns cached price and volatility data for `BTC`, `ETH`, `SOL`, or `XAU` (mapped to PAXG).

The service periodically updates the cache with fresh data from Deribit and serves results from memory.

