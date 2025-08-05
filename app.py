import requests
import datetime
import numpy as np
from flask import Flask, jsonify, request
from threading import Thread
import time
import os

app = Flask(__name__)

BASE_URL = "https://www.deribit.com/api/v2/"
ASSETS = ["BTC", "ETH", "PAXG", "SOL"]
CACHE = {}

# map query aliases to cache keys
# When a user queries XAU, return the cached PAXG data
QUERY_ALIASES = {"XAU": "PAXG"}

def cache_key(asset: str) -> str:
    """Return the key used for caching a given asset."""
    # Keep the original asset name in the cache map
    return asset.upper()
def get_asset_price(asset="BTC"):
    response = requests.get(BASE_URL + f"public/get_index_price?index_name={asset.lower()}_usd")
    data = response.json()
    return data['result']['index_price']
    
def get_price(asset):
    return get_asset_price(asset)

def fetch_options_with_iv(asset, price):
    instruments_response = requests.get(BASE_URL + f"public/get_instruments?currency="
                                                   f"any&kind=option&expired=false")
    instruments_data = instruments_response.json()
    now = datetime.datetime.now(datetime.timezone.utc)
    strike_range = (0.9 * price, 1.1 * price)
    options = []
    max_time_gold = 604800 
    max_time_all = 172800
    max_time = max_time_gold if asset == "PAXG" else max_time_all
    OPTIONS_MIN_LIFE_DAYS = 0.5

    for instr in instruments_data['result']:
        if instr['base_currency'] != asset.upper():
            continue
            
        expiry = datetime.datetime.fromtimestamp(instr['expiration_timestamp'] / 1000, datetime.timezone.utc)
        creation = datetime.datetime.fromtimestamp(instr['creation_timestamp'] / 1000, datetime.timezone.utc)
        if (expiry - now).total_seconds() <= max_time and (expiry - now).total_seconds() >= 12 * 3600 
            and ((now - creation).total_seconds() >= OPTIONS_MIN_LIFE_DAYS * 86400):
            strike = float(instr['strike'])
            if strike_range[0] <= strike <= strike_range[1]:
                ticker_response = requests.get(BASE_URL + f"public/ticker?instrument_name={instr['instrument_name']}")
                ticker_data = ticker_response.json()
                mark_iv = ticker_data['result'].get('mark_iv', 0.0)
                if mark_iv > 0:
                    instr['iv'] = mark_iv / 100
                    instr['option_type'] = 'call' if instr['instrument_name'][-1] == 'C' else 'put'
                    options.append(instr)

    return options

def get_atm_iv(options, price):
    expiry_groups = {}
    for opt in options:
        expiry_groups.setdefault(opt['expiration_timestamp'], []).append(opt)

    now = datetime.datetime.utcnow().timestamp()
    daily_vols, expiry_times = [], []

    for expiry, opts in expiry_groups.items():
        strikes = np.array([o['strike'] for o in opts])
        closest_strike_idx = (np.abs(strikes - price)).argmin()
        atm_strike = strikes[closest_strike_idx]
        call_iv = next((o['iv'] for o in opts if o['strike'] == atm_strike and o['option_type'] == 'call'), None)
        put_iv = next((o['iv'] for o in opts if o['strike'] == atm_strike and o['option_type'] == 'put'), None)

        if call_iv is None or put_iv is None:
            continue

        avg_iv = (call_iv + put_iv) / 2
        T_days = (expiry / 1000 - now) / 86400
        extrapolated_iv = avg_iv * np.sqrt(1 / T_days)
        daily_vols.append(extrapolated_iv)
        expiry_times.append(T_days)

    return daily_vols, expiry_times

def calculate_volatility(daily_vols, expiry_times):
    distances = [abs(t - 1) for t in expiry_times]
    inv_distances = [1/d if d != 0 else 0 for d in distances]
    weighted_avg = np.average(daily_vols, weights=inv_distances)
    simple_avg = np.mean(daily_vols)
    return simple_avg/365**0.5, weighted_avg/365**0.5

def update_cache():
    while True:
        for asset in ASSETS:
            print(asset)
            try:
                price = get_price(asset)
                options = fetch_options_with_iv(asset, price)
                if not options:
                    continue
                daily_vols, expiry_times = get_atm_iv(options, price)
                if daily_vols:
                    simple, weighted = calculate_volatility(daily_vols, expiry_times)
                    key = cache_key(asset)
                    CACHE[key] = {
                        "price": price,
                        "simple_avg_vol": round(simple, 6),
                        "weighted_avg_vol": round(weighted, 6),
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
            except Exception as e:
                print(f"Error updating {asset}: {e}")
        time.sleep(300)

@app.route("/")
def index():
    return jsonify({
        "message": "Use /volatility?asset=BTC, ETH, SOL or asset=XAU (PAXG) to get volatility data"
    })

@app.route("/volatility")
def volatility():
    asset = request.args.get("asset", "BTC").upper()
    asset = QUERY_ALIASES.get(asset, asset)
    if asset not in CACHE:
        return jsonify({"error": "Asset not found or not yet cached"}), 400
    return jsonify(CACHE[asset])

Thread(target=update_cache, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
